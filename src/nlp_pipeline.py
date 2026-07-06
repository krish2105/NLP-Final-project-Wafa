"""LISTEN: turn a raw message into a structured NLPOutput.

Flow:  detect language -> translate to English (optional) -> classify
       issue_type + churn_signal -> extract entities -> assemble contract.

Runtime classifier preference:
  1. fine-tuned DistilBERT (if models/ contains it)
  2. TF-IDF + LogReg baseline (always available after training)
  3. keyword heuristic (last-resort so the pipeline never hard-fails)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional

from . import config
from .contracts import validate_nlp_output
from .entity_extraction import extract_entities, extract_entities_multi
from .translation import translate_to_english

logger = logging.getLogger("wafa.nlp")


# --------------------------------------------------------------------------- #
# Language detection (langdetect optional; simple script heuristic fallback)
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    if not text.strip():
        return "en"
    # Arabic script?
    if any("؀" <= ch <= "ۿ" for ch in text):
        return "ar"
    # Devanagari (Hindi) script?
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return "hi"
    try:
        from langdetect import detect

        code = detect(text)
        return code if code in config.LANGUAGES else "en"
    except Exception:
        return "en"


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _load_distilbert(kind: str):
    """kind in {'issue','churn'}. Returns (tokenizer, model) or None."""
    if config.WAFA_LIGHT_MODE:
        return None
    path = config.DISTILBERT_ISSUE_DIR if kind == "issue" else config.DISTILBERT_CHURN_DIR
    if not path.exists():
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(path))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(path))
        mdl.eval()
        return tok, mdl
    except Exception as e:
        logger.warning("DistilBERT %s load failed: %s", kind, e)
        return None


@lru_cache(maxsize=2)
def _load_tfidf(kind: str):
    path = config.TFIDF_ISSUE_PATH if kind == "issue" else config.TFIDF_CHURN_PATH
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def _classify_distilbert(bundle, text: str):
    import torch

    tok, mdl = bundle
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = mdl(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    label = mdl.config.id2label[idx]
    return label, float(probs[idx])


def _classify_tfidf(pipe, text: str):
    proba = pipe.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = pipe.classes_[idx]
    return str(label), float(proba[idx])


# keyword heuristic (only if nothing trained is present)
_ISSUE_KEYWORDS = {
    "Account_Closure": ["close", "closing", "terminate account"],
    "Remittance_Transfer": ["remit", "transfer", "send money"],
    "Loan_Mortgage": ["loan", "mortgage"],
    "Fees_Charges": ["fee", "charge", "charged"],
    "Card_Services": ["card"],
    "App_Technical": ["app", "login", "otp", "crash"],
    "General_Query": ["timing", "hours", "how", "update"],
}


def _classify_heuristic_issue(text: str):
    low = text.lower()
    for label, kws in _ISSUE_KEYWORDS.items():
        if any(k in low for k in kws):
            return label, 0.4
    return "General_Query", 0.3


def _classify_heuristic_churn(text: str, entities: dict):
    if entities["leaving_uae"] or entities["account_closure_intent"]:
        return "High", 0.5
    low = text.lower()
    if any(w in low for w in ["angry", "frustrat", "worst", "close", "leave", "complaint"]):
        return "Medium", 0.4
    return "Low", 0.4


def classify_issue(text: str):
    bundle = _load_distilbert("issue")
    if bundle:
        return _classify_distilbert(bundle, text)
    pipe = _load_tfidf("issue")
    if pipe is not None:
        return _classify_tfidf(pipe, text)
    return _classify_heuristic_issue(text)


def classify_churn(text: str, entities: dict):
    bundle = _load_distilbert("churn")
    if bundle:
        return _classify_distilbert(bundle, text)
    pipe = _load_tfidf("churn")
    if pipe is not None:
        return _classify_tfidf(pipe, text)
    return _classify_heuristic_churn(text, entities)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def process_message(
    text: str,
    customer_id: str,
    message_id: Optional[str] = None,
    language: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict:
    lang = language or detect_language(text)
    translated_text, translated = translate_to_english(text, lang)
    clf_text = translated_text if translated_text else text

    # Extract entities from BOTH the original and the (possibly translated) text so
    # native-script keywords and English regex both get a chance (translation can
    # drop a clause or render "UAE" as "the Emirates").
    entities = extract_entities_multi(text, clf_text)
    issue_type, issue_conf = classify_issue(clf_text)
    churn_signal, churn_conf = classify_churn(clf_text, entities)

    # Stretch 1.5: low confidence -> flag for human review before decisioning.
    needs_review = (
        issue_conf < config.LOW_CONFIDENCE_THRESHOLD
        or churn_conf < config.LOW_CONFIDENCE_THRESHOLD
    )

    out = {
        "message_id": str(message_id) if message_id is not None else None,
        "customer_id": str(customer_id),
        "language": lang,
        "translated_text": translated_text if translated else None,
        "issue_type": issue_type,
        "issue_confidence": round(float(issue_conf), 4),
        "churn_signal": churn_signal,
        "churn_confidence": round(float(churn_conf), 4),
        "entities": entities,
        "low_confidence_flag": bool(needs_review),
    }
    return validate_nlp_output(out)
