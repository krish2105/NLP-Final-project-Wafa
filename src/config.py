"""Central configuration for Project Wafa.

Everything path- or constant-related lives here so the rest of the codebase
never hard-codes a path. Keeping this in one place also makes the whole system
runnable from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# src/ -> project_wafa/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
AUDIT_LOG_PATH = OUTPUTS_DIR / "audit_log.csv"

MESSAGES_CSV = DATA_DIR / "messages.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"

for _d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Label spaces
# --------------------------------------------------------------------------- #
ISSUE_TYPES = [
    "Account_Closure",
    "Remittance_Transfer",
    "Loan_Mortgage",
    "Fees_Charges",
    "Card_Services",
    "App_Technical",
    "General_Query",
]
CHURN_SIGNALS = ["Low", "Medium", "High"]
LANGUAGES = ["en", "ar", "hi", "tl"]
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic", "hi": "Hindi", "tl": "Tagalog"}

# --------------------------------------------------------------------------- #
# Churn model feature contract
# `nationality_region` is DELIBERATELY excluded from training features. It is
# only ever used afterwards in the fairness audit (see train_churn_model.py).
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "tenure_months",
    "products_held",
    "avg_balance_aed",
    "balance_trend_3m",
    "salary_credit_active",  # bool -> numeric via preprocessing
    "remittance_count_3m",
    "intl_transfer_spike",  # bool -> numeric via preprocessing
    "complaints_6m",
    "branch_visits_trend",
    "clv_estimate_aed",
]
CATEGORICAL_FEATURES = ["segment"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churned"
PROTECTED_ATTRIBUTE = "nationality_region"  # NEVER a training feature

# --------------------------------------------------------------------------- #
# Model artifact locations
# --------------------------------------------------------------------------- #
DISTILBERT_ISSUE_DIR = MODELS_DIR / "issue_classifier_distilbert"
DISTILBERT_CHURN_DIR = MODELS_DIR / "churn_signal_classifier_distilbert"
TFIDF_ISSUE_PATH = MODELS_DIR / "issue_classifier_tfidf_logreg.joblib"
TFIDF_CHURN_PATH = MODELS_DIR / "churn_signal_tfidf_logreg.joblib"
CHURN_MODEL_PATH = MODELS_DIR / "customer_churn_model.joblib"
PREPROCESSORS_PATH = MODELS_DIR / "preprocessors.joblib"

# --------------------------------------------------------------------------- #
# Translation model ids (free, CPU-runnable). Loaded lazily & optionally.
# --------------------------------------------------------------------------- #
OPUS_MT_MODELS = {
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"  # Tagalog only (tl_Latn->eng_Latn)
MULTILINGUAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --------------------------------------------------------------------------- #
# Outreach generation fallback chain (first that loads wins)
# --------------------------------------------------------------------------- #
# 0.5B first: laptop/CPU-friendly and strong Arabic for its size (brief's honest
# pick for a UAE scenario). 1.5B is a quality upgrade when a GPU is available;
# flan-t5-base is the light English fallback; rule-based templates are last resort.
OUTREACH_MODEL_CHAIN = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/flan-t5-base",
]

# --------------------------------------------------------------------------- #
# Confidence threshold below which a prediction is flagged for human review
# before the decision engine trusts it (Capability 1.5 stretch).
# --------------------------------------------------------------------------- #
LOW_CONFIDENCE_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Guardrails: only these product names may appear in an outreach draft.
# --------------------------------------------------------------------------- #
ALLOWED_PRODUCTS = ["account", "card", "loan", "mortgage", "remittance service", "app"]

# --------------------------------------------------------------------------- #
# Pinned demo messages for the Live Console (Capability: demo-ready cases).
# Chosen from messages.csv:
#   - M0007: English account-closure + resigned/moving back  -> LEAVER
#   - a high-churn non-English (Arabic) account-closure message
#   - a low-risk General_Query
# Resolved dynamically at import time so it survives dataset regeneration.
# --------------------------------------------------------------------------- #
def _resolve_demo_ids() -> list[str]:
    """Pick three representative message_ids for the demo picker."""
    fallback = ["M0007", "M0001", "M0002"]
    try:
        import pandas as pd

        m = pd.read_csv(MESSAGES_CSV)
    except Exception:
        return fallback

    ids: list[str] = []

    # 1) High-churn non-English account-closure / leaving-UAE case (prefer Arabic)
    non_en = m[(m.language == "ar") & (m.churn_signal == "High")]
    cand = non_en[non_en.issue_type == "Account_Closure"]
    pool = cand if len(cand) else non_en
    if len(pool):
        ids.append(str(pool.iloc[0].message_id))

    # 2) Genuine English leaver (account closure + high churn)
    leaver = m[(m.language == "en") & (m.issue_type == "Account_Closure") & (m.churn_signal == "High")]
    if len(leaver):
        ids.append(str(leaver.iloc[0].message_id))

    # 3) Routine low-risk General_Query
    routine = m[(m.issue_type == "General_Query") & (m.churn_signal == "Low")]
    if len(routine):
        ids.append(str(routine.iloc[0].message_id))

    # De-dup while preserving order, top up from fallback if short
    seen: set[str] = set()
    out: list[str] = []
    for i in ids + fallback:
        if i not in seen and i in set(m.message_id.astype(str)):
            seen.add(i)
            out.append(i)
        if len(out) == 3:
            break
    return out or fallback


try:
    DEMO_MESSAGE_IDS = _resolve_demo_ids()
except Exception:  # never let config import fail
    DEMO_MESSAGE_IDS = ["M0007", "M0001", "M0002"]

# Allow forcing "no heavy models" mode (CI / laptop / offline).
WAFA_LIGHT_MODE = os.environ.get("WAFA_LIGHT_MODE", "0") == "1"
