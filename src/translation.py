"""Translate-then-classify support (primary multilingual route).

Arabic  -> Helsinki-NLP/opus-mt-ar-en
Hindi   -> Helsinki-NLP/opus-mt-hi-en
Tagalog -> facebook/nllb-200-distilled-600M (tl_Latn -> eng_Latn)  [Tagalog only]
English -> passthrough

Everything here is OPTIONAL and lazy. If transformers is unavailable, or a
model fails to download (offline laptop), we fall back to returning the original
text and flag `translated=False`. The rest of the pipeline still works because
the entity keywords include romanised/translated equivalents and the classifier
was trained on English text (see note in the Architecture doc: accuracy on
untranslated non-English text is a documented limitation).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

from . import config

logger = logging.getLogger("wafa.translation")

_NLLB_LANG = {"tl": "tl_Latn"}


@lru_cache(maxsize=4)
def _load_opus(lang: str):
    from transformers import MarianMTModel, MarianTokenizer

    name = config.OPUS_MT_MODELS[lang]
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name)
    return tok, mdl


@lru_cache(maxsize=1)
def _load_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL)
    return tok, mdl


def _transformers_available() -> bool:
    if config.WAFA_LIGHT_MODE:
        return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def translate_to_english(text: str, language: str) -> Tuple[str, bool]:
    """Return (english_text, translated_flag)."""
    if not text or language == "en":
        return text, language == "en"

    if not _transformers_available():
        return text, False

    try:
        import torch

        if language in config.OPUS_MT_MODELS:
            tok, mdl = _load_opus(language)
            batch = tok([text], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                gen = mdl.generate(**batch, max_new_tokens=200)
            return tok.batch_decode(gen, skip_special_tokens=True)[0], True

        if language == "tl":
            tok, mdl = _load_nllb()
            tok.src_lang = _NLLB_LANG["tl"]
            batch = tok(text, return_tensors="pt", truncation=True)
            bos = tok.convert_tokens_to_ids("eng_Latn")
            with torch.no_grad():
                gen = mdl.generate(**batch, forced_bos_token_id=bos, max_new_tokens=200)
            return tok.batch_decode(gen, skip_special_tokens=True)[0], True
    except Exception as e:
        logger.warning("Translation failed for lang=%s (%s) -> passthrough.", language, e)

    return text, False
