"""Domain entity extraction.

Combines spaCy NER (MONEY/GPE/DATE on translated English text) with regex +
keyword dictionaries for banking-specific signals. spaCy is OPTIONAL — if
`en_core_web_sm` is not installed the regex/keyword layer still produces every
field in the contract, so the pipeline never breaks.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Optional spaCy
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Keyword dictionaries
# --------------------------------------------------------------------------- #
PRODUCT_KEYWORDS = {
    "salary account": ["salary account", "salary credit"],
    "account": ["account", "savings", "current account"],
    "loan": ["loan", "personal loan"],
    "mortgage": ["mortgage", "home loan"],
    "card": ["card", "credit card", "debit card"],
    "app": ["app", "application", "mobile banking", "online banking"],
    "remittance": ["remittance", "transfer", "remit", "send money", "wire"],
    "Emirates ID": ["emirates id", "eid"],
}

DESTINATION_KEYWORDS = {
    "India": ["india", "indian"],
    "Philippines": ["philippines", "manila", "filipino"],
    "home country": ["home country", "back home", "my country"],
    "abroad": ["abroad", "overseas", "another country"],
    "UAE": ["uae", "dubai", "abu dhabi", "sharjah", "emirates"],
}

# Leaving-UAE phrases, incl. translated equivalents of ar/hi/tl expressions.
LEAVING_PATTERNS = [
    r"leaving\s+(?:the\s+)?(?:uae|emirates|country)",
    r"leaving\s+(?:the\s+)?country",
    r"relocat",
    r"resign(?:ed|ing)?",
    r"moving\s+back",
    r"move\s+back",
    r"going\s+back\s+home",
    r"final\s+month",
    r"last\s+month\s+here",
    r"permanent(?:ly)?\s+leav",
    r"for\s+good",
    r"repatriat",
    r"end\s+of\s+service",
]

ACCOUNT_CLOSURE_PATTERNS = [
    r"close\s+(?:my\s+)?(?:the\s+)?(?:salary\s+|savings\s+|current\s+)?account",
    r"account\s+clos",
    r"closing\s+(?:my\s+)?account",
    r"terminate\s+(?:my\s+)?account",
    r"cancel\s+(?:my\s+)?account",
    r"shut\s+(?:down\s+)?(?:my\s+)?account",
]

# Native-script + romanised equivalents so intent is caught even when NO
# translation model is loaded (offline laptop). Checked as plain substrings.
MULTILINGUAL_LEAVING = [
    # Arabic: leaving / relocating / resigned / going back
    "سأغادر", "أغادر", "مغادرة", "سأترك", "الانتقال", "استقلت", "الاستقالة", "العودة إلى بلدي",
    # Hindi (Devanagari): leaving / going back / resigned
    "छोड़ रहा", "वापस जा", "इस्तीफा", "देश वापस", "स्थानांतर",
    # Romanised Hindi
    "chhod raha", "wapas ja", "istifa", "desh wapas", "relocate ho",
    # Tagalog
    "aalis na", "lilipat", "uuwi", "umuwi", "nagbitiw", "babalik na ako",
]
MULTILINGUAL_CLOSURE = [
    # Arabic: close my account
    "أغلق حساب", "اغلاق حساب", "إغلاق الحساب", "أغلق الحساب", "اغلق حسابي",
    # Hindi
    "खाता बंद", "अकाउंट बंद", "खाता बन्द",
    # Romanised Hindi
    "khata band", "account band", "khaata band",
    # Tagalog
    "isara ang account", "isasara ang account", "isara ang aking account", "isara ko ang",
]

_AED_AMOUNT_RE = re.compile(r"(?:aed|dhs?|dirhams?)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_BARE_AMOUNT_RE = re.compile(r"\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b")
_MONTH_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{0,4}\b",
    re.IGNORECASE,
)


def _keyword_hits(text_low: str, mapping: Dict[str, List[str]]) -> List[str]:
    out = []
    for canonical, variants in mapping.items():
        if any(v in text_low for v in variants):
            out.append(canonical)
    return out


def _any_pattern(text_low: str, patterns: List[str]) -> bool:
    return any(re.search(p, text_low) for p in patterns)


def extract_entities(text: str) -> Dict:
    """Return the `entities` sub-dict of NLPOutput for the given (English) text."""
    text = text or ""
    low = text.lower()

    # --- amounts ---
    amounts: List[str] = []
    for m in _AED_AMOUNT_RE.finditer(text):
        amounts.append(f"AED {m.group(1)}")
    for m in _BARE_AMOUNT_RE.finditer(text):
        val = m.group(1)
        if f"AED {val}" not in amounts:
            amounts.append(val)

    # --- dates ---
    dates: List[str] = [m.group(0).strip() for m in _MONTH_RE.finditer(text)]

    # --- spaCy augmentation (optional) ---
    nlp = _load_spacy()
    destinations_spacy: List[str] = []
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "MONEY" and ent.text not in amounts:
                    amounts.append(ent.text)
                elif ent.label_ == "DATE" and ent.text not in dates:
                    dates.append(ent.text)
                elif ent.label_ == "GPE":
                    destinations_spacy.append(ent.text)
        except Exception:
            pass

    products = _keyword_hits(low, PRODUCT_KEYWORDS)
    destinations = _keyword_hits(low, DESTINATION_KEYWORDS)
    for g in destinations_spacy:
        if g not in destinations:
            destinations.append(g)

    # regex on (translated) English + substring checks on native/romanised text
    leaving_uae = _any_pattern(low, LEAVING_PATTERNS) or any(k in text for k in MULTILINGUAL_LEAVING)
    account_closure_intent = _any_pattern(low, ACCOUNT_CLOSURE_PATTERNS) or any(
        k in text for k in MULTILINGUAL_CLOSURE
    )

    return {
        "amounts": _dedup(amounts),
        "dates": _dedup(dates),
        "products": _dedup(products),
        "destinations": _dedup(destinations),
        "leaving_uae": bool(leaving_uae),
        "account_closure_intent": bool(account_closure_intent),
    }


def extract_entities_multi(*texts: str) -> Dict:
    """Extract entities from several representations of the same message (e.g. the
    original non-English text AND its English translation) and merge them.

    Translation can drop a clause (opus-mt sometimes translates only one sentence)
    and native keywords won't match English output — so we OR the boolean intents
    and union the list fields across every representation. This makes leaver /
    closure detection robust whether or not a translation model is loaded.
    """
    parts = [extract_entities(t) for t in texts if t]
    if not parts:
        return extract_entities("")
    merged = {
        "amounts": [], "dates": [], "products": [], "destinations": [],
        "leaving_uae": False, "account_closure_intent": False,
    }
    for p in parts:
        for lst in ("amounts", "dates", "products", "destinations"):
            merged[lst].extend(p[lst])
        merged["leaving_uae"] = merged["leaving_uae"] or p["leaving_uae"]
        merged["account_closure_intent"] = (
            merged["account_closure_intent"] or p["account_closure_intent"]
        )
    for lst in ("amounts", "dates", "products", "destinations"):
        merged[lst] = _dedup(merged[lst])
    return merged


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out
