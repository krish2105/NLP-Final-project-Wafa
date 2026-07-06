"""Hallucination / dark-pattern guardrails for generated outreach drafts.

`check_draft` NEVER rewrites or suppresses a draft. It only inspects it and
returns a list of warnings for the human reviewer to see. Failing guardrails
does not block the draft from being shown — it blocks it from being treated as
safe-to-send without a human looking at the flags.
"""
from __future__ import annotations

import re
from typing import Dict, List

from . import config

# Phrases that are dark patterns / coercion — never allowed.
URGENCY_PATTERNS = [
    r"limited time",
    r"\bact now\b",
    r"\bexpires?\b",
    r"\bhurry\b",
    r"last chance",
    r"don'?t miss",
]
COERCION_PATTERNS = [
    r"you must stay",
    r"you have to stay",
    r"we require you to",
    r"obligated to (?:stay|remain)",
]
# Language that offers a retention incentive — forbidden on a dignified goodbye.
INCENTIVE_PATTERNS = [
    r"\bwaiv\w*",
    r"\bdiscount\b",
    r"\bcashback\b",
    r"\bbonus\b",
    r"\bspecial offer\b",
    r"\bincentive\b",
    r"\breward\b",
    r"\bfree (?:month|months|banking)\b",
    r"\bwe(?:'| a)re? offering\b",
]

_AMOUNT_RE = re.compile(r"(?:aed|dhs?|dirhams?)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_BARE_AMOUNT_RE = re.compile(r"\b([\d,]{3,}(?:\.\d+)?)\b")


def _find(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def _parse_amounts(text: str) -> List[float]:
    vals = []
    for m in _AMOUNT_RE.finditer(text):
        try:
            vals.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return vals


def check_draft(draft_text: str, decision: dict) -> Dict:
    """Return {"guardrail_passed": bool, "guardrail_warnings": [...]}."""
    warnings: List[str] = []
    text = draft_text or ""
    low = text.lower()

    # 1) Urgency / dark-pattern language
    for hit in _find(URGENCY_PATTERNS, low):
        warnings.append(f"Urgency/dark-pattern language matched: '{hit}'")

    # 2) Coercion language
    for hit in _find(COERCION_PATTERNS, low):
        warnings.append(f"Coercive language matched: '{hit}'")

    # 3) Any AED amount that does not match the approved max offer value
    max_offer = float(decision.get("max_offer_value_aed", 0.0) or 0.0)
    for amt in _parse_amounts(text):
        if abs(amt - max_offer) > 0.01:
            warnings.append(
                f"Draft mentions AED {amt:g} which does not match approved "
                f"max offer value AED {max_offer:g}."
            )

    # 4) Retention incentive present on a dignified goodbye
    if decision.get("dignified_goodbye"):
        for hit in _find(INCENTIVE_PATTERNS, low):
            warnings.append(
                f"Retention-incentive language '{hit}' present on a Dignified "
                f"Goodbye — must be a pressure-free farewell."
            )
        # a departing customer draft should not quote any offer value at all
        if _parse_amounts(text):
            warnings.append("Dignified Goodbye draft mentions a monetary amount — remove it.")

    # 5) Product names not on the allow-list
    allowed = set(p.lower() for p in config.ALLOWED_PRODUCTS)
    # products we tolerate seeing (allowed) — flag anything that looks like an
    # invented banking product ("platinum elite plan", "wealth booster", ...).
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:Plan|Account|Card|Bundle|Package|Booster|Saver))\b", text):
        phrase = m.group(1).lower()
        if not any(a in phrase for a in allowed):
            warnings.append(f"Possibly invented product name: '{m.group(1)}'")

    passed = len(warnings) == 0
    return {"guardrail_passed": passed, "guardrail_warnings": warnings}
