"""Frozen interface contracts for Project Wafa.

These are the *shapes* every module must produce/consume. They are frozen
BEFORE any model exists so mock tests can validate the plumbing first
(build order steps 1 & 11). We use light-weight validators (plain functions +
optional pydantic) rather than forcing pydantic everywhere, so the contracts
stay readable and the modules stay decoupled.
"""
from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Canonical example objects (also serve as documentation).
# --------------------------------------------------------------------------- #
RAW_MESSAGE_INPUT_EXAMPLE: Dict[str, Any] = {
    "message_id": "M0007",
    "customer_id": "FB1186",
    "text": "how do I close my salary account. I have resigned and am moving back home.",
    "timestamp": None,
}

NLP_OUTPUT_EXAMPLE: Dict[str, Any] = {
    "message_id": "M0007",
    "customer_id": "FB1186",
    "language": "en",
    "translated_text": "how do I close my salary account...",
    "issue_type": "Account_Closure",
    "issue_confidence": 0.91,
    "churn_signal": "High",
    "churn_confidence": 0.88,
    "entities": {
        "amounts": [],
        "dates": [],
        "products": ["salary account"],
        "destinations": ["home country"],
        "leaving_uae": True,
        "account_closure_intent": True,
    },
}

CUSTOMER_RISK_OUTPUT_EXAMPLE: Dict[str, Any] = {
    "customer_id": "FB1186",
    "tabular_churn_probability": 0.72,
    "top_drivers": ["balance_trend_3m", "salary_credit_active", "complaints_6m"],
    "customer_segment": "Premium",
    "clv_estimate_aed": 99627.0,
    "fairness_group": "South_Asia",
}

FUSED_RISK_OUTPUT_EXAMPLE: Dict[str, Any] = {
    "customer_id": "FB1186",
    "final_risk_score": 0.86,
    "risk_band": "High",
    "risk_reasons": ["salary credit stopped", "high CLV customer"],
    "text_score": 0.9,
    "behavior_score": 0.72,
}

DECISION_OUTPUT_EXAMPLE: Dict[str, Any] = {
    "action": "Dignified Goodbye Pathway",
    "action_reason": "Customer has confirmed relocation/departure intent.",
    "offer_allowed": False,
    "max_offer_value_aed": 0.0,
    "requires_human_review": True,
    "dignified_goodbye": True,
}

OUTREACH_OUTPUT_EXAMPLE: Dict[str, Any] = {
    "draft_language": "en",
    "draft_text": "Thank you for banking with Falcon Bank...",
    "guardrail_passed": True,
    "guardrail_warnings": [],
    "human_status": "Pending Review",
}

AUDIT_LOG_RECORD_EXAMPLE: Dict[str, Any] = {
    "timestamp": "2026-07-06T12:00:00",
    "message_id": "M0007",
    "customer_id": "FB1186",
    "issue_type": "Account_Closure",
    "churn_signal": "High",
    "tabular_churn_probability": 0.72,
    "final_risk_score": 0.86,
    "risk_band": "High",
    "recommended_action": "Dignified Goodbye Pathway",
    "draft_text": "Thank you for banking with Falcon Bank...",
    "human_decision": "Pending Review",
    "override_reason": None,
}

# --------------------------------------------------------------------------- #
# Validators. Each raises AssertionError with a helpful message on a bad shape.
# --------------------------------------------------------------------------- #
def _require_keys(d: Dict[str, Any], keys: List[str], name: str) -> None:
    assert isinstance(d, dict), f"{name} must be a dict, got {type(d)}"
    missing = [k for k in keys if k not in d]
    assert not missing, f"{name} missing keys: {missing}"


def validate_nlp_output(d: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        d,
        [
            "message_id",
            "customer_id",
            "language",
            "translated_text",
            "issue_type",
            "issue_confidence",
            "churn_signal",
            "churn_confidence",
            "entities",
        ],
        "NLPOutput",
    )
    _require_keys(
        d["entities"],
        [
            "amounts",
            "dates",
            "products",
            "destinations",
            "leaving_uae",
            "account_closure_intent",
        ],
        "NLPOutput.entities",
    )
    assert d["churn_signal"] in ("Low", "Medium", "High"), "bad churn_signal"
    assert 0.0 <= float(d["issue_confidence"]) <= 1.0, "issue_confidence out of range"
    assert 0.0 <= float(d["churn_confidence"]) <= 1.0, "churn_confidence out of range"
    assert isinstance(d["entities"]["leaving_uae"], bool)
    assert isinstance(d["entities"]["account_closure_intent"], bool)
    return d


def validate_fused_risk(d: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        d,
        [
            "customer_id",
            "final_risk_score",
            "risk_band",
            "risk_reasons",
            "text_score",
            "behavior_score",
        ],
        "FusedRiskOutput",
    )
    assert d["risk_band"] in ("Low", "Medium", "High"), "bad risk_band"
    assert 0.0 <= float(d["final_risk_score"]) <= 1.0, "final_risk_score out of range"
    return d


def validate_decision(d: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        d,
        [
            "action",
            "action_reason",
            "offer_allowed",
            "max_offer_value_aed",
            "requires_human_review",
            "dignified_goodbye",
        ],
        "DecisionOutput",
    )
    assert d["requires_human_review"] is True, "requires_human_review must always be True"
    assert isinstance(d["dignified_goodbye"], bool)
    if not d["offer_allowed"]:
        assert d["max_offer_value_aed"] == 0.0, "offer not allowed but value != 0"
    return d


def validate_outreach(d: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        d,
        ["draft_language", "draft_text", "guardrail_passed", "guardrail_warnings", "human_status"],
        "OutreachOutput",
    )
    assert d["human_status"] == "Pending Review", "human_status must start as 'Pending Review'"
    return d


def validate_audit_record(d: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(d, list(AUDIT_LOG_RECORD_EXAMPLE.keys()), "AuditLogRecord")
    return d
