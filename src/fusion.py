"""Fusion layer: combine tabular churn probability with the text churn signal
into a single, explainable risk view.

This is intentionally a plain, readable function (no ML) — the weights and
boosts are a transparent policy that a human can audit line by line.
"""
from __future__ import annotations

from typing import Dict

from .utils import as_bool, safe_float


def text_signal_to_score(churn_signal: str) -> float:
    return {"Low": 0.2, "Medium": 0.55, "High": 0.9}[churn_signal]


def fuse_risk(tabular_probability: float, churn_signal: str, customer_row: dict) -> Dict:
    text_score = text_signal_to_score(churn_signal)

    boost = 0.0
    reasons = []
    if not as_bool(customer_row["salary_credit_active"]):
        boost += 0.10
        reasons.append("salary credit stopped")
    if as_bool(customer_row["intl_transfer_spike"]):
        boost += 0.10
        reasons.append("international transfer spike")
    if safe_float(customer_row["balance_trend_3m"]) < -0.25:
        boost += 0.10
        reasons.append("balance declining sharply")
    if int(customer_row["complaints_6m"]) >= 4:
        boost += 0.05
        reasons.append("repeated complaints")

    final_score = min(1.0, 0.45 * float(tabular_probability) + 0.40 * text_score + boost)

    if final_score >= 0.75:
        band = "High"
    elif final_score >= 0.45:
        band = "Medium"
    else:
        band = "Low"

    if churn_signal == "High":
        reasons.append("customer explicitly signaling high churn intent in message")
    if safe_float(customer_row["clv_estimate_aed"]) > 20000:
        reasons.append("high CLV customer")

    return {
        "customer_id": str(customer_row.get("customer_id", "")),
        "final_risk_score": round(final_score, 3),
        "risk_band": band,
        "risk_reasons": reasons,
        "text_score": text_score,
        "behavior_score": round(float(tabular_probability), 3),
    }
