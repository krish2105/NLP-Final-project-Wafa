"""End-to-end orchestration: message -> NLP -> risk -> decision -> outreach.

This is the single function the dashboard's Live Console calls. It stitches the
independently-tested modules together and returns every intermediate contract
object so the UI (and the acceptance test) can inspect the whole chain.
"""
from __future__ import annotations

from typing import Dict, Optional

from . import churn_model
from .data_loader import get_customer_row
from .decision_engine import decide
from .fusion import fuse_risk
from .nlp_pipeline import process_message
from .outreach_generator import generate_outreach


def _fallback_customer(customer_id: str) -> Dict:
    """If a customer_id isn't in customers.csv, use a neutral Mass-segment row
    so the demo never crashes on free-text input."""
    return {
        "customer_id": customer_id,
        "nationality_region": "Unknown",
        "tenure_months": 24,
        "segment": "Mass",
        "products_held": 2,
        "avg_balance_aed": 10000.0,
        "balance_trend_3m": 0.0,
        "salary_credit_active": True,
        "remittance_count_3m": 2,
        "intl_transfer_spike": False,
        "complaints_6m": 0,
        "branch_visits_trend": 0.0,
        "clv_estimate_aed": 10000.0,
    }


def run_pipeline(
    text: str,
    customer_id: str,
    message_id: Optional[str] = None,
    language: Optional[str] = None,
    use_llm: bool = True,
    match_language: bool = False,
) -> Dict:
    customer_row = get_customer_row(customer_id) or _fallback_customer(customer_id)

    nlp_output = process_message(text, customer_id, message_id=message_id, language=language)

    tabular_prob = churn_model.predict_proba(customer_row)
    fused = fuse_risk(tabular_prob, nlp_output["churn_signal"], customer_row)

    customer_risk = {
        "customer_id": str(customer_id),
        "tabular_churn_probability": round(float(tabular_prob), 4),
        "top_drivers": churn_model.top_drivers(customer_row),
        "customer_segment": customer_row.get("segment", "Mass"),
        "clv_estimate_aed": float(customer_row.get("clv_estimate_aed", 0.0)),
        "fairness_group": customer_row.get("nationality_region", "Unknown"),
    }

    decision = decide(nlp_output, fused, customer_row)
    outreach = generate_outreach(nlp_output, decision, customer_row, use_llm=use_llm,
                                 match_language=match_language)

    return {
        "customer_row": customer_row,
        "nlp_output": nlp_output,
        "customer_risk": customer_risk,
        "fused_risk": fused,
        "decision": decision,
        "outreach": outreach,
        "tabular_churn_probability": round(float(tabular_prob), 4),
    }
