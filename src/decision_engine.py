"""Retention-action decision engine.

HARD REQUIREMENT: this layer is explicit, readable if/elif Python — never a
trained or black-box model. It decides what action a human should consider for
a customer, and it is defensible line by line. Nothing here ever *sends*
anything; `requires_human_review` is always True.
"""
from __future__ import annotations

from typing import Dict

from .utils import safe_float


def decide(nlp_output: dict, risk: dict, customer_row: dict) -> Dict:
    entities = nlp_output["entities"]

    # A "leaver" is someone who has confirmed they are departing/relocating, OR
    # who explicitly wants to close their account while signalling high churn.
    is_leaver = bool(
        entities["leaving_uae"]
        or (entities["account_closure_intent"] and nlp_output["churn_signal"] == "High")
    )

    if is_leaver:
        action = "Dignified Goodbye Pathway"
        reason = (
            "Customer has confirmed relocation/departure intent — "
            "smooth exit, no retention pressure."
        )
        dignified_goodbye = True
    elif risk["risk_band"] == "High" and customer_row["segment"] in ("Premium", "Private"):
        action = "Relationship Manager Call"
        reason = "High risk, high-value segment — warrants a personal outreach."
        dignified_goodbye = False
    elif nlp_output["issue_type"] == "Fees_Charges" and risk["risk_band"] in ("Medium", "High"):
        action = "Fee Waiver / Service Recovery"
        reason = "Fee complaint combined with elevated churn risk."
        dignified_goodbye = False
    elif nlp_output["issue_type"] == "Remittance_Transfer":
        action = "Remittance Support"
        reason = "Customer needs help with an international transfer."
        dignified_goodbye = False
    elif nlp_output["issue_type"] == "Loan_Mortgage":
        action = "Loan Settlement Advisory"
        reason = "Customer has a loan/mortgage question, possibly tied to relocation."
        dignified_goodbye = False
    elif nlp_output["issue_type"] in ("App_Technical", "Card_Services"):
        action = "Service Fix"
        reason = "Technical/card issue — resolve the friction directly."
        dignified_goodbye = False
    elif risk["risk_band"] == "Low" and nlp_output["issue_type"] == "General_Query":
        action = "Standard Information Reply"
        reason = "Low risk, routine informational query."
        dignified_goodbye = False
    else:
        action = "Retention Offer with Economics Check"
        reason = "Elevated risk without a more specific driver — standard retention offer applies."
        dignified_goodbye = False

    # ------------------------------------------------------------------ #
    # Offer economics check — never spend more than 5% of CLV, and never
    # spend on a low-CLV customer (service support is cheaper and fairer).
    # ------------------------------------------------------------------ #
    clv = safe_float(customer_row["clv_estimate_aed"])
    max_offer_value_aed = round(0.05 * clv, 2)
    offer_allowed = action in ("Fee Waiver / Service Recovery", "Retention Offer with Economics Check")
    if offer_allowed and clv < 5000:
        action = "Service Fix"  # low CLV -> service support, not an expensive discount
        reason += " (CLV too low to justify a financial incentive — service support recommended instead.)"
        offer_allowed = False

    return {
        "action": action,
        "action_reason": reason,
        "offer_allowed": offer_allowed,
        "max_offer_value_aed": max_offer_value_aed if offer_allowed else 0.0,
        "requires_human_review": True,
        "dignified_goodbye": dignified_goodbye,
    }
