"""Mock tests for the decision engine — no models required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import validate_decision  # noqa: E402
from src.decision_engine import decide  # noqa: E402


def _nlp(issue="General_Query", churn="Low", leaving=False, closure=False):
    return {
        "message_id": "M1",
        "customer_id": "FB1",
        "issue_type": issue,
        "churn_signal": churn,
        "entities": {
            "amounts": [],
            "dates": [],
            "products": [],
            "destinations": [],
            "leaving_uae": leaving,
            "account_closure_intent": closure,
        },
    }


def _risk(band="Low"):
    return {"risk_band": band, "final_risk_score": 0.3, "risk_reasons": []}


def _cust(segment="Mass", clv=10000.0):
    return {"customer_id": "FB1", "segment": segment, "clv_estimate_aed": clv}


def test_leaver_triggers_dignified_goodbye():
    d = decide(_nlp(issue="Account_Closure", churn="High", leaving=True), _risk("High"), _cust())
    validate_decision(d)
    assert d["dignified_goodbye"] is True
    assert d["action"] == "Dignified Goodbye Pathway"
    assert d["offer_allowed"] is False
    assert d["max_offer_value_aed"] == 0.0


def test_closure_plus_high_churn_is_leaver():
    d = decide(_nlp(issue="Account_Closure", churn="High", closure=True), _risk("Medium"), _cust())
    assert d["dignified_goodbye"] is True


def test_high_value_high_risk_rm_call():
    d = decide(_nlp(issue="General_Query", churn="Medium"), _risk("High"), _cust("Premium"))
    validate_decision(d)
    assert d["action"] == "Relationship Manager Call"
    assert d["dignified_goodbye"] is False


def test_fee_waiver_path():
    d = decide(_nlp(issue="Fees_Charges", churn="Medium"), _risk("Medium"), _cust(clv=40000))
    assert d["action"] == "Fee Waiver / Service Recovery"
    assert d["offer_allowed"] is True
    assert d["max_offer_value_aed"] == round(0.05 * 40000, 2)


def test_low_clv_downgrades_offer_to_service():
    d = decide(_nlp(issue="Fees_Charges", churn="High"), _risk("High"), _cust(clv=3000))
    assert d["action"] == "Service Fix"
    assert d["offer_allowed"] is False
    assert d["max_offer_value_aed"] == 0.0


def test_routine_query():
    d = decide(_nlp(issue="General_Query", churn="Low"), _risk("Low"), _cust())
    assert d["action"] == "Standard Information Reply"


def test_requires_human_review_always_true():
    for issue in ["Account_Closure", "Fees_Charges", "General_Query", "Card_Services"]:
        d = decide(_nlp(issue=issue), _risk("Medium"), _cust())
        assert d["requires_human_review"] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("decision engine mock tests OK")
