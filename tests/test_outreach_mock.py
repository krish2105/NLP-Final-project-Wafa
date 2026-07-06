"""Mock tests for the outreach generator's rule-based path + contract.

Forces WAFA_LIGHT_MODE so no model downloads — exercises the template tier,
which must always produce a coherent, guardrail-checked, human-review-gated draft.
"""
import os
import sys

os.environ["WAFA_LIGHT_MODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import validate_outreach  # noqa: E402
from src.outreach_generator import generate_outreach, OUTREACH_SYSTEM_PROMPT  # noqa: E402


def _nlp(issue="General_Query", lang="en"):
    return {"message_id": "M1", "customer_id": "FB1", "language": lang, "issue_type": issue,
            "churn_signal": "Medium", "entities": {}}


def _dec(action, dignified=False, offer=False, max_offer=0.0):
    return {"action": action, "action_reason": "test", "offer_allowed": offer,
            "max_offer_value_aed": max_offer, "requires_human_review": True,
            "dignified_goodbye": dignified}


def _cust(segment="Mass"):
    return {"customer_id": "FB1", "segment": segment, "clv_estimate_aed": 10000.0}


def test_outreach_contract_and_pending_review():
    out = generate_outreach(_nlp(), _dec("Standard Information Reply"), _cust(), use_llm=False)
    validate_outreach(out)
    assert out["human_status"] == "Pending Review"


def test_dignified_goodbye_has_no_incentive_and_passes_guardrails():
    out = generate_outreach(_nlp(issue="Account_Closure"),
                            _dec("Dignified Goodbye Pathway", dignified=True), _cust(), use_llm=False)
    assert out["guardrail_passed"] is True
    low = out["draft_text"].lower()
    for banned in ("waiver", "discount", "offer", "cashback"):
        assert banned not in low, f"dignified goodbye leaked incentive word: {banned}"


def test_non_english_flags_translation_note():
    out = generate_outreach(_nlp(lang="ar"), _dec("Remittance Support"), _cust(), use_llm=False)
    assert "translation" in out["draft_text"].lower()


def test_system_prompt_constants():
    # the prompt shown in the Ethics statement must contain the key constraints
    for phrase in ["urgency", "must", "Dignified Goodbye", "invent a product"]:
        assert phrase.lower() in OUTREACH_SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("outreach mock tests OK")
