"""Mock tests for the outreach guardrails — no models required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guardrails import check_draft  # noqa: E402


def _dec(max_offer=0.0, dignified=False):
    return {"max_offer_value_aed": max_offer, "dignified_goodbye": dignified}


def test_clean_draft_passes():
    r = check_draft("Dear valued customer, thank you for banking with us. Our team will help with your account.",
                    _dec())
    assert r["guardrail_passed"] is True
    assert r["guardrail_warnings"] == []


def test_urgency_language_flagged():
    r = check_draft("Act now! This limited time offer expires soon, hurry.", _dec(max_offer=500))
    assert r["guardrail_passed"] is False
    assert any("Urgency" in w for w in r["guardrail_warnings"])


def test_coercion_flagged():
    r = check_draft("You must stay with us, we require you to remain a customer.", _dec())
    assert r["guardrail_passed"] is False
    assert any("Coercive" in w for w in r["guardrail_warnings"])


def test_amount_mismatch_flagged():
    r = check_draft("We can offer you AED 2000 to stay.", _dec(max_offer=500))
    assert r["guardrail_passed"] is False
    assert any("2000" in w for w in r["guardrail_warnings"])


def test_amount_match_ok():
    r = check_draft("A goodwill gesture of AED 500 has been noted.", _dec(max_offer=500))
    # 500 matches the approved offer -> no amount warning
    assert not any("does not match" in w for w in r["guardrail_warnings"])


def test_incentive_on_dignified_goodbye_flagged():
    r = check_draft("We are sorry to see you go — here is a special offer of AED 300 to stay.",
                    _dec(max_offer=0.0, dignified=True))
    assert r["guardrail_passed"] is False
    assert any("Dignified Goodbye" in w or "incentive" in w.lower() for w in r["guardrail_warnings"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("guardrails mock tests OK")
