"""Mock/contract tests for the NLP pipeline + entity extraction.

These do NOT require any downloaded transformer — they exercise the regex/keyword
entity layer and the pipeline's contract shape (falling back to the TF-IDF model
if trained, or the keyword heuristic otherwise).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import validate_nlp_output  # noqa: E402
from src.entity_extraction import extract_entities  # noqa: E402
from src.nlp_pipeline import detect_language, process_message  # noqa: E402


def test_entities_leaver():
    e = extract_entities("I am relocating and want to close my account. Moving back to India.")
    assert e["leaving_uae"] is True
    assert e["account_closure_intent"] is True
    assert "India" in e["destinations"]


def test_entities_amounts_and_products():
    e = extract_entities("You charged me AED 250 on my salary account and my card.")
    assert any("250" in a for a in e["amounts"])
    assert "card" in e["products"]
    assert e["leaving_uae"] is False


def test_entities_routine():
    e = extract_entities("What are your branch timings on Friday?")
    assert e["leaving_uae"] is False
    assert e["account_closure_intent"] is False


def test_language_detection_scripts():
    assert detect_language("مرحبا اريد اغلاق حسابي") == "ar"
    assert detect_language("मुझे अपना खाता बंद करना है") == "hi"
    assert detect_language("hello I need help") == "en"


def test_process_message_contract():
    out = process_message(
        "how do I close my salary account. I have resigned and am moving back home.",
        customer_id="FB1186",
        message_id="M0007",
    )
    validate_nlp_output(out)
    assert out["entities"]["leaving_uae"] is True
    assert out["customer_id"] == "FB1186"
    assert out["issue_type"] in (
        "Account_Closure",
        "Remittance_Transfer",
        "Loan_Mortgage",
        "Fees_Charges",
        "Card_Services",
        "App_Technical",
        "General_Query",
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("nlp pipeline mock tests OK")
