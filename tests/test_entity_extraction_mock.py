"""Mock tests for entity extraction — regex/keyword layer, no spaCy required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_extraction import extract_entities, extract_entities_multi  # noqa: E402


def test_english_leaver():
    e = extract_entities("I am relocating and want to close my account. Moving back to India.")
    assert e["leaving_uae"] is True
    assert e["account_closure_intent"] is True
    assert "India" in e["destinations"]


def test_amounts_and_products():
    e = extract_entities("You charged me AED 250 on my salary account and my card.")
    assert any("250" in a for a in e["amounts"])
    assert "card" in e["products"]


def test_native_arabic_leaver():
    # Arabic: "I will leave the UAE next month" + close account
    e = extract_entities("سأغادر الإمارات الشهر القادم. أغلق حساب الراتب.")
    assert e["leaving_uae"] is True
    assert e["account_closure_intent"] is True


def test_romanised_hindi_leaver():
    # romanised Hindi: "I am leaving UAE next month, close my account"
    e = extract_entities("main agle mahine UAE chhod raha hoon. mera account band karo.")
    assert e["leaving_uae"] is True
    assert e["account_closure_intent"] is True


def test_routine_no_intent():
    e = extract_entities("What are your branch timings on Friday?")
    assert e["leaving_uae"] is False
    assert e["account_closure_intent"] is False


def test_multi_merges_original_and_translation():
    # original Arabic (native keyword) + a translation that drops the clause
    merged = extract_entities_multi("سأغادر الإمارات الشهر القادم.", "I have a question.")
    assert merged["leaving_uae"] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("entity extraction mock tests OK")
