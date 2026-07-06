"""Mock tests for the fusion layer — no models required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import validate_fused_risk  # noqa: E402
from src.fusion import fuse_risk, text_signal_to_score  # noqa: E402


def _cust(**kw):
    base = {
        "customer_id": "FB0001",
        "salary_credit_active": True,
        "intl_transfer_spike": False,
        "balance_trend_3m": 0.0,
        "complaints_6m": 0,
        "clv_estimate_aed": 10000.0,
    }
    base.update(kw)
    return base


def test_text_signal_scores():
    assert text_signal_to_score("Low") == 0.2
    assert text_signal_to_score("Medium") == 0.55
    assert text_signal_to_score("High") == 0.9


def test_low_risk_path():
    out = fuse_risk(0.1, "Low", _cust())
    validate_fused_risk(out)
    assert out["risk_band"] == "Low"
    assert out["behavior_score"] == 0.1


def test_high_risk_with_boosts():
    cust = _cust(
        salary_credit_active=False,
        intl_transfer_spike=True,
        balance_trend_3m=-0.5,
        complaints_6m=5,
        clv_estimate_aed=50000.0,
    )
    out = fuse_risk(0.9, "High", cust)
    validate_fused_risk(out)
    assert out["risk_band"] == "High"
    assert "salary credit stopped" in out["risk_reasons"]
    assert "international transfer spike" in out["risk_reasons"]
    assert "balance declining sharply" in out["risk_reasons"]
    assert "repeated complaints" in out["risk_reasons"]
    assert "high CLV customer" in out["risk_reasons"]
    assert out["final_risk_score"] <= 1.0  # clamped


def test_formula_exact():
    # 0.45*0.5 + 0.40*0.55 + 0 = 0.225 + 0.22 = 0.445 -> just below 0.45 => Low
    out = fuse_risk(0.5, "Medium", _cust())
    assert abs(out["final_risk_score"] - 0.445) < 1e-6
    assert out["risk_band"] == "Low"
    # a case that clears the 0.45 Medium threshold
    out2 = fuse_risk(0.6, "Medium", _cust())  # 0.27 + 0.22 = 0.49
    assert out2["risk_band"] == "Medium"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("fusion mock tests OK")
