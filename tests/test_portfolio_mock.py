"""Smoke/contract tests for the segment-level portfolio summary."""
import os
import sys

os.environ["WAFA_LIGHT_MODE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio_summary import build_portfolio_summary  # noqa: E402


def test_portfolio_shapes():
    res = build_portfolio_summary(save=False)
    s, table = res["summary"], res["table"]
    assert s["n_customers"] == len(table)
    assert s["n_customers"] > 0
    # every customer has a valid risk band
    assert set(table["risk_band"]).issubset({"Low", "Medium", "High"})
    # band counts by segment sum to total customers
    total = sum(sum(v.values()) for v in s["risk_band_by_segment"].values())
    assert total == s["n_customers"]


def test_high_risk_count_consistent():
    res = build_portfolio_summary(save=False)
    s, table = res["summary"], res["table"]
    assert s["high_risk_customer_count"] == int((table["risk_band"] == "High").sum())


def test_top_drivers_present():
    res = build_portfolio_summary(save=False)
    assert isinstance(res["summary"]["top_drivers_overall_high"], list)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("portfolio mock tests OK")
