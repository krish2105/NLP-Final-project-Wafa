"""Mock/contract tests for the churn model wrapper.

Works whether or not the real model is trained: if untrained it uses the
transparent heuristic, if trained it uses the sklearn pipeline. Either way the
output must be a probability in [0,1] and top_drivers must be a list.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import churn_model  # noqa: E402


def _cust(**kw):
    base = {
        "customer_id": "FB0001",
        "tenure_months": 20,
        "segment": "Mass",
        "products_held": 2,
        "avg_balance_aed": 8000.0,
        "balance_trend_3m": 0.0,
        "salary_credit_active": True,
        "remittance_count_3m": 2,
        "intl_transfer_spike": False,
        "complaints_6m": 0,
        "branch_visits_trend": 0.0,
        "clv_estimate_aed": 8000.0,
    }
    base.update(kw)
    return base


def test_proba_in_range():
    p = churn_model.predict_proba(_cust())
    assert 0.0 <= p <= 1.0


def test_adverse_customer_higher_risk():
    calm = churn_model.predict_proba(_cust())
    risky = churn_model.predict_proba(
        _cust(salary_credit_active=False, intl_transfer_spike=True,
              balance_trend_3m=-0.6, complaints_6m=6)
    )
    assert risky >= calm


def test_top_drivers_list():
    drivers = churn_model.top_drivers(
        _cust(salary_credit_active=False, balance_trend_3m=-0.4)
    )
    assert isinstance(drivers, list) and len(drivers) >= 1
    assert "salary_credit_active" in drivers


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("churn model mock tests OK")
