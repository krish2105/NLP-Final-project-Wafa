"""Segment-level risk summary (Capability 2 minimum bar).

Runs the churn model + fusion across ALL customers (using each customer's most
representative message where one exists) and aggregates into:
  * risk-band counts by segment
  * top risk drivers by segment
  * a per-customer table for the dashboard
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd

from . import churn_model, config
from .data_loader import load_customers, representative_message_per_customer
from .fusion import fuse_risk
from .nlp_pipeline import classify_churn
from .entity_extraction import extract_entities
from .utils import save_json


def _churn_signal_for_customer(rep_msgs: Dict[str, Dict], cid: str) -> str:
    """Prefer the labelled churn_signal from the representative message; if none,
    fall back to a light classifier on any text, else 'Low'."""
    msg = rep_msgs.get(cid)
    if not msg:
        return "Low"
    # messages.csv already carries a churn_signal label — use it as the text signal
    sig = msg.get("churn_signal")
    if sig in ("Low", "Medium", "High"):
        return sig
    ent = extract_entities(str(msg.get("text", "")))
    label, _ = classify_churn(str(msg.get("text", "")), ent)
    return label


def build_portfolio_summary(save: bool = True) -> Dict:
    customers = load_customers()
    rep_msgs = representative_message_per_customer()

    probs = churn_model.predict_proba_batch(customers)

    rows: List[Dict] = []
    for (_, cust), prob in zip(customers.iterrows(), probs):
        cid = cust["customer_id"]
        signal = _churn_signal_for_customer(rep_msgs, cid)
        cust_dict = cust.to_dict()
        fused = fuse_risk(prob, signal, cust_dict)
        drivers = churn_model.top_drivers(cust_dict)
        rows.append(
            {
                "customer_id": cid,
                "segment": cust["segment"],
                "nationality_region": cust["nationality_region"],
                "risk_band": fused["risk_band"],
                "final_risk_score": fused["final_risk_score"],
                "tabular_churn_probability": round(float(prob), 4),
                "text_churn_signal": signal,
                "top_driver": drivers[0] if drivers else "n/a",
                "top_drivers": drivers,
                "clv_estimate_aed": float(cust["clv_estimate_aed"]),
                "has_message": cid in rep_msgs,
            }
        )

    table = pd.DataFrame(rows)

    # risk band counts by segment
    band_by_segment = (
        table.groupby(["segment", "risk_band"]).size().unstack(fill_value=0)
    )
    for b in ("Low", "Medium", "High"):
        if b not in band_by_segment.columns:
            band_by_segment[b] = 0
    band_by_segment = band_by_segment[["Low", "Medium", "High"]]

    # top drivers among High-band customers, by segment
    drivers_by_segment: Dict[str, List] = {}
    high = table[table.risk_band == "High"]
    for seg, g in high.groupby("segment"):
        cnt = Counter()
        for drivers in g["top_drivers"]:
            cnt.update(drivers)
        drivers_by_segment[seg] = cnt.most_common(5)

    overall_high_drivers = Counter()
    for drivers in high["top_drivers"]:
        overall_high_drivers.update(drivers)

    summary = {
        "n_customers": int(len(table)),
        "risk_band_counts_overall": dict(table.risk_band.value_counts()),
        "risk_band_by_segment": band_by_segment.astype(int).to_dict(orient="index"),
        "top_drivers_by_segment": {k: v for k, v in drivers_by_segment.items()},
        "top_drivers_overall_high": overall_high_drivers.most_common(6),
        "high_risk_customer_count": int((table.risk_band == "High").sum()),
    }

    if save:
        save_json(summary, config.METRICS_DIR / "portfolio_summary.json")
        table_out = table.drop(columns=["top_drivers"]).copy()
        table_out.to_csv(config.OUTPUTS_DIR / "portfolio_table.csv", index=False)

    return {"summary": summary, "table": table}


if __name__ == "__main__":
    res = build_portfolio_summary()
    s = res["summary"]
    print("Customers:", s["n_customers"])
    print("Risk bands overall:", s["risk_band_counts_overall"])
    print("By segment:", s["risk_band_by_segment"])
    print("Top High-risk drivers:", s["top_drivers_overall_high"])
