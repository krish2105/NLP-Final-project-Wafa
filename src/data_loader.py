"""Load and lightly normalise the two source datasets.

Kept deliberately thin: it returns clean pandas DataFrames / dict rows and does
NO business logic. Everything downstream depends on the column contract here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .utils import as_bool


@lru_cache(maxsize=1)
def load_messages() -> pd.DataFrame:
    df = pd.read_csv(config.MESSAGES_CSV)
    df["message_id"] = df["message_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def load_customers() -> pd.DataFrame:
    df = pd.read_csv(config.CUSTOMERS_CSV)
    df["customer_id"] = df["customer_id"].astype(str)
    # normalise the two boolean columns coming from CSV
    for col in ("salary_credit_active", "intl_transfer_spike"):
        df[col] = df[col].map(as_bool)
    if "churned" in df.columns:
        df["churned"] = df["churned"].map(as_bool)
    return df


def get_customer_row(customer_id: str) -> Optional[Dict]:
    """Return a single customer as a plain dict (contract-friendly), or None."""
    df = load_customers()
    hit = df[df.customer_id == str(customer_id)]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    # guarantee native python types for the decision/fusion contracts
    row["salary_credit_active"] = bool(row["salary_credit_active"])
    row["intl_transfer_spike"] = bool(row["intl_transfer_spike"])
    row["clv_estimate_aed"] = float(row["clv_estimate_aed"])
    row["balance_trend_3m"] = float(row["balance_trend_3m"])
    row["complaints_6m"] = int(row["complaints_6m"])
    return row


def get_message_row(message_id: str) -> Optional[Dict]:
    df = load_messages()
    hit = df[df.message_id == str(message_id)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def representative_message_per_customer() -> Dict[str, Dict]:
    """Most 'representative' message per customer = the one with the strongest
    churn signal (High > Medium > Low). Used by the portfolio summary so each
    customer contributes one text signal.
    """
    order = {"High": 3, "Medium": 2, "Low": 1}
    df = load_messages().copy()
    df["_rank"] = df["churn_signal"].map(order).fillna(0)
    df = df.sort_values("_rank", ascending=False)
    out: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        cid = r["customer_id"]
        if cid not in out:
            out[cid] = r.drop(labels="_rank").to_dict()
    return out


def list_demo_messages() -> List[Dict]:
    df = load_messages()
    rows = []
    for mid in config.DEMO_MESSAGE_IDS:
        hit = df[df.message_id == mid]
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    return rows
