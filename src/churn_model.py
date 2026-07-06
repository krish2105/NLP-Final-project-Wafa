"""Runtime wrapper around the trained customer-churn model.

Loads the saved sklearn Pipeline (preprocessing + classifier) and exposes a
simple `predict_proba` + `top_drivers` API used by fusion/portfolio. If the
model artifact does not exist yet, `predict_proba` falls back to a transparent
heuristic so the rest of the platform still runs (and says so via `is_trained`).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .utils import as_bool, safe_float, load_json


@lru_cache(maxsize=1)
def _load_artifact():
    try:
        import joblib

        if config.CHURN_MODEL_PATH.exists():
            return joblib.load(config.CHURN_MODEL_PATH)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _load_importances() -> List[str]:
    """Ordered feature names by permutation importance (if metrics saved)."""
    try:
        meta = load_json(config.METRICS_DIR / "churn_model_metrics.json")
        imp = meta.get("permutation_importance", {})
        return [k for k, _ in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)]
    except Exception:
        return []


def is_trained() -> bool:
    return _load_artifact() is not None


def _row_to_frame(customer_row: dict) -> pd.DataFrame:
    rec = {}
    for col in config.FEATURE_COLUMNS:
        v = customer_row.get(col)
        if col in ("salary_credit_active", "intl_transfer_spike"):
            rec[col] = int(as_bool(v))
        elif col == "segment":
            rec[col] = str(v)
        else:
            rec[col] = safe_float(v)
    return pd.DataFrame([rec], columns=config.FEATURE_COLUMNS)


def _heuristic_proba(customer_row: dict) -> float:
    """Transparent fallback when no model is trained yet."""
    score = 0.2
    if not as_bool(customer_row.get("salary_credit_active", True)):
        score += 0.25
    if as_bool(customer_row.get("intl_transfer_spike", False)):
        score += 0.15
    if safe_float(customer_row.get("balance_trend_3m", 0)) < -0.25:
        score += 0.2
    if int(customer_row.get("complaints_6m", 0)) >= 4:
        score += 0.1
    return min(1.0, score)


def predict_proba(customer_row: dict) -> float:
    """Probability of churn for one customer (0..1)."""
    model = _load_artifact()
    if model is None:
        return _heuristic_proba(customer_row)
    X = _row_to_frame(customer_row)
    try:
        return float(model.predict_proba(X)[0, 1])
    except Exception:
        return _heuristic_proba(customer_row)


def predict_proba_batch(df: pd.DataFrame) -> List[float]:
    model = _load_artifact()
    if model is None:
        return [_heuristic_proba(r) for _, r in df.iterrows()]
    X = df[config.FEATURE_COLUMNS].copy()
    for c in ("salary_credit_active", "intl_transfer_spike"):
        X[c] = X[c].map(as_bool).astype(int)
    try:
        return [float(p) for p in model.predict_proba(X)[:, 1]]
    except Exception:
        return [_heuristic_proba(r) for _, r in df.iterrows()]


def top_drivers(customer_row: dict, k: int = 3) -> List[str]:
    """Global feature-importance ordering intersected with what's 'active'
    (adverse) for this specific customer — a lightweight local explanation.
    """
    global_order = _load_importances() or [
        "balance_trend_3m",
        "salary_credit_active",
        "complaints_6m",
        "intl_transfer_spike",
        "tenure_months",
    ]
    adverse: List[str] = []
    if not as_bool(customer_row.get("salary_credit_active", True)):
        adverse.append("salary_credit_active")
    if as_bool(customer_row.get("intl_transfer_spike", False)):
        adverse.append("intl_transfer_spike")
    if safe_float(customer_row.get("balance_trend_3m", 0)) < 0:
        adverse.append("balance_trend_3m")
    if int(customer_row.get("complaints_6m", 0)) >= 2:
        adverse.append("complaints_6m")
    # prefer adverse features that are also globally important, then fill
    ordered = [f for f in global_order if f in adverse]
    ordered += [f for f in global_order if f not in ordered]
    return ordered[:k]
