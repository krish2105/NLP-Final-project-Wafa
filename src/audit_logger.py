"""Append-only audit log for every decision/human action.

Records follow the AuditLogRecord contract. Every dashboard action (approve,
edit, reject, override) writes a row here — this is the accountability trail.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .contracts import AUDIT_LOG_RECORD_EXAMPLE
from .utils import now_iso

FIELDNAMES: List[str] = list(AUDIT_LOG_RECORD_EXAMPLE.keys())


def _ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def build_record(
    nlp_output: dict,
    fused_risk: dict,
    decision: dict,
    tabular_probability: float,
    draft_text: str,
    human_decision: str = "Pending Review",
    override_reason: Optional[str] = None,
) -> Dict:
    return {
        "timestamp": now_iso(),
        "message_id": nlp_output.get("message_id"),
        "customer_id": nlp_output.get("customer_id"),
        "issue_type": nlp_output.get("issue_type"),
        "churn_signal": nlp_output.get("churn_signal"),
        "tabular_churn_probability": round(float(tabular_probability), 4),
        "final_risk_score": fused_risk.get("final_risk_score"),
        "risk_band": fused_risk.get("risk_band"),
        "recommended_action": decision.get("action"),
        "draft_text": (draft_text or "").replace("\n", " ").strip(),
        "human_decision": human_decision,
        "override_reason": override_reason,
    }


def log_record(record: Dict, path: Optional[Path] = None) -> None:
    path = Path(path or config.AUDIT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_header(path)
    # keep only contract fields, in order
    row = {k: record.get(k) for k in FIELDNAMES}
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def read_log(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path or config.AUDIT_LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=FIELDNAMES)
    return pd.read_csv(path)
