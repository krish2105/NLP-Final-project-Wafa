"""Train the customer-churn propensity model + run the nationality fairness audit.

Key constraints honoured here:
  * `nationality_region` is EXCLUDED from the feature set (see config.FEATURE_COLUMNS).
    It is loaded separately and used ONLY for the post-hoc fairness audit.
  * Two models are trained (LogReg + RandomForest); the best by ROC-AUC is saved.
  * Feature importance uses permutation importance (SHAP optional, never blocking).

Run:  python -m src.train_churn_model   (or  python src/train_churn_model.py)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Support both `python -m src.train_churn_model` and `python src/train_churn_model.py`
try:
    from . import config
    from .utils import save_json
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config
    from src.utils import save_json

RANDOM_STATE = 42


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def _load_xy():
    df = pd.read_csv(config.CUSTOMERS_CSV)
    for c in ("salary_credit_active", "intl_transfer_spike", "churned"):
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    X = df[config.FEATURE_COLUMNS].copy()
    X["salary_credit_active"] = X["salary_credit_active"].astype(int)
    X["intl_transfer_spike"] = X["intl_transfer_spike"].astype(int)
    y = df[config.TARGET_COLUMN].astype(int)
    groups = df[config.PROTECTED_ATTRIBUTE]  # kept aside, NOT a feature
    return df, X, y, groups


def _plot_confusion(cm, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_roc(y_test, proba, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#0f4c81")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Churn Model ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_importance(names, vals, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], [vals[i] for i in order], color="#1a6fb0")
    ax.set_title("Permutation importance (churn model)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    df, X, y, groups = _load_xy()
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", _build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        fitted[name] = pipe
        print(f"  {name:22s} {results[name]}")

    # Select best by ROC-AUC
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]
    print(f"Selected best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    _plot_confusion(cm, config.FIGURES_DIR / "churn_confusion_matrix.png",
                    f"Churn CM ({best_name})")
    _plot_roc(y_test, proba, config.FIGURES_DIR / "churn_roc_curve.png")

    # Permutation importance on the test set
    perm = permutation_importance(
        best, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    imp = {config.FEATURE_COLUMNS[i]: round(float(perm.importances_mean[i]), 5)
           for i in range(len(config.FEATURE_COLUMNS))}
    imp_sorted = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    _plot_importance(config.FEATURE_COLUMNS, list(perm.importances_mean),
                     config.FIGURES_DIR / "churn_feature_importance.png")

    # ---------------- Fairness audit (uses nationality_region ONLY here) ------
    all_proba = best.predict_proba(X)[:, 1]
    audit_df = pd.DataFrame({"group": groups.values, "proba": all_proba})
    fairness = {}
    for grp, sub in audit_df.groupby("group"):
        fairness[str(grp)] = {
            "n": int(len(sub)),
            "mean_churn_proba": round(float(sub.proba.mean()), 4),
            "std_churn_proba": round(float(sub.proba.std(ddof=0)), 4),
        }
    means = [v["mean_churn_proba"] for v in fairness.values()]
    spread = round(max(means) - min(means), 4)
    fairness_summary = {
        "by_group": fairness,
        "max_minus_min_mean_proba": spread,
        "note": (
            "nationality_region was NEVER a training feature. This audit checks the "
            "model is not implicitly proxying it. A spread under ~0.10 is considered "
            "roughly even across regions."
        ),
        "verdict": "roughly even" if spread < 0.10 else "UNEVEN - investigate",
    }

    save_json(
        {
            "selected_model": best_name,
            "all_candidates": results,
            "best_metrics": results[best_name],
            "confusion_matrix": cm.tolist(),
            "permutation_importance": imp_sorted,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features_used": config.FEATURE_COLUMNS,
            "excluded_protected_attribute": config.PROTECTED_ATTRIBUTE,
        },
        config.METRICS_DIR / "churn_model_metrics.json",
    )
    save_json(fairness_summary, config.METRICS_DIR / "fairness_audit.json")

    import joblib

    joblib.dump(best, config.CHURN_MODEL_PATH)
    # persist preprocessor separately too (contract asks for preprocessors.joblib)
    joblib.dump(best.named_steps["pre"], config.PREPROCESSORS_PATH)

    print(f"Saved model -> {config.CHURN_MODEL_PATH}")
    print(f"Fairness spread (max-min mean proba): {spread}  -> {fairness_summary['verdict']}")
    return results, fairness_summary


if __name__ == "__main__":
    main()
