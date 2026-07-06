"""Train the message text classifiers.

Two independent approaches, reported honestly side by side:
  (a) PRIMARY (required trained model): fine-tuned DistilBERT — runs when
      `transformers` + `torch` are available and WAFA_LIGHT_MODE != 1.
  (b) BASELINE: TF-IDF (word + char n-grams) + Logistic Regression — always runs.

Both classify BOTH targets: issue_type (7-way) and churn_signal (3-way), on the
SAME stratified split so the comparison is fair.

Also runs the multilingual fairness test (per-language accuracy + mean
confidence) required for the Ethics Statement.

Run:  python -m src.train_text_models
      python -m src.train_text_models --distilbert   (force-attempt the transformer)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

try:
    from . import config
    from .utils import save_json
    from .translation import translate_to_english, _transformers_available
except ImportError:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config
    from src.utils import save_json
    from src.translation import translate_to_english, _transformers_available

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def _prepare_dataframe(translate: bool) -> pd.DataFrame:
    df = pd.read_csv(config.MESSAGES_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    if translate and _transformers_available():
        print("Translating non-English messages to English (this may download models)...")
        eng = []
        for _, r in df.iterrows():
            t, _flag = translate_to_english(r["text"], r["language"])
            eng.append(t)
        df["train_text"] = eng
        df["_translated"] = True
    else:
        # Honest fallback: train on raw text. char n-grams give cross-script signal.
        df["train_text"] = df["text"]
        df["_translated"] = False
    return df


def _build_tfidf_pipeline() -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    return Pipeline(
        [
            ("feats", FeatureUnion([("word", word), ("char", char)])),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]
    )


def _plot_cm(cm, labels, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3.5, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# TF-IDF baseline
# --------------------------------------------------------------------------- #
def train_tfidf(df, target, labels, split_idx, joblib_path, fig_name):
    import joblib

    train_idx, test_idx = split_idx
    Xtr = df.loc[train_idx, "train_text"]
    Xte = df.loc[test_idx, "train_text"]
    ytr = df.loc[train_idx, target]
    yte = df.loc[test_idx, target]

    pipe = _build_tfidf_pipeline()
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)
    conf = proba.max(axis=1)

    acc = accuracy_score(yte, pred)
    macro_f1 = f1_score(yte, pred, average="macro")
    report = classification_report(yte, pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(yte, pred, labels=labels)
    _plot_cm(cm, labels, config.FIGURES_DIR / fig_name, f"TF-IDF: {target}")

    joblib.dump(pipe, joblib_path)
    return {
        "model": "tfidf_logreg",
        "target": target,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "mean_confidence": round(float(conf.mean()), 4),
        "test_confidence": conf.tolist(),
    }


# --------------------------------------------------------------------------- #
# DistilBERT (optional, real fine-tune)
# --------------------------------------------------------------------------- #
def train_distilbert(df, target, labels, split_idx, out_dir, fig_name, epochs=3):
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_idx, test_idx = split_idx
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    class DS(Dataset):
        def __init__(self, texts, ys):
            self.enc = tok(list(texts), truncation=True, padding=True, max_length=128)
            self.ys = [label2id[y] for y in ys]

        def __len__(self):
            return len(self.ys)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[i])
            return item

    train_ds = DS(df.loc[train_idx, "train_text"], df.loc[train_idx, target])
    test_ds = DS(df.loc[test_idx, "train_text"], df.loc[test_idx, target])

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "_hf_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        report_to=[],
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    # eval
    preds_logits = trainer.predict(test_ds).predictions
    probs = torch.softmax(torch.tensor(preds_logits), dim=1).numpy()
    pred_ids = probs.argmax(axis=1)
    pred = [id2label[i] for i in pred_ids]
    yte = list(df.loc[test_idx, target])
    conf = probs.max(axis=1)

    acc = accuracy_score(yte, pred)
    macro_f1 = f1_score(yte, pred, average="macro")
    report = classification_report(yte, pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(yte, pred, labels=labels)
    _plot_cm(cm, labels, config.FIGURES_DIR / fig_name, f"DistilBERT: {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    return {
        "model": "distilbert",
        "target": target,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "mean_confidence": round(float(conf.mean()), 4),
        "epochs": epochs,
    }


# --------------------------------------------------------------------------- #
# Per-language fairness test
# --------------------------------------------------------------------------- #
def per_language_fairness(df, split_idx, target="issue_type"):
    """Evaluate the TF-IDF issue classifier per language on the test split."""
    import joblib

    _, test_idx = split_idx
    pipe = joblib.load(config.TFIDF_ISSUE_PATH)
    sub = df.loc[test_idx].copy()
    pred = pipe.predict(sub["train_text"])
    proba = pipe.predict_proba(sub["train_text"]).max(axis=1)
    sub = sub.assign(_pred=pred, _conf=proba)
    table = {}
    for lang, g in sub.groupby("language"):
        table[lang] = {
            "language_name": config.LANGUAGE_NAMES.get(lang, lang),
            "n": int(len(g)),
            "accuracy": round(float((g["_pred"] == g[target]).mean()), 4),
            "mean_confidence": round(float(g["_conf"].mean()), 4),
        }
    en_acc = table.get("en", {}).get("accuracy", None)
    gaps = {
        l: round(en_acc - v["accuracy"], 4)
        for l, v in table.items()
        if en_acc is not None and l != "en"
    }
    return {
        "per_language": table,
        "english_minus_other_accuracy_gap": gaps,
        "note": (
            "Same underlying issues phrased across 4 languages, run through the full "
            "pipeline. Lower non-English accuracy is a documented limitation "
            "(translation quality + training on raw/short multilingual text)."
        ),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilbert", action="store_true", help="force-attempt DistilBERT fine-tune")
    ap.add_argument("--no-distilbert", action="store_true", help="skip DistilBERT even if available")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--translate", action="store_true", help="translate before training if possible")
    args = ap.parse_args()

    df = _prepare_dataframe(translate=args.translate)
    print(f"Loaded {len(df)} messages. Translated={df['_translated'].iloc[0]}")

    # single shared stratified split (stratify on issue_type)
    idx = df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=RANDOM_STATE, stratify=df["issue_type"]
    )
    split_idx = (train_idx, test_idx)

    metrics = {"translated": bool(df["_translated"].iloc[0]), "models": {}}

    # ---- TF-IDF baseline (always) ----
    print("\n[Baseline] TF-IDF + LogReg — issue_type")
    m_issue_tfidf = train_tfidf(
        df, "issue_type", config.ISSUE_TYPES, split_idx,
        config.TFIDF_ISSUE_PATH, "tfidf_issue_confusion.png",
    )
    print(f"  acc={m_issue_tfidf['accuracy']} macroF1={m_issue_tfidf['macro_f1']}")

    print("[Baseline] TF-IDF + LogReg — churn_signal")
    m_churn_tfidf = train_tfidf(
        df, "churn_signal", config.CHURN_SIGNALS, split_idx,
        config.TFIDF_CHURN_PATH, "tfidf_churn_confusion.png",
    )
    print(f"  acc={m_churn_tfidf['accuracy']} macroF1={m_churn_tfidf['macro_f1']}")

    metrics["models"]["tfidf_issue"] = m_issue_tfidf
    metrics["models"]["tfidf_churn"] = m_churn_tfidf

    # ---- DistilBERT (optional) ----
    want_bert = (args.distilbert or _transformers_available()) and not args.no_distilbert
    if want_bert and _transformers_available():
        try:
            print("\n[Primary] Fine-tuning DistilBERT — issue_type")
            m_issue_bert = train_distilbert(
                df, "issue_type", config.ISSUE_TYPES, split_idx,
                config.DISTILBERT_ISSUE_DIR, "distilbert_issue_confusion.png", args.epochs,
            )
            print(f"  acc={m_issue_bert['accuracy']} macroF1={m_issue_bert['macro_f1']}")
            print("[Primary] Fine-tuning DistilBERT — churn_signal")
            m_churn_bert = train_distilbert(
                df, "churn_signal", config.CHURN_SIGNALS, split_idx,
                config.DISTILBERT_CHURN_DIR, "distilbert_churn_confusion.png", args.epochs,
            )
            print(f"  acc={m_churn_bert['accuracy']} macroF1={m_churn_bert['macro_f1']}")
            metrics["models"]["distilbert_issue"] = m_issue_bert
            metrics["models"]["distilbert_churn"] = m_churn_bert
        except Exception as e:
            print(f"DistilBERT fine-tune failed ({e}); baseline metrics still saved.")
            metrics["distilbert_error"] = str(e)
    else:
        note = "transformers/torch unavailable or disabled — DistilBERT skipped; TF-IDF baseline is the active trained model."
        print("\n" + note)
        metrics["distilbert_skipped"] = note

    # ---- per-language fairness ----
    print("\n[Fairness] Per-language accuracy test")
    fair = per_language_fairness(df, split_idx)
    for lang, row in fair["per_language"].items():
        print(f"  {row['language_name']:10s} n={row['n']:2d} acc={row['accuracy']} conf={row['mean_confidence']}")
    metrics["per_language_fairness"] = fair
    save_json(fair, config.METRICS_DIR / "per_language_fairness.json")

    save_json(metrics, config.METRICS_DIR / "text_model_metrics.json")
    print(f"\nSaved metrics -> {config.METRICS_DIR / 'text_model_metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
