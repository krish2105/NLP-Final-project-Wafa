# 🏦 Project Wafa — Falcon Bank UAE Retention Intelligence Platform

A Customer Retention Intelligence Platform for the (fictional) Falcon Bank UAE.
It triages multilingual customer messages and, for each one:

**Listen** (structured NLP signals) → **Understand** (fuse text + behaviour into
churn risk, per-customer and per-segment) → **Act** (transparent retention action +
human-reviewed outreach draft). **Nothing is ever auto-sent.**

---

## Quick start (laptop, CPU-only, zero downloads)

```bash
cd project_wafa
python -m venv .venv && source .venv/bin/activate        # optional
pip install -r requirements.txt

# 1) Train the required models (fast, sklearn)
python src/train_churn_model.py       # churn model + nationality fairness audit
python src/train_text_models.py       # TF-IDF baseline (+ DistilBERT if transformers installed)

# 2) (optional) build the segment view / seed metrics
python -m src.portfolio_summary

# 3) Launch the dashboard
streamlit run app.py
```

The **default path needs no model downloads**: it uses the trained TF-IDF text
classifier, the sklearn churn model, native-keyword multilingual entity extraction,
and rule-based outreach templates.

### Enabling the heavy (optional) models
Install the NLP extras (`transformers`, `spacy`, etc. — already in
`requirements.txt`) and:
```bash
# spaCy English model (better MONEY/GPE/DATE entities).
# If `python -m spacy download en_core_web_sm` fails on newer `click`, install the wheel directly:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

python src/train_text_models.py --distilbert --epochs 12  # fine-tune DistilBERT (issue + churn)
# NOTE: use ~12 epochs — at the default 3–4 epochs DistilBERT is undertrained on
# this small (~190-row) set (issue acc ~0.60). At 12 epochs: issue 0.984, churn 1.00.

python -m src.zero_shot_compare                           # innovation: trained vs zero-shot LLM bake-off
```
Translation (opus-mt / NLLB) and the outreach LLM (**Qwen2.5-0.5B first**, then
1.5B / Flan-T5) load **automatically and lazily**; if any fail to download, the
platform silently degrades to the working fallbacks (trained TF-IDF + rule-based drafts).

### Force the lightweight path (CI / offline)
```bash
export WAFA_LIGHT_MODE=1     # skip all heavy model loads, use trained + rule-based
```

## Run the tests
```bash
for t in tests/test_*_mock.py; do python "$t"; done
```
All contract/mock tests pass with **no** trained models present (they were written
before the models existed — build-order step 1).

## What's where
```
project_wafa/
├── app.py                      # 4-tab Streamlit dashboard (thin presentation layer)
├── src/                        # all logic (config, contracts, pipeline, models, ACT layer)
├── models/                     # saved artifacts (churn model, TF-IDF, DistilBERT dirs)
├── outputs/{figures,metrics}/  # confusion matrices, ROC, metrics JSON, fairness audits
├── outputs/audit_log.csv       # append-only human-decision log
├── notebooks/                  # 01 EDA · 02 text models · 03 churn · 04 fairness/eval
├── tests/                      # contract/mock tests
├── reports/                    # Architecture, Business, Ethics, Contributions, Script
└── data/                       # messages.csv, customers.csv, generate_wafa_data.py
```

## Key guarantees (graded requirements)
- **Free only** — no paid API keys anywhere in the default code path.
- **Trained model(s)** — sklearn churn model + TF-IDF text classifiers are genuinely
  trained by this code; DistilBERT fine-tune is a real, optional primary route.
- **Transparent decisions** — `src/decision_engine.py` is explicit `if/elif`, never a
  black box.
- **Nationality excluded + audited** — never a training feature; `fairness_audit.json`.
- **Dignified goodbye** — distinct, tested code path (`dignified_goodbye = True`).
- **No dark patterns** — outreach system prompt shown in the Ethics Statement;
  guardrails flag violations for humans.
- **Human in the loop** — every draft starts `Pending Review`; no auto-send exists.
- **Audit log** — every action recorded to `outputs/audit_log.csv`.

## Dashboard tabs
1. **Live Console** — per-message Listen→Understand→Act with Approve/Edit/Reject/Override.
2. **Portfolio Risk Overview** — risk bands by segment, top drivers, sortable table.
3. **Model Evaluation** — confusion matrices, ROC, importance, per-language & nationality fairness.
4. **Audit Log** — filterable live view of every human decision.
