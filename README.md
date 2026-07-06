# 🏦 Project Wafa — Falcon Bank UAE Retention Intelligence Platform

A Customer Retention Intelligence Platform for the (fictional) Falcon Bank UAE.
It triages multilingual customer messages and, for each one:

**Listen** (structured NLP signals) → **Understand** (fuse text + behaviour into
churn risk, per-customer and per-segment) → **Act** (transparent retention action +
human-reviewed outreach draft). **Nothing is ever auto-sent.**

---

## Requirements: slim vs full

| File | Use | Contents |
|---|---|---|
| `requirements.txt` | **Streamlit Cloud / minimal local** | streamlit, pandas, scikit-learn, plotly … (no torch) |
| `requirements-full.txt` | **Full local features** | + torch, transformers, spaCy, Qwen, sentence-transformers |

The app **degrades gracefully**: with the slim set it serves the committed TF-IDF +
churn models, rule-based outreach, regex entities, and pre-computed metrics/figures —
every tab works. Installing the full set unlocks DistilBERT, translation, and the
Qwen outreach LLM automatically.

## Quick start (laptop)

```bash
cd project_wafa
python -m venv .venv && source .venv/bin/activate        # optional

# Minimal (fast, no heavy downloads) — runs on the committed models:
pip install -r requirements.txt

# …or FULL features (DistilBERT, translation, Qwen, spaCy):
pip install -r requirements-full.txt

# (optional) retrain / regenerate artifacts
python src/train_churn_model.py                          # churn model + fairness audit
python src/train_text_models.py --distilbert --epochs 12 # TF-IDF baseline + DistilBERT
python -m src.portfolio_summary

# Launch the dashboard
streamlit run app.py
```

## 🚀 Deploy to Streamlit Community Cloud (free)

The repo is deploy-ready — `app.py` is at the root, `requirements.txt` is the
cloud-safe slim set, and `.streamlit/config.toml` sets the theme.

1. Push to GitHub (already done).
2. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns the repo.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `krish2105/NLP-Final-project-Wafa`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a subdomain, e.g. `project-wafa` → `https://project-wafa.streamlit.app`
   - **Advanced settings → Python version:** `3.11`
5. Click **Deploy** and wait ~2–5 min for the first build.

On the cloud (free tier ~1 GB RAM) the app runs in light mode: **TF-IDF classifier +
rule-based outreach**. The Model Evaluation tab still shows the full **DistilBERT** and
**zero-shot bake-off** numbers (read from the committed metrics JSON), so nothing is lost
for the demo. To show the live Qwen LLM drafting, run locally with `requirements-full.txt`.

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
