# Contributing to Project Wafa

Welcome! This is a team capstone. This guide keeps our parallel work painless —
the golden rule of the brief is *agree the interfaces first, integrate early*.

## 1. One-time setup

```bash
git clone https://github.com/krish2105/NLP-Final-project-Wafa.git
cd NLP-Final-project-Wafa

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Full local features (DistilBERT, translation, Qwen, spaCy):
pip install -r requirements-full.txt
# …or the slim/cloud set (TF-IDF + rule-based, no torch):
# pip install -r requirements.txt

# Optional heavy models (DistilBERT / translation / Qwen). If the spaCy CLI fails on
# newer click, install the model wheel directly:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The repo ships the small trained models (`*.joblib`) so the app runs immediately.
The **large DistilBERT weights are git-ignored** (255MB each, over GitHub's limit) —
regenerate them locally when you need them:

```bash
python src/train_churn_model.py
python src/train_text_models.py --distilbert --epochs 12
python -m src.zero_shot_compare
streamlit run app.py
```

## 2. Branch & PR workflow (please don't push to `main`)

```bash
git checkout main && git pull                 # start from latest
git checkout -b <yourname>/<short-topic>       # e.g. rania/fusion-weights
# ...make changes...
python -m pytest tests/ -q                      # or: for t in tests/test_*_mock.py; do python "$t"; done
git add -p && git commit -m "clear message"
git push -u origin <yourname>/<short-topic>
```

Then open a **Pull Request** into `main`. Keep PRs small and focused. At least one
teammate reviews before merge. `main` should always run.

## 3. Module ownership (edit within your module to avoid conflicts)

| Area | Files | Owner |
|---|---|---|
| Config & contracts | `src/config.py`, `src/contracts.py` | Student 1 |
| NLP + entities | `src/nlp_pipeline.py`, `src/translation.py`, `src/entity_extraction.py` | Student 2 |
| Churn model + fairness | `src/churn_model.py`, `src/train_churn_model.py` | Student 3 |
| Fusion + decisions | `src/fusion.py`, `src/decision_engine.py` | Student 3 / 4 |
| Outreach + guardrails | `src/outreach_generator.py`, `src/guardrails.py` | Student 4 |
| Dashboard + UI | `app.py`, `src/ui.py`, `src/portfolio_summary.py` | Student 5 |
| Evaluation + notebooks | `src/zero_shot_compare.py`, `notebooks/` | Student 6 |

> Replace `Student N` with real names. If two people must touch the same file,
> talk first and split by function.

## 4. The interface contracts are frozen

Every module hands the next a fixed dict shape (see `src/contracts.py`). **Do not
change a contract without telling the team** — a change there ripples through every
module. Validate against the contracts with the mock tests before you push.

## 5. Non-negotiable rules (from the assignment brief)

- **Free only** — no paid API key may be required for the demo to run.
- `nationality_region` is **never** a training feature (fairness-audited only).
- The decision engine stays **explicit `if/elif`** — never a trained/black-box model.
- **No auto-send.** Every draft starts `Pending Review`; only a human changes it.
- Confirmed leavers get the **dignified goodbye** — never retention pressure.
- Keep `app.py` a **thin presentation layer** — logic lives in `src/`.

## 6. Before you commit

- [ ] Mock tests pass (`tests/test_*_mock.py`).
- [ ] You didn't commit large weights, `.venv`, or `.DS_Store` (see `.gitignore`).
- [ ] You didn't change a frozen contract without team sign-off.
- [ ] `streamlit run app.py` still boots and your tab has no errors.

Thanks for keeping Wafa clean and runnable. 🏦
