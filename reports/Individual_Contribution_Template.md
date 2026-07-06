# Individual Contribution Statements

_One half-page per student. Fill in the placeholders; keep evidence concrete
(commit hashes, notebook cells, file/function names)._

---

## Student 1 — *(replace with real name)*
- **Module owned:** NLP pipeline & multilingual translation (`src/nlp_pipeline.py`, `src/translation.py`)
- **Responsibilities:** language detection, translate-then-classify route, runtime
  classifier selection (DistilBERT → TF-IDF → heuristic), confidence flagging.
- **Hardest technical problem:** keeping the pipeline runnable offline when
  translation models can't download — solved with native/romanised keyword
  fallbacks so non-English intent is still caught.
- **What I learned:** the trade-off between translate-then-classify and
  multilingual-native embeddings, and why confidence gaps matter even at equal accuracy.
- **Evidence:** `src/nlp_pipeline.py`, `notebooks/02_train_text_models.ipynb`, commit *(add hash)*.

## Student 2 — *(replace with real name)*
- **Module owned:** Churn model & fairness (`src/train_churn_model.py`, `src/churn_model.py`)
- **Responsibilities:** feature contract (excluding `nationality_region`),
  LogReg vs RF selection, permutation importance, nationality fairness audit.
- **Hardest technical problem:** proving the model isn't proxying nationality —
  designed the separate post-hoc audit (spread 0.089).
- **What I learned:** why a protected attribute must be excluded from features yet
  retained for auditing.
- **Evidence:** `outputs/metrics/churn_model_metrics.json`, `fairness_audit.json`,
  `notebooks/03_train_customer_churn.ipynb`, commit *(add hash)*.

## Student 3 — *(replace with real name)*
- **Module owned:** Fusion & decision engine (`src/fusion.py`, `src/decision_engine.py`)
- **Responsibilities:** the exact fusion formula, transparent if/elif retention
  policy, offer-economics (5%-of-CLV) check, dignified-goodbye path.
- **Hardest technical problem:** ordering the rules so a confirmed leaver can never
  fall through to a retention offer.
- **What I learned:** why "transparent rules for money and trust" must stay
  human-readable, not learned.
- **Evidence:** `tests/test_fusion_mock.py`, `tests/test_decision_engine_mock.py`, commit *(add hash)*.

## Student 4 — *(replace with real name)*
- **Module owned:** Outreach generation & guardrails (`src/outreach_generator.py`, `src/guardrails.py`)
- **Responsibilities:** Qwen→Flan→template fallback chain, the named system prompt,
  hallucination/dark-pattern guardrails.
- **Hardest technical problem:** guaranteeing a coherent, on-policy draft with zero
  model downloads (rule-based tier) while still catching fabricated offers.
- **What I learned:** guardrails must surface, never silently fix, violations.
- **Evidence:** guardrail catch documented in Ethics Statement §5, commit *(add hash)*.

## Student 5 — *(replace with real name)*
- **Module owned:** Dashboard, portfolio summary & audit (`app.py`, `src/portfolio_summary.py`, `src/audit_logger.py`)
- **Responsibilities:** 4-tab Streamlit UI (thin presentation layer), segment-level
  aggregation, append-only audit log, model-evaluation views.
- **Hardest technical problem:** keeping `app.py` free of business logic while still
  rendering the full Listen→Understand→Act flow.
- **What I learned:** separation of concerns is what reads as professional
  engineering, independent of the UI framework.
- **Evidence:** `app.py`, `outputs/metrics/portfolio_summary.json`,
  `notebooks/04_fairness_and_evaluation.ipynb`, commit *(add hash)*.

## Student 6 — *(replace with real name)*
- **Module owned:** Evaluation & innovation (`src/zero_shot_compare.py`, `notebooks/`)
- **Responsibilities:** trained-vs-zero-shot LLM bake-off, per-language fairness
  reporting, notebook execution, report evidence.
- **Hardest technical problem:** making the bake-off honest — realising the sample
  overlapped training data and citing the held-out numbers as the fair reference.
- **What I learned:** why a small owned model can beat a zero-shot LLM at small data
  scale, and the cost/latency/privacy/fairness trade-offs behind that choice.
- **Evidence:** `outputs/metrics/zero_shot_comparison.json`,
  `notebooks/02_train_text_models.ipynb`, commit *(add hash)*.

> _6–7 person team: pair up on the heavier modules (e.g. two students share the NLP
> pipeline + entity extraction, or the dashboard + portfolio). Adjust ownership to
> match how your team actually split the work._
