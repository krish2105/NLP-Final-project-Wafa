# Individual Contribution Statements

**Project Wafa — Falcon Bank UAE Retention Intelligence Platform**
**Team (8 members):** Krishna Mathur · Kartik Joshi · Neha Thapa · Harsh Verma ·
Gagandeep Singh · Tanishk Verma · Anish Borkar · Zedan Parol

_One half-page per member. Add your own commit hashes / notebook cells as evidence._

---

## 1. Krishna Mathur — Team lead · Config, contracts & integration
- **Module owned:** `src/config.py`, `src/contracts.py`, end-to-end `src/pipeline.py`
- **Responsibilities:** froze the interface contracts before the build, wired the
  Listen→Understand→Act orchestration, kept `app.py` a thin presentation layer, ran
  integration and the demo-message pinning.
- **Hardest technical problem:** getting one message to flow raw-text → dashboard early
  (milestone M3) so integration never slipped to the final week.
- **What I learned:** why agreeing dict-shaped contracts first is what makes parallel
  module work actually integrate.
- **Evidence:** `src/contracts.py`, `src/pipeline.py`, commit *(add hash)*.

## 2. Kartik Joshi — NLP pipeline & translation
- **Module owned:** `src/nlp_pipeline.py`, `src/translation.py`
- **Responsibilities:** language detection, translate-then-classify route (opus-mt / NLLB),
  runtime classifier selection (DistilBERT → TF-IDF → heuristic), confidence flagging.
- **Hardest technical problem:** discovering that `opus-mt-hi-en` garbles **romanised**
  Hindi, then adding `is_romanised()` to skip translation instead of mistranslating.
- **What I learned:** the trade-off between translate-then-classify and multilingual-native,
  and why a translation failure must be surfaced, not hidden.
- **Evidence:** `src/translation.py`, `outputs/metrics/translation_audit.json`, commit *(add hash)*.

## 3. Neha Thapa — Entity extraction & multilingual handling
- **Module owned:** `src/entity_extraction.py`, `src/multilingual_native.py`
- **Responsibilities:** spaCy + regex + native/romanised keyword extraction (leaver /
  closure / amounts / destinations), and the multilingual-native embedding classifier.
- **Hardest technical problem:** catching leaver intent across English, Arabic script and
  romanised Hindi — solved by extracting from both original and translated text and merging.
- **What I learned:** how multilingual embeddings handle code-switched text natively
  (Hindi 1.00) where translation fails.
- **Evidence:** `tests/test_entity_extraction_mock.py`, `outputs/metrics/multilingual_native.json`, commit *(add hash)*.

## 4. Harsh Verma — Churn model & fairness audit
- **Module owned:** `src/churn_model.py`, `src/train_churn_model.py`
- **Responsibilities:** feature contract (excluding `nationality_region`), LogReg vs RF
  selection, permutation importance, the nationality fairness audit, cross-validation.
- **Hardest technical problem:** proving the model isn't proxying nationality — designed the
  separate post-hoc audit (spread 0.089) and 5-fold CV with confidence intervals.
- **What I learned:** why a protected attribute is excluded from features yet retained for auditing.
- **Evidence:** `outputs/metrics/churn_model_metrics.json`, `fairness_audit.json`,
  `notebooks/03_train_customer_churn.ipynb`, commit *(add hash)*.

## 5. Gagandeep Singh — Fusion & decision engine
- **Module owned:** `src/fusion.py`, `src/decision_engine.py`
- **Responsibilities:** the exact fusion formula, the transparent if/elif retention policy,
  the offer-economics (5%-of-CLV) check, and the dignified-goodbye path.
- **Hardest technical problem:** ordering the rules so a confirmed leaver can never fall
  through to a retention offer.
- **What I learned:** why "transparent rules for money and trust" must stay human-readable,
  never a learned black box.
- **Evidence:** `tests/test_fusion_mock.py`, `tests/test_decision_engine_mock.py`, commit *(add hash)*.

## 6. Tanishk Verma — Outreach generation & guardrails
- **Module owned:** `src/outreach_generator.py`, `src/guardrails.py`
- **Responsibilities:** the Qwen→Flan→template fallback chain, the named system prompt,
  own-language (Arabic) drafting, and the hallucination/dark-pattern guardrails.
- **Hardest technical problem:** guaranteeing a coherent, on-policy draft with zero downloads
  (template tier) while still catching a fabricated "AED 300" offer.
- **What I learned:** guardrails must surface violations for a human, never silently fix them.
- **Evidence:** guardrail catch in Ethics §5, `tests/test_guardrails_mock.py`,
  `outputs/metrics/outreach_comparison.json`, commit *(add hash)*.

## 7. Anish Borkar — Dashboard, UI & portfolio
- **Module owned:** `app.py`, `src/ui.py`, `src/portfolio_summary.py`, `src/audit_logger.py`
- **Responsibilities:** the 4-tab Streamlit console, the light/dark theme engine, the
  segment-level portfolio view, and the append-only audit log.
- **Hardest technical problem:** building a real light/dark theme that themes the whole
  Streamlit chrome (tables, charts, cards) without breaking any tab.
- **What I learned:** separation of concerns — keeping `app.py` free of business logic — is
  what reads as professional engineering.
- **Evidence:** `app.py`, `src/ui.py`, `outputs/metrics/portfolio_summary.json`, commit *(add hash)*.

## 8. Zedan Parol — Evaluation & innovation
- **Module owned:** `src/zero_shot_compare.py`, `src/train_lstm.py`, `src/cross_validation.py`, `notebooks/`
- **Responsibilities:** the trained-vs-zero-shot bake-off, the from-scratch LSTM, the
  noise/robustness stress test, per-language fairness reporting and notebook execution.
- **Hardest technical problem:** making the bake-off honest — realising the sample overlapped
  training data and citing the held-out numbers as the fair reference.
- **What I learned:** why a small owned model beats a zero-shot LLM at small data scale, and
  the cost/latency/privacy/fairness trade-offs behind that choice.
- **Evidence:** `outputs/metrics/zero_shot_comparison.json`, `lstm_metrics.json`,
  `cross_validation.json`, `notebooks/02_train_text_models.ipynb`, commit *(add hash)*.
