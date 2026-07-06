# Project Wafa — Ethics Statement

Each point below is backed by a **concrete finding from this build**, not just a
policy statement.

## 1. Dignified goodbye (verified)
Confirmed leavers are routed to a distinct, testable `Dignified Goodbye Pathway`
with `dignified_goodbye = True`. Verified end-to-end on two demo messages:
- **M0007 (English):** "close my salary account… resigned and moving back home" →
  `dignified_goodbye: true`, `offer_allowed: false`, `max_offer_value_aed: 0.0`.
- **M0010 (Arabic):** "سأغادر الإمارات الشهر القادم" (leaving the UAE next month) →
  detected via **native-script keywords** → `dignified_goodbye: true`, no offer.
The generated draft thanks the customer and offers help with a smooth
transfer/closure — no retention incentive. Unit test:
`tests/test_decision_engine_mock.py::test_leaver_triggers_dignified_goodbye`.

## 2. No dark patterns — the exact outreach prompt is shown
The system prompt is a named constant (`OUTREACH_SYSTEM_PROMPT`) and is reproduced
verbatim:

```
You are a retention specialist at a UAE bank drafting a short outreach message.
Rules you must follow, without exception:
- Never use urgency language ("limited time", "act now", "expires soon").
- Never say the customer "must" stay or imply any obligation to remain a customer.
- Never state a fee, rate, or offer value beyond what is explicitly given to you.
- Never reference the customer's anxiety, distress, or emotional state to persuade them.
- Never invent a product name that was not given to you.
- If the action is "Dignified Goodbye Pathway": do NOT offer any retention incentive.
  Thank the customer, offer help with a smooth transfer/closure, and preserve goodwill only.
- Be honest, warm, professional, and concise (3-5 sentences).
- Match the customer's language when a translation model is available; otherwise write in English
  and note the target language for human translation.
```

## 3. Multilingual fairness (measured gap)
The **same underlying issues phrased in 4 languages** were run through the pipeline
(`src/train_text_models.py::per_language_fairness`). Held-out results:

| Language | n | Accuracy | Mean confidence |
|---|---|---|---|
| English | 30 | 1.00 | **0.824** |
| Arabic | 12 | 1.00 | 0.762 |
| Hindi | 6 | 1.00 | 0.726 |
| Tagalog | 15 | 1.00 | **0.646** |

**Finding:** accuracy is saturated by the synthetic data, but **model confidence is
markedly lower for non-English, worst for Tagalog (0.646 vs 0.824, a 0.178 gap).**
This is a documented limitation: on noisier real text this confidence gap would
likely become an **accuracy** gap. Mitigation in place: low-confidence predictions
set `low_confidence_flag` and are surfaced for human review before the decision
engine trusts them.

**Zero-shot reinforces the point.** In the trained-vs-zero-shot bake-off, an unguarded
zero-shot Qwen-0.5B classified issue type at only 0.55 overall and **0.00 on Hindi**
(vs 0.64 English) — i.e. a prompted LLM would deliver visibly worse service in some
languages. Using a **trained, evaluated** classifier is therefore also a fairness
decision, not only an accuracy one.

## 4. Human in the loop (approval gate)
No message is ever auto-sent. Every `DecisionOutput` has
`requires_human_review = True` and every `OutreachOutput` starts at
`human_status = "Pending Review"`. Only a dashboard action (Approve / Edit &
Approve / Reject / Override) changes state, and **every action writes an
`AuditLogRecord`** to `outputs/audit_log.csv`. There is no auto-send code path
anywhere in the repository.

## 5. Hallucination guardrails (a real catch)
`src/guardrails.py` inspects every draft and never silently rewrites it. A **real
hallucination caught in testing:** a draft for a departing customer read *"here is a
special offer of AED 300 to stay with us."* On a `Dignified Goodbye` this is both a
dark pattern and a fabricated offer (approved max = AED 0). The guardrails flagged
three violations:
1. `Draft mentions AED 300 which does not match approved max offer value AED 0.`
2. `Retention-incentive language 'special offer' present on a Dignified Goodbye.`
3. `Dignified Goodbye draft mentions a monetary amount — remove it.`
The draft is shown to the reviewer **with the warnings attached**, not sent.

## 6. Nationality fairness audit (numbers)
`nationality_region` is **never** a training feature (`config.FEATURE_COLUMNS`
excludes it). The trained model's predicted churn probabilities by region:

| Region | n | Mean churn prob | Std |
|---|---|---|---|
| East_Asia | 44 | 0.357 | 0.426 |
| Southeast_Asia | 44 | 0.358 | 0.437 |
| South_Asia | 49 | 0.352 | 0.407 |
| MENA | 43 | 0.280 | 0.402 |
| Western | 60 | 0.269 | 0.384 |

**Spread (max − min mean) = 0.089 → "roughly even"** (threshold 0.10). The model is
not visibly proxying nationality. Saved to `outputs/metrics/fairness_audit.json`.

## 7. Data limitations
Synthetic data (`generate_wafa_data.py`), small (252 / 240 rows), with churn drivers
deliberately and strongly encoded — the near-perfect scores reflect the **generator**,
not production readiness. The platform is decision-support: it ranks and drafts;
accountable humans decide and act. See Business Report §8.
