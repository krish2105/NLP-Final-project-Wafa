# Project Wafa — 12–15 Minute Demo Script

> Backup plan: if the live app fails, play the **2-minute screen recording**
> (`reports/backup_demo.mp4` — record it during rehearsal) and narrate from these notes.

---

## 0. Setup (before you present)
```bash
cd project_wafa
streamlit run app.py           # leave running on the Live Console tab
```
Pinned demo messages are auto-resolved in `config.DEMO_MESSAGE_IDS`
(currently M0010 Arabic leaver, M0007 English leaver, M0243 Arabic routine query).

## 1. Business framing (2 min)
- Falcon Bank UAE, 18-person CX team, mass relocation → thousands of multilingual
  messages, ~45% non-English.
- The job: **Listen → Understand → Act**, with humans in the loop and no dark patterns.
- Show the architecture diagram (Architecture Design Document §2). Name who owned
  what (one line each).

## 2. Architecture walkthrough (1 min)
- "`app.py` is a thin presentation layer — all logic lives in `src/`."
- Contracts were frozen first; mock tests passed before any model existed
  (`tests/test_*_mock.py`). Point at the Listen/Understand/Act boxes.

## 3. Live demo (7 min) — all three pinned messages
**A. Arabic high-churn leaver (M0010)**
- Pick it → Process. Show Understanding: language=Arabic, issue=Account_Closure,
  churn=High, **Leaving UAE ✅** (caught from native Arabic keywords).
- Decision: **Dignified Goodbye** (purple badge), **no offer**.
- Outreach: warm farewell, guardrails passed. Click **Approve** → point at the toast.

**B. English leaver (M0007)**
- Process. Same Dignified Goodbye path from English text, `dignified_goodbye=true`,
  `max_offer_value_aed = 0`. Emphasise: a confirmed leaver can **never** fall through
  to a retention offer — that's the rule ordering in `decision_engine.py`.

**C. Routine query (M0243)**
- Process. Low text signal; show how fusion still weighs behaviour. Contrast with the
  leaver: routine → informational/again-human-reviewed, not pressure.

**(Optional) Guardrail catch** — Free-text tab: paste a draft with "special offer of
AED 300" for a goodbye and show it flagged (or reference Ethics Statement §5).

## 4. Evidence (3 min) — Model Evaluation tab
- Churn model: ROC-AUC 1.000, confusion matrix, permutation importance
  (`balance_trend_3m` dominates).
- Text: TF-IDF issue/churn accuracy 1.00; fine-tuned **DistilBERT** matches on churn
  (1.00) and 0.984 on issue, with higher confidence (0.91–0.99). **Honest line:**
  "the transformer needed 12 epochs to catch up and still shows no accuracy lift over
  a char-ngram baseline on this templated data — so we run the cheaper model by
  default and keep DistilBERT as the primary trained transformer." Also mention the
  Arabic demo: DistilBERT misclassifies it (0.39 conf, auto-flagged for review) but
  the dignified-goodbye decision comes from **entity extraction**, so the system
  stays correct — a nice robustness point.
- Per-language fairness table: **confidence** gap (English 0.824 vs Tagalog 0.646).
- Nationality fairness: spread 0.089 → roughly even; feature was excluded from training.

## 5. Business close (2 min)
- 50 High-risk customers; Premium carries the most valuable High-risk share.
- Retention economics: offers capped at 5% of CLV; retaining ~1/3 of High-risk
  preserves ~AED 510k of CLV vs ~AED 75k max offer exposure.
- Team capacity: turns "read everything" into "review a ranked, pre-drafted shortlist."
- What's next: real data + drift monitoring, calibrated probabilities, feedback loop
  from the audit log.

## Timing cheat-sheet
| Section | Target |
|---|---|
| Business framing | 2:00 |
| Architecture | 1:00 |
| Live demo (3 cases) | 7:00 |
| Evidence | 3:00 |
| Close | 2:00 |
| **Total** | **15:00** |
