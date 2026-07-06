"""Project Wafa — Falcon Bank UAE Retention Console (Streamlit).

Thin presentation layer ONLY. All business logic lives in src/. app.py never
loads a model directly, never wrangles data, never makes a decision — it calls
functions from src/ and renders the results. Theming/HTML helpers live in src/ui.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import audit_logger, config, ui  # noqa: E402
from src.data_loader import get_message_row, list_demo_messages, load_messages  # noqa: E402
from src.outreach_generator import get_active_generator_name  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.portfolio_summary import build_portfolio_summary  # noqa: E402
from src.utils import load_json  # noqa: E402

# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Project Wafa — Falcon Bank Retention Console",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme state — read BEFORE injecting CSS so the toggle takes effect on rerun.
st.session_state.setdefault("wafa_dark", False)
DARK = st.session_state["wafa_dark"]
st.markdown(ui.theme_css(DARK), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources (models load once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _warm_pipeline():
    from src import churn_model
    from src.nlp_pipeline import _load_tfidf

    _load_tfidf("issue")
    _load_tfidf("churn")
    return {"churn_trained": churn_model.is_trained()}


@st.cache_data(show_spinner="Building portfolio view…")
def _portfolio():
    res = build_portfolio_summary(save=False)
    return res["summary"], res["table"]


def _metric_json(name: str):
    p = config.METRICS_DIR / name
    return load_json(p) if p.exists() else None


def _risk_gauge(score: float, band: str):
    import plotly.graph_objects as go

    color = {"High": ui.ACCENT["high"], "Medium": ui.ACCENT["medium"], "Low": ui.ACCENT["low"]}[band]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(47,163,107,.22)"},
                    {"range": [45, 75], "color": "rgba(224,160,44,.22)"},
                    {"range": [75, 100], "color": "rgba(229,72,77,.22)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "value": round(score * 100, 1)},
            },
        )
    )
    fig.update_layout(**ui.plotly_layout(DARK, height=200))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — brand, theme toggle, live status, legend
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.15rem;font-weight:800;">🏦 Project Wafa</div>'
        '<div style="color:var(--muted);font-size:.82rem;margin-bottom:.4rem;">'
        "Falcon Bank UAE · Retention Intelligence</div>",
        unsafe_allow_html=True,
    )
    st.toggle("🌙 Dark mode", key="wafa_dark", help="Switch between light and dark themes.")
    st.divider()

    warm = _warm_pipeline()
    gen_name = get_active_generator_name()
    gen_label = "Rule-based" if gen_name in ("not yet loaded", "rule-based-template") else gen_name.split("/")[-1]
    st.markdown(ui.section("System status"), unsafe_allow_html=True)
    st.markdown(
        f"""<div style="font-size:.86rem;line-height:1.9">
        <div>🧠 Churn model &nbsp;<b>{'trained' if warm['churn_trained'] else 'heuristic'}</b></div>
        <div>🔤 Text classifier &nbsp;<b>DistilBERT / TF-IDF</b></div>
        <div>✉️ Outreach &nbsp;<b>{gen_label}</b></div>
        <div>🛡️ Guardrails &nbsp;<b>active</b></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(ui.section("Risk legend"), unsafe_allow_html=True)
    st.markdown(
        f'{ui.badge("High")} &nbsp; {ui.badge("Medium")} &nbsp; {ui.badge("Low")}<br>'
        f'<span style="font-size:.8rem;color:var(--muted)">Leaver path → {ui.badge("", leaver=True)}</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Nothing auto-sends. Every draft is human-reviewed and logged.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(ui.header_html(gen_label, "Dark" if DARK else "Light"), unsafe_allow_html=True)

tab_console, tab_portfolio, tab_eval, tab_audit = st.tabs(
    ["🎯 Live Console", "📊 Portfolio Risk", "🧪 Model Evaluation", "📜 Audit Log"]
)

# =========================================================================== #
# TAB 1 — LIVE CONSOLE
# =========================================================================== #
with tab_console:
    msgs = load_messages()
    demo_rows = list_demo_messages()
    demo_ids = [r["message_id"] for r in demo_rows]

    with st.container(border=True):
        st.markdown(ui.section("Triage a customer message"), unsafe_allow_html=True)
        mode = st.radio(
            "Input source", ["Pick a message", "Free text"], horizontal=True,
            label_visibility="collapsed",
        )
        selected_text, selected_cid, selected_mid, selected_lang = "", "", None, None

        if mode == "Pick a message":
            ordered = demo_ids + [m for m in msgs["message_id"].tolist() if m not in demo_ids]

            def _fmt(mid):
                r = get_message_row(mid)
                star = "⭐ " if mid in demo_ids else ""
                return f"{star}{mid} · {r['language']} · {r['issue_type']} · {r['churn_signal']}"

            c1, c2 = st.columns([3, 1])
            selected_mid = c1.selectbox("Message (⭐ = pinned demo)", ordered, format_func=_fmt)
            row = get_message_row(selected_mid)
            selected_text, selected_cid, selected_lang = row["text"], row["customer_id"], row["language"]
            c2.markdown(
                f'<div style="margin-top:1.8rem"><span class="badge badge-soft">👤 {selected_cid}</span></div>',
                unsafe_allow_html=True,
            )
            st.text_area("Message text", selected_text, height=84, disabled=True)
        else:
            selected_text = st.text_area(
                "Message text", "I want to close my account, I am leaving the UAE next month.", height=84
            )
            selected_cid = st.text_input("Customer ID", value="FB1000")

        b1, b2, b3 = st.columns([1.2, 1.5, 1.5])
        use_llm = b2.toggle(
            "Use open LLM (Qwen2.5-0.5B) for the draft",
            value=False, help="Off = fast rule-based template.",
        )
        match_lang = b3.toggle(
            "Draft in customer's language",
            value=False, help="Needs the LLM on. Qwen drafts in Arabic/Tagalog/etc.",
        )
        go = b1.button("▶ Process message", type="primary", use_container_width=True)

    if go:
        with st.spinner("Running Listen → Understand → Act…"):
            st.session_state["result"] = run_pipeline(
                selected_text, selected_cid, message_id=selected_mid,
                language=selected_lang, use_llm=use_llm, match_language=match_lang,
            )

    result = st.session_state.get("result")
    if not result:
        st.info("Pick a message (or type one) and press **Process** to see the full pipeline.")
    else:
        nlp, risk, dec, out = (
            result["nlp_output"], result["fused_risk"], result["decision"], result["outreach"]
        )
        e = nlp["entities"]

        col_u, col_r = st.columns(2)
        # ---- Understanding ----
        with col_u:
            ent_bits = []
            ent_bits.append(f'<div class="row"><span class="k">Leaving UAE</span>'
                            f'<span class="v">{"✅ yes" if e["leaving_uae"] else "—"}</span></div>')
            ent_bits.append(f'<div class="row"><span class="k">Close account</span>'
                            f'<span class="v">{"✅ yes" if e["account_closure_intent"] else "—"}</span></div>')
            if e["products"]:
                ent_bits.append(f'<div class="row"><span class="k">Products</span><span class="v">{", ".join(e["products"])}</span></div>')
            if e["destinations"]:
                ent_bits.append(f'<div class="row"><span class="k">Destinations</span><span class="v">{", ".join(e["destinations"])}</span></div>')
            if e["amounts"]:
                ent_bits.append(f'<div class="row"><span class="k">Amounts</span><span class="v">{", ".join(map(str, e["amounts"]))}</span></div>')
            lc = ('<div style="margin-top:.5rem" class="badge badge-soft">⚠ low confidence — flag for review</div>'
                  if nlp.get("low_confidence_flag") else "")
            trans = (f'<div style="margin-top:.5rem;color:var(--muted);font-size:.82rem">↳ {nlp["translated_text"]}</div>'
                     if nlp.get("translated_text") else "")
            st.markdown(
                f"""<div class="wafa-card"><h4>🔎 Understanding</h4>
                <div class="row"><span class="k">Language</span><span class="v">{config.LANGUAGE_NAMES.get(nlp['language'], nlp['language'])}</span></div>
                <div class="row"><span class="k">Issue type</span><span class="v">{nlp['issue_type']} · {nlp['issue_confidence']:.0%}</span></div>
                <div class="row"><span class="k">Churn signal</span><span class="v">{ui.badge(nlp['churn_signal'])} {nlp['churn_confidence']:.0%}</span></div>
                {"".join(ent_bits)}{lc}{trans}</div>""",
                unsafe_allow_html=True,
            )
        # ---- Risk ----
        with col_r:
            reasons = "".join(f"<li>{r}</li>" for r in risk["risk_reasons"]) or "<li>no elevated drivers</li>"
            st.markdown(
                f"""<div class="wafa-card"><h4>📈 Fused risk &nbsp; {ui.badge(risk['risk_band'])}</h4>
                <div class="row"><span class="k">Behaviour (model)</span><span class="v">{risk['behavior_score']:.0%}</span></div>
                <div class="row"><span class="k">Text signal</span><span class="v">{risk['text_score']:.0%}</span></div>
                <div style="margin-top:.4rem;color:var(--muted);font-size:.82rem">Why:</div>
                <ul>{reasons}</ul></div>""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(_risk_gauge(risk["final_risk_score"], risk["risk_band"]),
                            use_container_width=True, config={"displayModeBar": False})

        col_d, col_o = st.columns(2)
        # ---- Decision ----
        with col_d:
            leaver = ui.badge("", leaver=True) if dec["dignified_goodbye"] else ""
            offer = (f'AED {dec["max_offer_value_aed"]:,.2f} <span style="color:var(--muted);font-size:.8rem">(≤5% CLV)</span>'
                     if dec["offer_allowed"] else "none")
            st.markdown(
                f"""<div class="wafa-card"><h4>🧭 Decision {leaver}</h4>
                <div class="row"><span class="k">Action</span><span class="v">{dec['action']}</span></div>
                <div class="row"><span class="k">Max offer</span><span class="v">{offer}</span></div>
                <div style="margin-top:.5rem;font-size:.88rem">{dec['action_reason']}</div>
                <div style="margin-top:.5rem;color:var(--muted);font-size:.78rem">Transparent if/elif policy — never a black box.</div>
                </div>""",
                unsafe_allow_html=True,
            )
        # ---- Outreach (interactive) ----
        with col_o:
            with st.container(border=True):
                st.markdown(ui.section("✉️ Outreach · Pending Review"), unsafe_allow_html=True)
                st.text_area("Editable draft", out["draft_text"], height=150, key="draft_edit",
                             label_visibility="collapsed")
                st.caption(f"Generator: {out.get('generator_used', get_active_generator_name())} · "
                           f"target language: {config.LANGUAGE_NAMES.get(nlp['language'], nlp['language'])}")
                if out["guardrail_passed"]:
                    st.success("✅ Guardrails passed.")
                else:
                    st.error("🛡️ Guardrail warnings — review before sending:")
                    for w in out["guardrail_warnings"]:
                        st.write(f"- {w}")

        # ---- Human actions -> audit log ----
        st.markdown(ui.section("Human action → writes to audit log"), unsafe_allow_html=True)
        a1, a2, a3, a4, a5 = st.columns([1, 1.2, 1, 1, 2])
        override_reason = a5.text_input("Override / note (optional)", "", label_visibility="collapsed",
                                        placeholder="Override / note (optional)")

        def _write(decision_label, reason=None):
            rec = audit_logger.build_record(
                nlp, risk, dec, result["tabular_churn_probability"],
                st.session_state.get("draft_edit", out["draft_text"]),
                human_decision=decision_label, override_reason=reason or None,
            )
            audit_logger.log_record(rec)
            st.toast(f"Logged: {decision_label}", icon="✅")

        if a1.button("✅ Approve", use_container_width=True):
            _write("Approved", override_reason)
        if a2.button("✏️ Edit & Approve", use_container_width=True):
            _write("Edited & Approved", override_reason or "edited draft text")
        if a3.button("❌ Reject", use_container_width=True):
            _write("Rejected", override_reason)
        if a4.button("↩ Override", use_container_width=True):
            _write("Overridden", override_reason or "action overridden by agent")

# =========================================================================== #
# TAB 2 — PORTFOLIO
# =========================================================================== #
with tab_portfolio:
    st.markdown(ui.section("Segment-level risk — who is at risk, in which segments, driven by what"),
                unsafe_allow_html=True)
    summary, table = _portfolio()
    counts = summary["risk_band_counts_overall"]

    st.markdown(
        ui.stat_grid([
            ui.stat_card("Customers", summary["n_customers"], "in portfolio", "primary"),
            ui.stat_card("High risk", summary["high_risk_customer_count"],
                         f'{summary["high_risk_customer_count"]/summary["n_customers"]:.0%} of book', "high"),
            ui.stat_card("Medium risk", int(counts.get("Medium", 0)), "watchlist", "med"),
            ui.stat_card("Low risk", int(counts.get("Low", 0)), "stable", "primary"),
        ]),
        unsafe_allow_html=True,
    )

    seg_df = pd.DataFrame(summary["risk_band_by_segment"]).T[["Low", "Medium", "High"]]
    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown(ui.section("Risk band by segment"), unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            colors = {"Low": ui.ACCENT["low"], "Medium": ui.ACCENT["medium"], "High": ui.ACCENT["high"]}
            for band in ["Low", "Medium", "High"]:
                fig.add_bar(x=seg_df.index, y=seg_df[band], name=band, marker_color=colors[band])
            fig.update_layout(barmode="stack", legend_title="Risk band", **ui.plotly_layout(DARK, 360))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.bar_chart(seg_df)
    with g2:
        st.markdown(ui.section("Top drivers · High-band"), unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go

            dd = pd.DataFrame(summary["top_drivers_overall_high"], columns=["driver", "count"]).iloc[::-1]
            fig2 = go.Figure(go.Bar(x=dd["count"], y=dd["driver"], orientation="h",
                                    marker_color=ui.ACCENT["high"]))
            fig2.update_layout(**ui.plotly_layout(DARK, 360))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.dataframe(pd.DataFrame(summary["top_drivers_overall_high"], columns=["driver", "count"]))

    st.markdown(ui.section("All customers"), unsafe_allow_html=True)
    show = table.drop(columns=["top_drivers"]).sort_values("final_risk_score", ascending=False)
    fc1, fc2 = st.columns([1, 3])
    band_filter = fc1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    show = show[show.risk_band.isin(band_filter)]
    show = show.rename(columns={
        "customer_id": "Customer", "segment": "Segment", "nationality_region": "Region",
        "risk_band": "Risk", "final_risk_score": "Score", "tabular_churn_probability": "P(churn)",
        "text_churn_signal": "Text", "top_driver": "Top driver", "clv_estimate_aed": "CLV (AED)",
    })
    cols = ["Customer", "Segment", "Region", "Risk", "Score", "P(churn)", "Text", "Top driver", "CLV (AED)"]
    show = show[cols].copy()
    show["CLV (AED)"] = show["CLV (AED)"].map(lambda v: f"{v:,.0f}")
    st.markdown(ui.render_table(show, height=420, band_col="Risk"), unsafe_allow_html=True)
    st.caption(f"{len(show)} customers shown · sorted by fused risk score.")

# =========================================================================== #
# TAB 3 — MODEL EVALUATION
# =========================================================================== #
with tab_eval:
    st.markdown(ui.section("Model evaluation — trained artifacts, fairness, and the bake-off"),
                unsafe_allow_html=True)

    churn_metrics = _metric_json("churn_model_metrics.json")
    text_metrics = _metric_json("text_model_metrics.json")
    fairness = _metric_json("fairness_audit.json")
    per_lang = _metric_json("per_language_fairness.json")
    zeroshot = _metric_json("zero_shot_comparison.json")

    # -- churn --
    st.markdown(ui.section("Customer churn model"), unsafe_allow_html=True)
    if churn_metrics:
        bm = churn_metrics["best_metrics"]
        st.markdown(
            ui.stat_grid([
                ui.stat_card("Selected", churn_metrics["selected_model"].replace("_", " ").title(), "best by ROC-AUC", "primary"),
                ui.stat_card("ROC-AUC", bm["roc_auc"], "held-out", "primary"),
                ui.stat_card("F1", bm["f1"], "churn class", "med"),
                ui.stat_card("Excludes", churn_metrics["excluded_protected_attribute"], "not a feature", "high"),
            ]),
            unsafe_allow_html=True,
        )
        figs = [("churn_confusion_matrix.png", "Confusion matrix"),
                ("churn_roc_curve.png", "ROC curve"),
                ("churn_feature_importance.png", "Permutation importance")]
        cc = st.columns(3)
        for col, (f, cap) in zip(cc, figs):
            html = ui.figure_card(config.FIGURES_DIR / f, cap)
            if html:
                col.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Run `python src/train_churn_model.py` to populate churn metrics.")

    # -- text classifiers --
    st.markdown(ui.section("Message text classifiers"), unsafe_allow_html=True)
    if text_metrics:
        rows = []
        for key, m in text_metrics["models"].items():
            rows.append({"Model · target": key, "Accuracy": m["accuracy"], "Macro-F1": m["macro_f1"],
                         "Mean conf.": m.get("mean_confidence"), "Epochs": m.get("epochs", "—")})
        st.markdown(ui.render_table(pd.DataFrame(rows), height=240), unsafe_allow_html=True)
        figs = [(f, f.replace("_", " ").replace(".png", "").title())
                for f in ["tfidf_issue_confusion.png", "distilbert_issue_confusion.png",
                          "tfidf_churn_confusion.png", "distilbert_churn_confusion.png"]
                if (config.FIGURES_DIR / f).exists()]
        ccols = st.columns(2)
        for i, (f, cap) in enumerate(figs):
            html = ui.figure_card(config.FIGURES_DIR / f, cap)
            if html:
                ccols[i % 2].markdown(html, unsafe_allow_html=True)
    else:
        st.info("Run `python src/train_text_models.py` to populate text metrics.")

    # -- bake-off --
    st.markdown(ui.section("💡 Innovation — trained model vs zero-shot LLM"), unsafe_allow_html=True)
    if zeroshot:
        zr = pd.DataFrame(zeroshot["results"]).T.reset_index().rename(
            columns={"index": "Model", "type": "Type", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}
        )[["Model", "Type", "Accuracy", "Macro-F1"]]
        st.markdown(ui.render_table(zr, height=180), unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin:.4rem 0"><span class="badge badge-soft">🏆 Winner: {zeroshot["winner"]}</span> '
            f'<span class="badge badge-soft">Zero-shot by language: {zeroshot["zero_shot_accuracy_by_language"]}</span></div>',
            unsafe_allow_html=True,
        )
        st.info(zeroshot["finding"])
        st.caption("Honesty caveat: " + zeroshot.get("honesty_caveat", ""))
    else:
        st.info("Run `python -m src.zero_shot_compare` to populate the bake-off.")

    # -- four-way trained scoreboard (issue_type) --
    st.markdown(ui.section("🏁 Trained-model scoreboard (issue_type)"), unsafe_allow_html=True)
    lstm = _metric_json("lstm_metrics.json")
    mln = _metric_json("multilingual_native.json")
    board = []
    if text_metrics and "tfidf_issue" in text_metrics["models"]:
        board.append({"Approach": "TF-IDF + LogReg", "Type": "trained (classical)",
                      "Accuracy": text_metrics["models"]["tfidf_issue"]["accuracy"]})
    if text_metrics and "distilbert_issue" in text_metrics["models"]:
        board.append({"Approach": "DistilBERT (fine-tuned)", "Type": "trained (transformer)",
                      "Accuracy": text_metrics["models"]["distilbert_issue"]["accuracy"]})
    if mln:
        board.append({"Approach": "Multilingual embeddings + LogReg", "Type": "trained (no translation)",
                      "Accuracy": mln["accuracy"]})
    if lstm:
        board.append({"Approach": "Bi-LSTM from scratch", "Type": "trained (recurrent)",
                      "Accuracy": lstm["accuracy"]})
    if zeroshot and "zero_shot_qwen0.5b" in zeroshot["results"]:
        board.append({"Approach": "Qwen0.5B zero-shot", "Type": "not trained (prompt only)",
                      "Accuracy": zeroshot["results"]["zero_shot_qwen0.5b"]["accuracy"]})
    if board:
        st.markdown(ui.render_table(pd.DataFrame(board).sort_values("Accuracy", ascending=False), height=220),
                    unsafe_allow_html=True)
        st.caption("Honest finding: trained TF-IDF/DistilBERT lead; the from-scratch LSTM and zero-shot LLM "
                   "trail at this small data scale — the expected, defensible trade-off.")

    # -- credible evaluation: CV + robustness + translation audit --
    cv = _metric_json("cross_validation.json")
    ta = _metric_json("translation_audit.json")
    ev1, ev2 = st.columns(2)
    with ev1:
        st.markdown(ui.section("Cross-validation & noise robustness"), unsafe_allow_html=True)
        if cv:
            rows = []
            for n, v in cv["churn_model_cv"].items():
                rows.append({"Model": n, "ROC-AUC (5-fold)": f"{v['roc_auc']['mean']}±{v['roc_auc']['std']}"})
            for t, v in cv["robustness_noise_test"].items():
                rows.append({"Model": f"text {t} (clean→noisy)",
                             "ROC-AUC (5-fold)": f"{v['clean_accuracy']}→{v['noisy_accuracy']}"})
            st.markdown(ui.render_table(pd.DataFrame(rows), height=200), unsafe_allow_html=True)
    with ev2:
        st.markdown(ui.section("Translation quality audit"), unsafe_allow_html=True)
        if ta:
            rows = [{"Language": v["language_name"], "Route": v["route"], "Faithful rate": v["faithful_rate"]}
                    for v in ta["per_language"].values()]
            st.markdown(ui.render_table(pd.DataFrame(rows), height=200), unsafe_allow_html=True)
            st.caption("Romanised Hindi (0.0) is left untranslated on purpose — honest fairness finding.")

    # -- fairness --
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown(ui.section("Multilingual fairness (per-language)"), unsafe_allow_html=True)
        if per_lang:
            pl = pd.DataFrame(per_lang["per_language"]).T.reset_index().rename(columns={"index": "lang"})
            st.markdown(ui.render_table(pl, height=220), unsafe_allow_html=True)
    with fcol2:
        st.markdown(ui.section("Nationality fairness audit"), unsafe_allow_html=True)
        if fairness:
            fa = pd.DataFrame(fairness["by_group"]).T.reset_index().rename(columns={"index": "region"})
            st.markdown(ui.render_table(fa, height=220), unsafe_allow_html=True)
            verdict = fairness["verdict"]
            (st.success if "even" in verdict else st.error)(
                f"Max−min mean churn probability = {fairness['max_minus_min_mean_proba']} → {verdict}"
            )

# =========================================================================== #
# TAB 4 — AUDIT LOG
# =========================================================================== #
with tab_audit:
    st.markdown(ui.section("Audit log — every human action is recorded"), unsafe_allow_html=True)
    df = audit_logger.read_log()
    if df.empty:
        st.info("No audit records yet. Process a message in the Live Console and take an action.")
    else:
        dec_counts = df["human_decision"].value_counts().to_dict()
        st.markdown(
            ui.stat_grid([
                ui.stat_card("Total records", len(df), "logged actions", "primary"),
                ui.stat_card("Approved", dec_counts.get("Approved", 0) + dec_counts.get("Edited & Approved", 0), "incl. edited", "primary"),
                ui.stat_card("Rejected", dec_counts.get("Rejected", 0), "", "high"),
                ui.stat_card("Overridden", dec_counts.get("Overridden", 0), "", "med"),
            ]),
            unsafe_allow_html=True,
        )
        f1, f2, f3 = st.columns(3)
        band_opt = f1.multiselect("Risk band", sorted(df["risk_band"].dropna().unique().tolist()))
        act_opt = f2.multiselect("Action", sorted(df["recommended_action"].dropna().unique().tolist()))
        dec_opt = f3.multiselect("Human decision", sorted(df["human_decision"].dropna().unique().tolist()))
        view = df.copy()
        if band_opt:
            view = view[view.risk_band.isin(band_opt)]
        if act_opt:
            view = view[view.recommended_action.isin(act_opt)]
        if dec_opt:
            view = view[view.human_decision.isin(dec_opt)]
        display = view.iloc[::-1].copy()
        if "draft_text" in display:
            display["draft_text"] = display["draft_text"].astype(str).str.slice(0, 70) + "…"
        st.markdown(ui.render_table(display, height=420, band_col="risk_band"), unsafe_allow_html=True)
        st.download_button("⬇ Download audit_log.csv", df.to_csv(index=False), "audit_log.csv", "text/csv")
