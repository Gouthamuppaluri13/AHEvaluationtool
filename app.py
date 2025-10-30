import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
import os
import logging
from typing import Dict, Any

from next_gen_vc_engine import NextGenVCEngine, FounderPersonality
from ui_theme import apply_theme, hero, section_heading, card
from services.pdf_ingest import PDFIngestor

# =========================
# Global theme + setup
# =========================
apply_theme(page_title="Anthill AI+ Evaluation", page_icon="🦅")

# Wizard state
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0  # 0 = Company Profile, 1 = Metrics & Financials
if "profile_inputs" not in st.session_state:
    st.session_state.profile_inputs = {}
if "view" not in st.session_state:
    st.session_state.view = "input"

# Pass-through secrets (optional)
for k in ["HUGGINGFACE_MODEL_ID", "MODEL_ARTIFACT_NAME", "MODEL_ASSET_URL", "FX_INR_PER_USD", "RESEARCH_API_URL", "RESEARCH_MODEL", "RESEARCH_RESPONSES_URL"]:
    try:
        val = st.secrets.get(k)
        if val:
            os.environ[k] = str(val)
    except Exception:
        pass

# Hero
hero("Anthill AI+ Evaluation", "Minimal, elegant, glass‑themed VC copilot with deep research.")

@st.cache_resource
def load_engine():
    try:
        engine = NextGenVCEngine(
            st.secrets.get("TAVILY_API_KEY"),
            st.secrets.get("ALPHA_VANTAGE_KEY"),
            None,  # no Gemini
            st.secrets.get("RESEARCH_API_KEY") or st.secrets.get("GROK_API_KEY"),
        )
        return engine
    except Exception as e:
        st.error("🔴 Could not initialize engine. Check keys.")
        logging.exception(e)
        st.stop()

# =========================
# Charts
# =========================
def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'family': 'ui-monospace, SFMono-Regular', 'size': 16, 'color': '#0F172A'}},
        number={'font': {'color': '#0F172A'}},
        gauge={'axis': {'range': [None, 10], 'tickcolor': '#94A3B8'},
               'bar': {'color': "#0A84FF"},
               'bgcolor': "rgba(255,255,255,0)",
               'borderwidth': 0,
               'steps': [
                   {'range': [0, 3.3], 'color': 'rgba(255,59,48,0.10)'},
                   {'range': [3.3, 6.6], 'color': 'rgba(255,204,0,0.10)'},
                   {'range': [6.6, 10], 'color': 'rgba(52,199,89,0.10)'}
               ]}
    ))
    fig.update_layout(template="simple_white", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#0F172A"))
    return fig

def create_spider_chart(data, title):
    if not isinstance(data, dict) or not data:
        data = {"Market": 5, "Execution": 5, "Technology": 5, "Regulatory": 5, "Competition": 5}
    fig = go.Figure(go.Scatterpolar(
        r=list(data.values()), theta=list(data.keys()), fill='toself',
        line=dict(color='#0A84FF', width=2), fillcolor='rgba(10,132,255,0.15)'
    ))
    fig.update_layout(template="simple_white",
        title=dict(text=title, font=dict(family='-apple-system, SF Pro Display, Inter', size=20, color='#0F172A')),
        polar=dict(bgcolor='rgba(255,255,255,0)',
                   radialaxis=dict(visible=True, range=[0, 10], gridcolor='#E2E8F0', linecolor='#CBD5E1', tickfont=dict(color='#475569')),
                   angularaxis=dict(gridcolor='#E2E8F0', linecolor='#CBD5E1', tickfont=dict(color='#475569'))),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0F172A'), margin=dict(l=10, r=10, t=40, b=10))
    return fig

# =========================
# Research display
# =========================
def _render_paragraphs(title: str, content: str):
    if not content:
        return
    with card():
        st.markdown(f"### {title}")
        parts = [p.strip() for p in content.replace("\r\n", "\n").split("\n\n") if p.strip()]
        for p in parts:
            st.markdown(p)

def display_research(md: Dict[str, Any]):
    ext = md.get("external_research", {}) or {}
    diags = (md.get("_diagnostics", {}) or {}).get("research", {}) if isinstance(md, dict) else {}
    st.markdown("#### 🔎 External Research (Web Deep Dive)")
    # Diagnostics (provider-agnostic)
    with st.expander("Diagnostics"):
        st.caption(f"Enabled: {diags.get('enabled', True)} | Cache hit: {diags.get('cache_hit', False)} | Last error: {diags.get('error','')}")
        if isinstance(ext, dict) and "_diagnostics" in ext:
            st.caption(f"Route: {ext['_diagnostics'].get('route','n/a')}")

    if "notice" in ext:
        st.info(ext["notice"])
        return
    if "error" in ext:
        st.info("External research was unavailable. Please try again.")
        return

    if ext.get("summary"):
        _render_paragraphs("Research Summary", ext["summary"])

    sections = ext.get("sections", {}) or {}
    if sections:
        st.markdown("---")
        colL, colR = st.columns(2)
        left = ["overview", "products", "business_model", "gtm", "unit_economics", "funding", "investors", "leadership"]
        right = ["hiring", "traction", "customers", "pricing", "competitors", "moat", "partnerships", "regulatory", "risks", "tech_stack", "roadmap"]
        with colL:
            for k in left:
                if sections.get(k):
                    _render_paragraphs(k.replace("_", " ").title(), sections[k])
        with colR:
            for k in right:
                if sections.get(k):
                    _render_paragraphs(k.replace("_", " ").title(), sections[k])

    sources = ext.get("sources", []) or []
    if sources:
        st.markdown("---")
        with card():
            st.markdown("### Citations")
            for i, s in enumerate(sources[:20], start=1):
                title = s.get("title", "Source")
                url = s.get("url", "#")
                snippet = s.get("snippet", "")
                conf = s.get("confidence", None)
                conf_str = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
                st.markdown(f"- [{i}] [{title}]({url}) — {snippet}{conf_str}")

# =========================
# PDF Quick Analysis — first page
# =========================
def render_pdf_quick_analysis():
    section_heading("📄 Quick Analysis From PDF", "Upload a deck and generate the full evaluation without manual entry")
    with card():
        uploaded = st.file_uploader("Upload a startup PDF deck", type=["pdf"], key="pdf_uploader_firstpage")
        comps_ticker = st.text_input("Public Comps Ticker (optional)", "ZOMATO.BSE", help="Used for the public comps lookup")

        col1, col2, col3 = st.columns([1, 1, 1])
        parse_clicked = col1.button("🔎 Parse PDF", use_container_width=True, key="parse_pdf_btn_firstpage")
        run_clicked = col2.button("🚀 Run Analysis from PDF", use_container_width=True, key="run_pdf_btn_firstpage")
        clear_clicked = col3.button("🧹 Clear", use_container_width=True, key="clear_pdf_btn_firstpage")

        if parse_clicked:
            if uploaded is None:
                st.warning("Please upload a PDF first.")
            else:
                with st.spinner("Extracting content and building inputs from PDF..."):
                    try:
                        ingestor = PDFIngestor(st.secrets.get("GEMINI_API_KEY"))  # regex fallback works without key
                        extracted = ingestor.extract(uploaded.getvalue(), file_name=getattr(uploaded, "name", None))
                        st.session_state["pdf_extracted_inputs"] = extracted
                        st.session_state["pdf_extracted_ticker"] = comps_ticker
                        st.success("Parsed PDF successfully. Review extracted fields below.")
                    except Exception:
                        st.error("We couldn't parse the PDF. Please try a different file.")

        if clear_clicked:
            st.session_state.pop("pdf_extracted_inputs", None)
            st.session_state.pop("pdf_extracted_ticker", None)
            st.info("Cleared parsed PDF data.")

        parsed = st.session_state.get("pdf_extracted_inputs")
        if parsed:
            st.caption("Status: Parsed ✅ — you can run analysis now.")
            prev_cols = st.columns(2)
            left_keys = ["company_name", "sector", "stage", "location", "focus_area", "founder_type", "team_size", "num_investors"]
            right_keys = ["arr", "burn", "cash", "ltv_cac_ratio", "gross_margin_pct", "monthly_churn_pct", "product_stage_score", "team_score"]
            with prev_cols[0]:
                for k in left_keys:
                    st.write(f"• {k}: {parsed.get(k)}")
            with prev_cols[1]:
                for k in right_keys:
                    st.write(f"• {k}: {parsed.get(k)}")
            with st.expander("Show all extracted fields"):
                st.json(parsed)

        if run_clicked:
            if "pdf_extracted_inputs" not in st.session_state:
                st.warning("Please parse a PDF first.")
            else:
                st.session_state.view = 'analysis'
                st.session_state.inputs = st.session_state["pdf_extracted_inputs"]
                st.session_state.comps_ticker = st.session_state.get("pdf_extracted_ticker") or comps_ticker or "ZOMATO.BSE"
                st.rerun()

# =========================
# Profile + Metrics (wizard)
# =========================
FOCUS_AREA_OPTIONS = {
    "Enhance Urban Lifestyle": ['E-commerce & D2C', 'Consumer Services', 'FinTech', 'PropTech', 'Logistics & Supply Chain', 'Travel & Hospitality', 'Media & Entertainment', 'Gaming'],
    "Live Healthy": ['Digital Health', 'MedTech', 'BioTech'],
    "Mitigate Climate Change": ['Clean Energy', 'EV Mobility', 'AgriTech']
}

def render_company_profile_step() -> Dict[str, Any]:
    section_heading("📝 Step 1 — Company Profile", "Fill this first to unlock Metrics & Financials")
    inputs: Dict[str, Any] = {}
    with card():
        c1, c2 = st.columns(2, gap="large")
        with c1:
            inputs['company_name'] = st.text_input("Company Name", st.session_state.profile_inputs.get('company_name', ''))
            fa_keys = list(FOCUS_AREA_OPTIONS.keys())
            default_fa = st.session_state.profile_inputs.get('focus_area', fa_keys[0])
            if default_fa not in fa_keys:
                default_fa = fa_keys[0]
            inputs['focus_area'] = st.selectbox("Investment Focus Area", fa_keys, index=fa_keys.index(default_fa))
            sector_list = FOCUS_AREA_OPTIONS[inputs['focus_area']]
            default_sector = st.session_state.profile_inputs.get('sector', sector_list[0])
            if default_sector not in sector_list:
                default_sector = sector_list[0]
            inputs['sector'] = st.selectbox("Sector", sector_list, index=sector_list.index(default_sector))
            inputs['location'] = st.selectbox("Location", ["India", "US", "SEA", "MENA", "EU"],
                                              index=(["India", "US", "SEA", "MENA", "EU"].index(st.session_state.profile_inputs.get('location', 'India'))))
        with c2:
            stage_options = ["Pre-Seed", "Seed", "Series A", "Series B"]
            default_stage = st.session_state.profile_inputs.get('stage', "Series A")
            idx = stage_options.index(default_stage) if default_stage in stage_options else 2
            inputs['stage'] = st.selectbox("Funding Stage", stage_options, index=idx)
            ft_values = [p.value for p in FounderPersonality]
            default_ft = st.session_state.profile_inputs.get('founder_type', ft_values[0])
            if default_ft not in ft_values:
                default_ft = ft_values[0]
            inputs['founder_type'] = st.selectbox("Founder Archetype", ft_values, index=ft_values.index(default_ft))
        st.markdown("---")
        inputs['founder_bio'] = st.text_area("Founder Bio", st.session_state.profile_inputs.get('founder_bio', ''), height=100)
        inputs['product_desc'] = st.text_area("Product Description", st.session_state.profile_inputs.get('product_desc', ''), height=100)

        required = ["company_name", "focus_area", "sector", "location", "stage", "founder_type", "product_desc"]
        is_valid = all(bool(inputs.get(k)) for k in required)
        col_next, col_hint = st.columns([1, 3])
        if col_next.button("Next ➡️", disabled=not is_valid, use_container_width=True):
            st.session_state.profile_inputs = inputs
            st.session_state.wizard_step = 1
            st.rerun()
        if not is_valid:
            col_hint.caption("Fill all required fields to continue to Metrics & Financials.")
    return inputs

def render_sector_specific_metrics(focus_area: str, sector: str) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    with card():
        st.markdown("#### Sector-specific Metrics")
        if focus_area == "Enhance Urban Lifestyle":
            if sector == "E-commerce & D2C":
                c1, c2, c3 = st.columns(3)
                extra["aov_inr"] = c1.number_input("Average Order Value (₹)", value=1200, min_value=0)
                extra["monthly_orders"] = c2.number_input("Monthly Orders", value=25000, min_value=0)
                extra["repeat_rate_pct"] = c3.number_input("Repeat Purchase Rate (%)", value=35.0, min_value=0.0, max_value=100.0)
            elif sector == "FinTech":
                c1, c2, c3 = st.columns(3)
                extra["take_rate_pct"] = c1.number_input("Take Rate (%)", value=1.5, min_value=0.0, max_value=100.0, step=0.1)
                extra["transaction_volume_gmv"] = c2.number_input("Monthly Volume (GMV, ₹)", value=1_000_000_000, min_value=0)
                extra["delinquency_rate_pct"] = c3.number_input("Delinquency/Default Rate (%)", value=1.2, min_value=0.0, max_value=100.0, step=0.1)
        elif focus_area == "Live Healthy":
            c1, c2, c3 = st.columns(3)
            extra["regulatory_stage"] = c1.selectbox("Regulatory/Clinical Stage", ["None", "Pre-clinical", "Phase I/II", "Phase III", "Approved"], index=0)
            extra["gross_retention_pct"] = c2.number_input("Gross Revenue Retention (%)", value=90.0, min_value=0.0, max_value=120.0, step=0.5)
            extra["enterprise_sales_cycle_days"] = c3.number_input("Enterprise Sales Cycle (days)", value=120, min_value=0, step=5)
        elif focus_area == "Mitigate Climate Change":
            c1, c2, c3 = st.columns(3)
            extra["trl_level"] = c1.slider("Technology Readiness Level (1-9)", min_value=1, max_value=9, value=6)
            extra["capex_per_unit_inr"] = c2.number_input("CapEx per Unit (₹)", value=500000, min_value=0)
            extra["opex_per_unit_inr"] = c3.number_input("OpEx per Unit (₹/month)", value=5000, min_value=0)
    return extra

def render_metrics_step(profile: Dict[str, Any]) -> Dict[str, Any]:
    section_heading("📊 Step 2 — Metrics & Financials", "Now add the quantitative context")
    inputs: Dict[str, Any] = {}

    with card():
        c1, c2, c3 = st.columns(3)
        inputs['founded_year'] = c1.number_input("Founded Year", 2010, 2025, profile.get('founded_year', 2022))
        inputs['age'] = 2025 - inputs['founded_year']
        inputs['total_funding_usd'] = c2.number_input("Total Funding (USD)", value=5_000_000, min_value=0)
        inputs['team_size'] = c3.number_input("Team Size", value=50, min_value=1)

        c4, c5 = st.columns(2)
        inputs['num_investors'] = c4.number_input("Number of Investors", value=5, min_value=0)
        comps_ticker = c5.text_input("Public Comps Ticker", st.session_state.get("comps_ticker", "ZOMATO.BSE"))

    with card():
        st.markdown("#### Scores & Unit Economics")
        c1, c2, c3, c4 = st.columns(4)
        inputs['product_stage_score'] = c1.slider("Product Stage (0-10)", 0.0, 10.0, 8.0)
        inputs['team_score'] = c2.slider("Team (Execution) (0-10)", 0.0, 10.0, 8.0)
        inputs['moat_score'] = c3.slider("Moat (0-10)", 0.0, 10.0, 7.0)
        inputs['investor_quality_score'] = c4.slider("Investor Quality (1-10)", 1.0, 10.0, 7.0)

        c5, c6, c7 = st.columns(3)
        inputs['ltv_cac_ratio'] = c5.slider("LTV:CAC", 0.1, 10.0, 3.5, 0.1)
        inputs['gross_margin_pct'] = c6.slider("Gross Margin (%)", 0.0, 95.0, 60.0, 1.0)
        inputs['monthly_churn_pct'] = c7.slider("Monthly Revenue Churn (%)", 0.0, 20.0, 2.0, 0.1)

    with card():
        st.markdown("#### Financials & Growth")
        c1, c2, c3 = st.columns(3)
        inputs['arr'] = c1.number_input("Current ARR (₹)", value=80_000_000, min_value=0)
        inputs['burn'] = c2.number_input("Monthly Burn (₹)", value=10_000_000, min_value=0)
        inputs['cash'] = c3.number_input("Cash Reserves (₹)", value=90_000_000, min_value=0)

        c4, c5, c6 = st.columns(3)
        inputs['expected_monthly_growth_pct'] = c4.number_input("Expected Monthly Growth (%)", value=5.0, min_value=-50.0, max_value=200.0, step=0.5)
        inputs['growth_volatility_pct'] = c5.number_input("Growth Volatility (σ, %)", value=3.0, min_value=0.0, max_value=100.0, step=0.5)
        inputs['lead_to_customer_conv_pct'] = c6.number_input("Lead → Customer Conversion (%)", value=5.0, min_value=0.1, max_value=100.0, step=0.1)

        traffic_string = st.text_input("Last 12 Months Web Traffic (comma-separated)",
                                       "5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000")
        try:
            inputs['monthly_web_traffic'] = [int(x.strip()) for x in traffic_string.split(',') if x.strip()]
            if len(inputs['monthly_web_traffic']) != 12:
                st.warning("Please enter exactly 12 comma-separated numbers for web traffic.", icon="⚠️")
        except ValueError:
            st.error("Invalid web traffic. Please enter only comma-separated numbers.", icon="🛑")
            st.stop()

    extra = render_sector_specific_metrics(profile['focus_area'], profile['sector'])
    inputs.update(extra)

    with card():
        col_back, col_run = st.columns([1, 2])
        if col_back.button("⬅️ Back to Company Profile"):
            st.session_state.wizard_step = 0
            st.rerun()
        if col_run.button("🚀 Run AI+ Analysis", use_container_width=True):
            merged = {**profile, **inputs}
            st.session_state.view = 'analysis'
            st.session_state.inputs = merged
            st.session_state.comps_ticker = comps_ticker or "ZOMATO.BSE"
            st.rerun()

    st.session_state.comps_ticker = comps_ticker
    return inputs

# =========================
# Dashboards
# =========================
def memo_to_markdown(memo: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Investment Memo\n")
    if memo.get("executive_summary"):
        lines.append("## Executive Summary")
        lines.append(memo["executive_summary"])
    blocks = [
        "investment_thesis","market","product","traction","unit_economics","gtm","competition","team",
        "risks","catalysts","round_dynamics","use_of_proceeds","valuation_rationale","kpis_next_12m","exit_paths"
    ]
    for k in blocks:
        v = memo.get(k)
        if v:
            lines.append(f"## {k.replace('_',' ').title()}")
            lines.append(str(v))
    lines.append("## Bull Case")
    lines.append(memo.get("bull_case_narrative",""))
    lines.append("## Bear Case")
    lines.append(memo.get("bear_case_narrative",""))
    lines.append(f"\nRecommendation: {memo.get('recommendation','')}, Conviction: {memo.get('conviction','')}")
    return "\n\n".join([x for x in lines if x is not None])

def render_summary_dashboard(report):
    memo = report.get('investment_memo', {}) or {}
    risk = report.get('risk_matrix', {}) or {}
    verdict = report.get('final_verdict', {}) or {}
    ssq = report.get('ssq_report', {}) or {}

    conviction = memo.get('conviction', 'Low')
    rec_icon = {"High": "🚀", "Medium": "👀", "Low": "✋"}.get(conviction, "❓")

    section_heading("Executive Dashboard")
    with card():
        col1, col2, col3 = st.columns([1.5, 1.2, 1.5])
        with col1:
            st.markdown(
                f'<div class="recommendation-card {conviction.lower()}-conviction">'
                f'<h2>{rec_icon} {memo.get("recommendation", "Watchlist")}</h2>'
                f'<p>Conviction: {conviction}</p></div>',
                unsafe_allow_html=True
            )
            st.metric("🎯 Predicted Valuation", verdict.get("predicted_valuation_range_usd", "N/A"))
            probability = verdict.get('success_probability_percent', 0.0)
            if probability > 1:
                probability /= 100.0
            st.metric("📈 Success Probability", f"{probability:.1%}")

        with col2:
            st.metric("⚡ Speedscaling Quotient (SSQ)", f"{ssq.get('ssq_score', 0.0)} / 10")
            with card():
                st.markdown(f"<small>Momentum: <b class='mono'>{ssq.get('momentum', 0)}</b></small>", unsafe_allow_html=True)
                st.progress(ssq.get('momentum', 0) / 10.0)
                st.markdown(f"<small>Efficiency: <b class='mono'>{ssq.get('efficiency', 0)}</b></small>", unsafe_allow_html=True)
                st.progress(ssq.get('efficiency', 0) / 10.0)
                st.markdown(f"<small>Scalability: <b class='mono'>{ssq.get('scalability', 0)}</b></small>", unsafe_allow_html=True)
                st.progress(ssq.get('scalability', 0) / 10.0)

        with col3:
            if risk:
                highest_risk, highest_risk_score = max(risk.items(), key=lambda item: item[1])
            else:
                highest_risk, highest_risk_score = ("N/A", 0)
            rec_score = {"High": 10, "Medium": 6, "Low": 3}.get(conviction, 0)
            deal_score = (rec_score * 0.4) + (ssq.get('ssq_score', 0) * 0.6)
            st.plotly_chart(create_gauge_chart(deal_score, "Overall Deal Score"), use_container_width=True)
            st.metric("🛡️ Highest Risk Factor", f"{highest_risk} ({highest_risk_score:.1f}/10)", delta_color="inverse")

def render_analysis_area(report):
    with card():
        col1, col2 = st.columns([1, 5])
        col1.image("Anthill Logo-Falcon.png", width=120)
        col2.markdown("<h1>Anthill AI+ Evaluation</h1>", unsafe_allow_html=True)

    st.markdown("---")
    prof = report.get('profile', None)
    company_name = getattr(prof, 'company_name', 'Company') if prof else 'Company'
    c1, c2 = st.columns([3, 1])
    c1.header(f"Diagnostic Report: {company_name}")
    if c2.button("⬅️ New Analysis"):
        keys_to_clear = ['report', 'inputs', 'comps_ticker', 'pdf_extracted_inputs', 'pdf_extracted_ticker', 'wizard_step', 'profile_inputs']
        st.session_state.view = 'input'
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    render_summary_dashboard(report)
    st.markdown("---")
    section_heading("Deep Dive Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Investment Memo",
        "📈 Risk & Financial Simulation",
        "🌐 Market Deep-Dive & Comps",
        "🔎 Research & Sources",
        "📥 Submitted Inputs",
        "💸 Fundraise Forecast",
        "🤖 ML Predictions"
    ])

    memo = report.get('investment_memo', {}) or {}
    sim_res = report.get('simulation', {}) or {}
    md = report.get('market_deep_dive', {}) or {}

    with tab1:
        section_heading("Executive Summary")
        with card():
            st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)

        with st.expander("Full IC memo (details)"):
            ic_fields = [
                "investment_thesis", "market", "product", "traction", "unit_economics", "gtm",
                "competition", "team", "risks", "catalysts", "round_dynamics",
                "use_of_proceeds", "valuation_rationale", "kpis_next_12m", "exit_paths"
            ]
            cols = st.columns(2)
            for i, k in enumerate(ic_fields):
                with cols[i % 2]:
                    if memo.get(k):
                        with card():
                            st.markdown(f"**{k.replace('_',' ').title()}**")
                            st.write(memo[k])

        st.markdown("---")
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            with card():
                st.subheader("Bull Case")
                st.markdown(memo.get('bull_case_narrative', 'No bull case generated.'), unsafe_allow_html=True)
        with cc2:
            with card():
                st.subheader("Bear Case")
                st.markdown(memo.get('bear_case_narrative', 'No bear case generated.'), unsafe_allow_html=True)

        # Download memo as Markdown
        with card():
            md_text = memo_to_markdown(memo)
            st.download_button("⬇️ Download Memo (Markdown)", data=md_text, file_name=f"{company_name}_Investment_Memo.md", mime="text/markdown")

    with tab2:
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            with card():
                st.plotly_chart(create_spider_chart(report.get('risk_matrix', {}), "Heuristic Risk Profile"), use_container_width=True)
        with cc2:
            with card():
                st.subheader("Financial Runway Simulation")
                st.info(sim_res.get('narrative_summary', 'Simulation not available.'))
                ts = sim_res.get('time_series_data', pd.DataFrame())
                if not ts.empty:
                    st.line_chart(ts.set_index('Month'))

    with tab3:
        with card():
            section_heading("Market Deep-Dive (Context)")
            trends = md.get('indian_funding_trends', {})
            news = md.get('recent_news', {}).get('news', [])
            india_ctx = md.get('india_funding_dataset_context', {})
            if trends:
                with card():
                    st.markdown("#### 🇮🇳 Funding Trends (Web)")
                    st.json(trends)
            if news:
                with card():
                    st.markdown("#### 🗞️ Recent News")
                    for n in news[:10]:
                        st.markdown(f"- [{n.get('title','(untitled)')}]({n.get('url','#')}) — {n.get('published_date','')}")
            if india_ctx:
                with card():
                    st.markdown("#### 📊 India Funding Dataset Context (Kaggle)")
                    st.json(india_ctx)

    with tab4:
        with card():
            display_research(md)

    with tab5:
        with card():
            st.json(st.session_state.get('inputs', {}))

    with tab6:
        with card():
            st.subheader("Fundraise Forecast")
            forecast = report.get('fundraise_forecast', {}) or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Round Likelihood (6 months)", f"{forecast.get('round_likelihood_6m', 0.0):.1%}")
            c2.metric("Round Likelihood (12 months)", f"{forecast.get('round_likelihood_12m', 0.0):.1%}")
            c3.metric("Expected Time to Next Round", f"{forecast.get('expected_time_to_next_round_months', 0.0):.1f} months")
            st.caption("Probabilities calibrate to online model when available; otherwise robust heuristics are applied.")

    with tab7:
        with card():
            st.subheader("Online Model Results")
            ml = report.get("ml_predictions", {}) or {}
            online = ml.get("online", {}) or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Round Likelihood (12m, online)", f"{online.get('round_probability_12m', 0.0):.1%}")
            c2.metric("Round Likelihood (6m, online)", f"{online.get('round_probability_6m', 0.0):.1%}")
            val = online.get("predicted_valuation_usd", None)
            c3.metric("Predicted Next Valuation (USD, online)", f"${val:,.0f}" if isinstance(val, (int, float)) and val else "N/A")

# =========================
# Main
# =========================
async def main():
    engine = load_engine()

    if st.session_state.view == 'input':
        # Header
        with card():
            col1, col2 = st.columns([1, 5])
            col1.image("Anthill Logo-Falcon.png", width=120)
            col2.markdown("<h1>Anthill AI+ Evaluation</h1>", unsafe_allow_html=True)

        # Quick PDF path (optional)
        render_pdf_quick_analysis()
        st.markdown("---")

        # Step indicator
        with card():
            st.markdown("### Setup Steps")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("1) Company Profile — required")
            with cols[1]:
                st.markdown("2) Metrics & Financials — unlocked after profile")

        # Wizard
        if st.session_state.wizard_step == 0:
            render_company_profile_step()
        else:
            profile_inputs = st.session_state.get("profile_inputs", {})
            render_metrics_step(profile_inputs)

    elif st.session_state.view == 'analysis':
        with st.spinner("Calculating SSQ... Running deep web research... Building IC-grade memo..."):
            try:
                report = await engine.comprehensive_analysis(
                    st.session_state.inputs,
                    st.session_state.get('comps_ticker', 'ZOMATO.BSE')
                )
                st.session_state.report = report
                render_analysis_area(report)
            except Exception as e:
                st.error("A critical error occurred during analysis. Please try again.")
                logging.error("Analysis failed", exc_info=True)
                if st.button("⬅️ Try Again"):
                    st.session_state.view = 'input'
                    st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
