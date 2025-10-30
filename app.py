import os
import logging
import asyncio
from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from next_gen_vc_engine import NextGenVCEngine, FounderPersonality
from ui_theme import apply_theme, hero, section_heading, card
from services.pdf_ingest import PDFIngestor

# =========================
# Global theme + compact layout
# =========================
apply_theme(page_title="Anthill AI+ Evaluation", page_icon="🦅")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { height: 0px; padding: 0; background: transparent; }
    .block-container { padding-top: 0.35rem !important; padding-bottom: 0.75rem !important; max-width: 1200px; }
    [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem !important; }
    .stTabs [data-baseweb="tab"] { padding: 0.28rem 0.5rem !important; }
    div[data-testid="stMetric"] { margin-bottom: 0.2rem !important; }
    .stTextInput, .stSelectbox, .stTextArea, .stNumberInput, .stSlider { margin-bottom: 0.35rem !important; }
    .stButton > button { padding: 0.42rem 0.75rem !important; border-radius: 8px !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; }
    .stMarkdown p { margin: 0.20rem 0 !important; }
    .css-1dp5vir, .e1f1d6gn3 { margin: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Wizard state
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
if "profile_inputs" not in st.session_state:
    st.session_state.profile_inputs = {}
if "view" not in st.session_state:
    st.session_state.view = "input"

# Env passthrough
for k in ["FX_INR_PER_USD", "RESEARCH_API_URL", "RESEARCH_MODEL", "RESEARCH_RESPONSES_URL"]:
    try:
        val = st.secrets.get(k)
        if val:
            os.environ[k] = str(val)
    except Exception:
        pass

hero("Anthill AI+ Evaluation", "VC copilot with research‑amplified memos and calibrated valuations.")

@st.cache_resource
def load_engine():
    try:
        return NextGenVCEngine(
            st.secrets.get("TAVILY_API_KEY"),
            st.secrets.get("ALPHA_VANTAGE_KEY"),
            None,
            st.secrets.get("RESEARCH_API_KEY") or st.secrets.get("GROK_API_KEY"),
        )
    except Exception as e:
        st.error("Could not initialize the engine. Check API keys.")
        logging.exception(e)
        st.stop()

# =========================
# Charts
# =========================
def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={'text': title, 'font': {'size': 14}},
                                 number={'font': {'size': 20}},
                                 gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "#0A84FF"},
                                        'bgcolor': "rgba(255,255,255,0)", 'borderwidth': 0,
                                        'steps': [{'range': [0, 3.3], 'color': 'rgba(255,59,48,0.10)'},
                                                  {'range': [3.3, 6.6], 'color': 'rgba(255,204,0,0.10)'},
                                                  {'range': [6.6, 10], 'color': 'rgba(52,199,89,0.10)'}]}))
    fig.update_layout(template="simple_white", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0), height=200)
    return fig

def create_spider_chart(data, title):
    if not isinstance(data, dict) or not data:
        data = {"Market": 5, "Execution": 5, "Technology": 5, "Regulatory": 5, "Competition": 5}
    fig = go.Figure(go.Scatterpolar(r=list(data.values()), theta=list(data.keys()), fill='toself',
                                    line=dict(color='#0A84FF', width=2), fillcolor='rgba(10,132,255,0.15)'))
    fig.update_layout(template="simple_white", title=dict(text=title, font=dict(size=16)),
                      polar=dict(bgcolor='rgba(255,255,255,0)', radialaxis=dict(visible=True, range=[0, 10])),
                      paper_bgcolor='rgba(255,255,255,0)', margin=dict(l=0, r=0, t=30, b=0), height=260)
    return fig

# =========================
# Research view
# =========================
def _render_paragraphs(title: str, content: str):
    if not content: return
    st.markdown(f"**{title}**")
    parts = [p.strip() for p in content.replace("\r\n", "\n").split("\n\n") if p.strip()]
    for p in parts:
        st.markdown(p)

def display_research(md: Dict[str, Any]):
    ext = md.get("external_research", {}) or {}

    section_heading("Research & Sources")

    if "notice" in ext:
        st.info(ext["notice"]); return
    if "error" in ext:
        st.info("External research was unavailable. Please try again."); return

    if ext.get("summary"):
        with card(): _render_paragraphs("Summary", ext["summary"])

    sections = ext.get("sections", {}) or {}
    if sections:
        colL, colR = st.columns(2, gap="small")
        left = ["overview", "products", "business_model", "gtm", "unit_economics", "funding", "investors", "leadership"]
        right = ["hiring", "traction", "customers", "pricing", "competitors", "moat", "partnerships", "regulatory", "risks", "tech_stack", "roadmap"]
        with colL:
            for k in left:
                if sections.get(k):
                    with card():
                        _render_paragraphs(k.replace("_"," ").title(), sections[k])
        with colR:
            for k in right:
                if sections.get(k):
                    with card():
                        _render_paragraphs(k.replace("_"," ").title(), sections[k])

    sources = ext.get("sources", []) or []
    if sources:
        with card():
            st.markdown("**Citations**")
            for i, s in enumerate(sources[:20], start=1):
                title = s.get("title", "Source"); url = s.get("url", "#")
                snippet = s.get("snippet", ""); conf = s.get("confidence", None)
                conf_str = f" · conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
                st.markdown(f"- [{i}] [{title}]({url}) — {snippet}{conf_str}")

# =========================
# PDF quick path (no ticker)
# =========================
def render_pdf_quick_analysis():
    section_heading("Quick Analysis From PDF", "Upload a deck and generate the full evaluation")
    uploaded = st.file_uploader("Upload PDF deck", type=["pdf"], key="pdf_uploader_firstpage")
    run_clicked = st.button("Run Analysis from PDF", use_container_width=True)
    if run_clicked:
        if uploaded is None:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Parsing PDF and preparing inputs..."):
                try:
                    ingestor = PDFIngestor(st.secrets.get("GEMINI_API_KEY"))
                    extracted = ingestor.extract(uploaded.getvalue(), file_name=getattr(uploaded, "name", None))
                    st.session_state.view = 'analysis'
                    st.session_state.inputs = extracted
                    st.rerun()
                except Exception:
                    st.error("We couldn't parse the PDF. Try another file.")

# =========================
# Wizard steps
# =========================
FOCUS_AREA_OPTIONS = {
    "Enhance Urban Lifestyle": ['E-commerce & D2C', 'Consumer Services', 'FinTech', 'PropTech', 'Logistics & Supply Chain', 'Travel & Hospitality', 'Media & Entertainment', 'Gaming'],
    "Live Healthy": ['Digital Health', 'MedTech', 'BioTech'],
    "Mitigate Climate Change": ['Clean Energy', 'EV Mobility', 'AgriTech']
}

def render_company_profile_step() -> Dict[str, Any]:
    section_heading("Step 1 — Company Profile", "Complete this to unlock Metrics")
    inputs: Dict[str, Any] = {}
    c1, c2 = st.columns(2, gap="small")
    with c1:
        inputs['company_name'] = st.text_input("Company Name", st.session_state.profile_inputs.get('company_name', ''), key="pi_company_name")
        fa_keys = list(FOCUS_AREA_OPTIONS.keys())
        default_fa = st.session_state.profile_inputs.get('focus_area', fa_keys[0])
        if default_fa not in fa_keys: default_fa = fa_keys[0]
        inputs['focus_area'] = st.selectbox("Focus Area", fa_keys, index=fa_keys.index(default_fa), key="pi_focus_area")
        sector_list = FOCUS_AREA_OPTIONS[inputs['focus_area']]
        default_sector = st.session_state.profile_inputs.get('sector', sector_list[0])
        if default_sector not in sector_list: default_sector = sector_list[0]
        inputs['sector'] = st.selectbox("Sector", sector_list, index=sector_list.index(default_sector), key="pi_sector")
        inputs['location'] = st.selectbox("Location", ["India", "US", "SEA", "MENA", "EU"],
                                          index=(["India", "US", "SEA", "MENA", "EU"].index(st.session_state.profile_inputs.get('location', 'India'))),
                                          key="pi_location")
    with c2:
        stage_options = ["Pre-Seed", "Seed", "Series A", "Series B"]
        default_stage = st.session_state.profile_inputs.get('stage', "Series A")
        idx = stage_options.index(default_stage) if default_stage in stage_options else 2
        inputs['stage'] = st.selectbox("Funding Stage", stage_options, index=idx, key="pi_stage")
        ft_values = [p.value for p in FounderPersonality]
        default_ft = st.session_state.profile_inputs.get('founder_type', ft_values[0])
        if default_ft not in ft_values: default_ft = ft_values[0]
        inputs['founder_type'] = st.selectbox("Founder Archetype", ft_values, index=ft_values.index(default_ft), key="pi_founder_type")
        inputs['founder_linkedin_url'] = st.text_input("Founder LinkedIn URL", st.session_state.profile_inputs.get('founder_linkedin_url', ''), placeholder="https://www.linkedin.com/in/...", key="pi_founder_linkedin")
        inputs['founder_x_url'] = st.text_input("Founder X (Twitter) URL (optional)", st.session_state.profile_inputs.get('founder_x_url', ''), placeholder="https://x.com/username", key="pi_founder_x")

    required = ["company_name", "focus_area", "sector", "location", "stage", "founder_type", "founder_linkedin_url"]
    is_valid = all(bool(inputs.get(k)) for k in required)
    col_next, _ = st.columns([1, 5], gap="small")
    if col_next.button("Next →", disabled=not is_valid, use_container_width=True, key="pi_next"):
        if is_valid:
            st.session_state.profile_inputs = inputs
            st.session_state.wizard_step = 1
            st.rerun()
    if not is_valid:
        st.caption("Fill all required fields (including founder LinkedIn) to continue.")
    return inputs

def render_ssq_deep_dive_inputs() -> Dict[str, Any]:
    section_heading("Speed Scaling Deep‑Dive", "These factors feed the SSQ deep-dive score")
    f: Dict[str, Any] = {}
    # Market & Moat
    st.markdown("**Market & Moat**")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    f["maximum_market_size_usd"]   = c1.number_input("Max Market Size (USD)", value=500_000_000, min_value=0, key="ssq_max_market_size_usd")
    f["market_growth_rate_pct"]    = c2.number_input("Market Growth Rate (%)", value=25.0, min_value=-100.0, max_value=500.0, step=0.5, key="ssq_market_growth_rate_pct")
    f["economic_condition_index"]  = c3.slider("Economic Condition (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_economic_condition_index")
    f["readiness_index"]           = c4.slider("Readiness (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_readiness_index")

    c5, c6, c7, c8 = st.columns(4, gap="small")
    f["originality_index"]         = c5.slider("Originality (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_originality_index")
    f["need_index"]                = c6.slider("Need (0–10)", 0.0, 10.0, 8.0, 0.1, key="ssq_need_index")
    f["testing_index"]             = c7.slider("Testing (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_testing_index")
    f["pmf_index"]                 = c8.slider("PMF (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_pmf_index")

    c9, c10, c11, c12 = st.columns(4, gap="small")
    f["scalability_index"]         = c9.slider("Scalability (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_scalability_index")
    f["technology_duplicacy_index"]= c10.slider("Tech Duplicacy (0–10)", 0.0, 10.0, 4.0, 0.1, key="ssq_technology_duplicacy_index")
    f["execution_duplicacy_index"] = c11.slider("Execution Duplicacy (0–10)", 0.0, 10.0, 4.0, 0.1, key="ssq_execution_duplicacy_index")
    f["first_mover_advantage_index"]= c12.slider("First Mover Advantage (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_fma_index")

    c13, c14, c15, c16 = st.columns(4, gap="small")
    f["barriers_to_entry_index"]   = c13.slider("Barriers to Entry (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_barriers_to_entry_index")
    f["num_close_competitors"]     = c14.number_input("Close Competitors (count)", value=5, min_value=0, key="ssq_num_close_competitors")
    f["price_advantage_pct"]       = c15.number_input("Price Advantage (%)", value=10.0, min_value=-100.0, max_value=100.0, step=0.5, key="ssq_price_advantage_pct")
    f["channels_of_promotion"]     = c16.number_input("Channels of Promotion (count)", value=3, min_value=0, max_value=25, key="ssq_channels_of_promotion")

    # GTM & Revenue
    st.markdown("**GTM & Revenue**")
    g1, g2, g3, g4 = st.columns(4, gap="small")
    f["mrr_inr"]                    = g1.number_input("MRR (₹)", value=6_000_000, min_value=0, key="ssq_mrr_inr")
    f["sales_growth_pct"]           = g2.number_input("Sales Growth (YoY, %)", value=80.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_sales_growth_pct")
    f["lead_to_close_ratio_pct"]    = g3.number_input("Lead→Close Ratio (%)", value=12.0, min_value=0.0, max_value=100.0, step=0.1, key="ssq_lead_to_close_ratio_pct")
    f["marketing_spend_inr"]        = g4.number_input("Marketing Spend / mo (₹)", value=2_000_000, min_value=0, key="ssq_marketing_spend_inr")

    g5, g6, g7 = st.columns(3, gap="small")
    f["ltv_cac_ratio"]              = g5.slider("LTV:CAC", 0.1, 10.0, 3.5, 0.1, key="ssq_ltv_cac_ratio")
    f["customer_growth_pct"]        = g6.number_input("Customer Growth (YoY, %)", value=100.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_customer_growth_pct")
    f["repurchase_ratio_pct"]       = g7.number_input("Repurchase Ratio (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="ssq_repurchase_ratio_pct")

    # Team & ops
    st.markdown("**Team & Ops**")
    t1, t2, t3, t4 = st.columns(4, gap="small")
    f["domain_experience_years"]    = t1.number_input("Domain Experience (yrs)", value=6, min_value=0, max_value=40, key="ssq_domain_experience_years")
    f["quality_of_experience_index"]= t2.slider("Quality of Experience (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_quality_of_experience_index")
    f["team_size"]                  = t3.number_input("Team Size", value=50, min_value=1, key="ssq_team_size")
    f["avg_salary_inr"]             = t4.number_input("Avg Salary / yr (₹)", value=1_500_000, min_value=0, key="ssq_avg_salary_inr")

    t5, t6, t7, t8 = st.columns(4, gap="small")
    f["equity_dilution_pct"]        = t5.number_input("Equity Dilution to Date (%)", value=22.0, min_value=0.0, max_value=100.0, step=0.5, key="ssq_equity_dilution_pct")
    f["runway_months"]              = t6.number_input("Runway (months)", value=12.0, min_value=0.0, max_value=120.0, step=0.5, key="ssq_runway_months")
    f["cac_inr"]                    = t7.number_input("CAC (₹)", value=8_000, min_value=0, key="ssq_cac_inr")
    f["variance_analysis_index"]    = t8.slider("Variance Analysis (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_variance_analysis_index")

    # Finance
    st.markdown("**Finance & Ratios**")
    f1, f2, f3, f4 = st.columns(4, gap="small")
    f["mrr_growth_rate_pct"]        = f1.number_input("MRR Growth Rate (YoY, %)", value=90.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_mrr_growth_rate_pct")
    f["de_ratio"]                   = f2.number_input("D/E Ratio", value=0.2, min_value=0.0, max_value=10.0, step=0.05, key="ssq_de_ratio")
    f["gpm_pct"]                    = f3.number_input("Gross Profit Margin (%)", value=60.0, min_value=-100.0, max_value=100.0, step=0.5, key="ssq_gpm_pct")
    f["nr_inr"]                     = f4.number_input("Net Revenue (₹)", value=120_000_000, min_value=0, key="ssq_nr_inr")

    f5, f6, f7 = st.columns(3, gap="small")
    f["net_income_ratio_pct"]       = f5.number_input("Net Income Ratio (%)", value=5.0, min_value=-200.0, max_value=200.0, step=0.5, key="ssq_net_income_ratio_pct")
    f["filed_patents"]              = f6.number_input("Filed Patents (count)", value=2, min_value=0, max_value=200, key="ssq_filed_patents")
    f["approved_patents"]           = f7.number_input("Approved Patents (count)", value=0, min_value=0, max_value=200, key="ssq_approved_patents")
    return f

def render_outcomes_section() -> Dict[str, Any]:
    section_heading("Desired Outcomes from Investment", "Targets and milestones to bake into the IC memo")
    outcomes: Dict[str, Any] = {}
    c1, c2, c3 = st.columns(3, gap="small")
    outcomes["target_arr_usd"]         = c1.number_input("Target ARR (USD)", value=5_000_000, min_value=0, key="outcome_target_arr_usd")
    outcomes["target_burn_multiple"]   = c2.number_input("Target Burn Multiple", value=1.5, min_value=0.0, max_value=20.0, step=0.1, key="outcome_target_burn_multiple")
    outcomes["target_nrr_pct"]         = c3.number_input("Target NRR (%)", value=115.0, min_value=0.0, max_value=500.0, step=0.5, key="outcome_target_nrr_pct")
    c4, c5, c6 = st.columns(3, gap="small")
    outcomes["target_gm_pct"]          = c4.number_input("Target Gross Margin (%)", value=70.0, min_value=-100.0, max_value=100.0, step=0.5, key="outcome_target_gm_pct")
    outcomes["target_runway_months"]   = c5.number_input("Target Runway (months)", value=18.0, min_value=0.0, max_value=120.0, step=0.5, key="outcome_target_runway_months")
    outcomes["milestones"]             = c6.text_input("Top 3 Milestones (comma-separated)", "Marquee logos, New geography, Platform launch", key="outcome_milestones")
    return outcomes

def render_metrics_step(profile: Dict[str, Any]) -> Dict[str, Any]:
    section_heading("Step 2 — Metrics & Financials", "Quantitative context")
    inputs: Dict[str, Any] = {}

    c1, c2, c3 = st.columns(3, gap="small")
    inputs['founded_year']          = c1.number_input("Founded Year", 2010, 2025, profile.get('founded_year', 2022), key="metrics_founded_year")
    inputs['total_funding_usd']     = c2.number_input("Total Funding (USD)", value=5_000_000, min_value=0, key="metrics_total_funding_usd")
    inputs['team_size']             = c3.number_input("Team Size", value=50, min_value=1, key="metrics_team_size")

    c4, c5, c6, c7 = st.columns(4, gap="small")
    inputs['product_stage_score']   = c4.slider("Product Stage (0-10)", 0.0, 10.0, 8.0, key="metrics_product_stage_score")
    inputs['team_score']            = c5.slider("Execution (0-10)", 0.0, 10.0, 8.0, key="metrics_team_score")
    inputs['moat_score']            = c6.slider("Moat (0-10)", 0.0, 10.0, 7.0, key="metrics_moat_score")
    inputs['investor_quality_score']= c7.slider("Investor Quality (1-10)", 1.0, 10.0, 7.0, key="metrics_investor_quality_score")

    st.markdown("**AI Scoring Options (subjectives)**")
    a1, a2 = st.columns(2, gap="small")
    inputs["ai_score_team_execution"]   = a1.checkbox("Use AI to score Team Execution", value=False, key="metrics_ai_score_team_execution")
    inputs["ai_score_investor_quality"] = a2.checkbox("Use AI to score Investor Quality", value=False, key="metrics_ai_score_investor_quality")
    e1, e2 = st.columns(2, gap="small")
    inputs["team_ai_evidence"]          = e1.text_area("Team Evidence (links, bios, achievements)", "", height=70, key="metrics_team_ai_evidence")
    inputs["investor_ai_evidence"]      = e2.text_area("Investor Evidence (cap table, fund brands, track record)", "", height=70, key="metrics_investor_ai_evidence")

    c8, c9, c10 = st.columns(3, gap="small")
    inputs['ltv_cac_ratio']         = c8.slider("LTV:CAC", 0.1, 10.0, 3.5, 0.1, key="metrics_ltv_cac_ratio")
    inputs['gross_margin_pct']      = c9.slider("Gross Margin (%)", 0.0, 95.0, 60.0, 1.0, key="metrics_gross_margin_pct")
    inputs['monthly_churn_pct']     = c10.slider("Monthly Churn (%)", 0.0, 20.0, 2.0, 0.1, key="metrics_monthly_churn_pct")

    c11, c12, c13 = st.columns(3, gap="small")
    inputs['arr']                   = c11.number_input("Current ARR (₹)", value=80_000_000, min_value=0, key="metrics_arr")
    inputs['burn']                  = c12.number_input("Monthly Burn (₹)", value=10_000_000, min_value=0, key="metrics_burn")
    inputs['cash']                  = c13.number_input("Cash Reserves (₹)", value=90_000_000, min_value=0, key="metrics_cash")

    c14, c15, c16 = st.columns(3, gap="small")
    inputs['expected_monthly_growth_pct'] = c14.number_input("Expected Monthly Growth (%)", value=5.0, min_value=-50.0, max_value=200.0, step=0.5, key="metrics_expected_monthly_growth_pct")
    inputs['growth_volatility_pct']       = c15.number_input("Growth Volatility σ (%)", value=3.0, min_value=0.0, max_value=100.0, step=0.5, key="metrics_growth_volatility_pct")
    inputs['lead_to_customer_conv_pct']   = c16.number_input("Lead→Customer Conversion (%)", value=5.0, min_value=0.1, max_value=100.0, step=0.1, key="metrics_lead_to_customer_conv_pct")

    traffic_string = st.text_input("Last 12 Months Web Traffic (comma-separated)",
                                   "5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000",
                                   key="metrics_monthly_web_traffic_text")
    try:
        inputs['monthly_web_traffic'] = [int(x.strip()) for x in traffic_string.split(',') if x.strip()]
        if len(inputs['monthly_web_traffic']) != 12:
            st.caption("Enter exactly 12 values for web traffic.")
    except ValueError:
        st.error("Invalid web traffic. Use comma-separated numbers.")
        st.stop()

    # AI valuation assist toggle
    inputs["use_ai_valuation_assist"] = st.checkbox("Use AI to assist valuation multiples (sector/stage peers)", value=True, key="metrics_use_ai_valuation_assist")

    # SSQ Deep-Dive + AI weighting
    inputs["ssq_deep_dive_factors"]   = render_ssq_deep_dive_inputs()
    inputs["use_ai_ssq_weights"]      = st.checkbox("Use AI to weigh SSQ factors", value=True, key="metrics_use_ai_ssq_weights")

    # Desired outcomes section
    inputs["desired_outcomes"]        = render_outcomes_section()

    # Actions
    col_back, col_run = st.columns([1, 2], gap="small")
    if col_back.button("← Back", key="metrics_back_btn"):
        st.session_state.wizard_step = 0
        st.rerun()
    if col_run.button("Run Analysis", use_container_width=True, key="metrics_run_btn"):
        merged = {**profile, **inputs}
        st.session_state.view = 'analysis'
        st.session_state.inputs = merged
        st.rerun()

    return inputs

# =========================
# Dashboards and analysis
# =========================
def memo_to_markdown(memo: Dict[str, Any]) -> str:
    lines = ["# Investment Memo"]
    if memo.get("executive_summary"):
        lines += ["## Executive Summary", memo["executive_summary"]]
    blocks = ["investment_thesis","market","product","traction","unit_economics","gtm","competition","team","risks","catalysts","round_dynamics","use_of_proceeds","valuation_rationale","kpis_next_12m","exit_paths"]
    for k in blocks:
        v = memo.get(k)
        if v: lines += [f"## {k.replace('_',' ').title()}", str(v)]
    if memo.get("speed_scaling_deep_dive"):
        lines += ["## Speed Scaling Deep-Dive", str(memo["speed_scaling_deep_dive"])]
    lines += ["## Bull Case", memo.get("bull_case_narrative",""), "## Bear Case", memo.get("bear_case_narrative",""),
              f"\nRecommendation: {memo.get('recommendation','')}, Conviction: {memo.get('conviction','')}"]
    return "\n\n".join([x for x in lines if x is not None])

def render_summary_dashboard(report):
    memo = report.get('investment_memo', {}) or {}
    risk = report.get('risk_matrix', {}) or {}
    verdict = report.get('final_verdict', {}) or {}
    ssq = report.get('ssq_report', {}) or {}

    conviction = memo.get('conviction', 'Low')
    rec_icon = {"High": "🚀", "Medium": "👀", "Low": "✋"}.get(conviction, "❓")

    section_heading("Executive Dashboard")
    row1 = st.columns([1.1, 0.9, 1.0], gap="small")
    with row1[0]:
        with card():
            st.markdown(f"### {rec_icon} {memo.get('recommendation','Watchlist')}")
            st.caption(f"Conviction: {conviction}")
            st.metric("Valuation", verdict.get("predicted_valuation_range_usd", "N/A"))
            p = verdict.get('success_probability_percent', 0.0); p = p/100.0 if p > 1 else p
            st.metric("Success Probability", f"{p:.1%}")
    with row1[1]:
        with card():
            st.markdown("### SSQ")
            st.metric("Score", f"{ssq.get('ssq_score', 0.0)} / 10")
            st.progress(ssq.get('ssq_score', 0.0) / 10.0)
            st.caption(f"Momentum {ssq.get('momentum',0)} · Efficiency {ssq.get('efficiency',0)} · Scalability {ssq.get('scalability',0)}")
    with row1[2]:
        with card():
            risk_item = ("N/A", 0)
            if risk: risk_item = max(risk.items(), key=lambda x: x[1])
            st.plotly_chart(create_gauge_chart(({"High":10,"Medium":6,"Low":3}.get(conviction,5)*0.4 + ssq.get('ssq_score', 0)*0.6), "Overall Deal Score"), use_container_width=True)
            st.caption(f"Highest Risk: {risk_item[0]} ({risk_item[1]:.1f}/10)")

def render_analysis_area(report):
    prof = report.get('profile', None)
    company_name = getattr(prof, 'company_name', 'Company') if prof else 'Company'
    header_cols = st.columns([3, 1], gap="small")
    header_cols[0].markdown(f"## Diagnostic Report: {company_name}")
    if header_cols[1].button("New Analysis", use_container_width=True):
        keys_to_clear = ['report', 'inputs', 'wizard_step', 'profile_inputs']
        st.session_state.view = 'input'
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()

    render_summary_dashboard(report)
    section_heading("Deep Dive")

    tabs = st.tabs(["Investment Memo", "Risk & Simulation", "Research & Sources", "Founder Profile", "SSQ Deep‑Dive", "Inputs", "Forecast", "ML"])
    memo = report.get('investment_memo', {}) or {}
    sim_res = report.get('simulation', {}) or {}
    md = report.get('market_deep_dive', {}) or {}
    founder_prof = report.get('founder_profile', {}) or {}

    with tabs[0]:
        st.markdown("### Executive Summary")
        with card(): st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)
        with st.expander("Full memo"):
            cols = st.columns(2, gap="small")
            detail_fields = ["investment_thesis","market","product","traction","unit_economics","gtm","competition","team","risks","catalysts","round_dynamics","use_of_proceeds","valuation_rationale","kpis_next_12m","exit_paths"]
            for i, k in enumerate(detail_fields):
                if memo.get(k):
                    with cols[i % 2]:
                        st.markdown(f"**{k.replace('_',' ').title()}**")
                        st.write(memo[k])
        st.download_button("Download Memo (Markdown)", data=memo_to_markdown(memo), file_name=f"{company_name}_Investment_Memo.md", mime="text/markdown")

    with tabs[1]:
        cols = st.columns([1,1], gap="small")
        with cols[0]:
            st.plotly_chart(create_spider_chart(report.get('risk_matrix', {}), "Risk Profile"), use_container_width=True)
        with cols[1]:
            st.markdown("### Financial Runway")
            st.info(sim_res.get('narrative_summary', 'Simulation not available.'))
            ts = sim_res.get('time_series_data', pd.DataFrame())
            if not ts.empty: st.line_chart(ts.set_index('Month'))

    with tabs[2]:
        display_research(md)

    with tabs[3]:
        st.markdown("### Founder Profile (LinkedIn + X)")
        if founder_prof:
            with card():
                st.write(founder_prof.get("summary", ""))
                c1, c2, c3 = st.columns(3, gap="small")
                c1.metric("Experience (yrs)", founder_prof.get("experience_years", "N/A"))
                c2.metric("Network Strength (0–10)", founder_prof.get("network_strength", "N/A"))
                c3.metric("Execution Signals (0–10)", founder_prof.get("execution_signals", "N/A"))
            with st.expander("Details"):
                st.json(founder_prof)
        else:
            st.info("No founder profile analysis available. Provide a LinkedIn URL in the Company Profile step.")

    with tabs[4]:
        st.markdown("### Speed Scaling Deep‑Dive")
        deep = report.get("ssq_deep_dive", {}) or {}
        if deep:
            with card():
                st.metric("SSQ Deep‑Dive Score", deep.get("ssq_deep_dive_score", 0))
                st.caption(f"AI Adjustment: {deep.get('ai_adjustment', 0)}")
            with st.expander("Per‑factor scores and weights", expanded=False):
                st.json(deep)

    with tabs[5]:
        st.json(st.session_state.get('inputs', {}))

    with tabs[6]:
        forecast = report.get('fundraise_forecast', {}) or {}
        col1, col2, col3 = st.columns(3, gap="small")
        col1.metric("Round Likelihood (6m)", f"{forecast.get('round_likelihood_6m', 0.0):.1%}")
        col2.metric("Round Likelihood (12m)", f"{forecast.get('round_likelihood_12m', 0.0):.1%}")
        col3.metric("Time to Next Round", f"{forecast.get('expected_time_to_next_round_months', 0.0):.1f} months")

    with tabs[7]:
        ml = report.get("ml_predictions", {}) or {}
        online = ml.get("online", {}) or {}
        col1, col2, col3 = st.columns(3, gap="small")
        col1.metric("Online Likelihood (12m)", f"{online.get('round_probability_12m', 0.0):.1%}")
        col2.metric("Online Likelihood (6m)", f"{online.get('round_probability_6m', 0.0):.1%}")
        val = online.get("predicted_valuation_usd", None)
        col3.metric("Next Valuation (USD)", f"${val:,.0f}" if isinstance(val, (int, float)) and val else "N/A")

# =========================
# Main
# =========================
async def main():
    engine = load_engine()

    if st.session_state.view == 'input':
        render_pdf_quick_analysis()
        step_cols = st.columns(2, gap="small")
        step_cols[0].markdown("**1) Company Profile** ✅" if st.session_state.wizard_step == 1 else "**1) Company Profile**")
        step_cols[1].markdown("**2) Metrics & Financials**" + (" ✅" if st.session_state.wizard_step == 1 else ""))

        if st.session_state.wizard_step == 0:
            render_company_profile_step()
        else:
            render_metrics_step(st.session_state.get("profile_inputs", {}))

    elif st.session_state.view == 'analysis':
        with st.spinner("Running deep research, founder profiling, AI scoring, and valuation..."):
            try:
                report = await engine.comprehensive_analysis(
                    st.session_state.inputs,
                    ""  # comps removed
                )
                st.session_state.report = report
                render_analysis_area(report)
            except Exception as e:
                st.error("Analysis failed. Please try again.")
                logging.error("Analysis failed", exc_info=True)
                if st.button("Try Again"):
                    st.session_state.view = 'input'
                    st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
