import os
import math
import logging
import asyncio
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from next_gen_vc_engine import NextGenVCEngine, FounderPersonality
from ui_theme import apply_theme, hero, section_heading, card
from services.pdf_ingest import PDFIngestor

# =========================
# Plotly theme (consistent fonts/colors)
# =========================
PALETTE = {
    "primary": "#0A84FF",
    "accent": "#6E56CF",
    "success": "#34C759",
    "warn": "#FF9F0A",
    "danger": "#FF3B30",
    "muted": "#8E8E93",
}
pio.templates["anthill"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, system-ui, sans-serif", size=12, color="#0f1221"),
        title=dict(font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,1)",
        colorway=[PALETTE["primary"], PALETTE["accent"], PALETTE["success"], PALETTE["warn"], PALETTE["danger"], "#2BBBAD", "#FFA07A"],
        margin=dict(l=8, r=8, t=36, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
)
pio.templates.default = "anthill+plotly_white"

# =========================
# Global theme + compact layout
# =========================
apply_theme(page_title="Anthill AI+ Evaluation", page_icon="🦅")
hero("Anthill AI+ Evaluation", "VC copilot with research‑amplified memos and calibrated valuations.")

# =========================
# State and secrets
# =========================
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
if "profile_inputs" not in st.session_state:
    st.session_state.profile_inputs = {}
if "view" not in st.session_state:
    st.session_state.view = "input"

for k in ["FX_INR_PER_USD", "RESEARCH_API_URL", "RESEARCH_MODEL", "RESEARCH_RESPONSES_URL"]:
    try:
        val = st.secrets.get(k)
        if val:
            os.environ[k] = str(val)
    except Exception:
        pass

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
# Chart helpers
# =========================
def create_gauge_chart(score, title, height=180, key=None):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score) if isinstance(score, (int, float)) else 0.0,
            title={'text': title, 'font': {'size': 13}},
            number={'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': PALETTE["primary"]},
                'bgcolor': "rgba(255,255,255,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 3.3], 'color': 'rgba(255,59,48,0.08)'},
                    {'range': [3.3, 6.6], 'color': 'rgba(255,204,0,0.08)'},
                    {'range': [6.6, 10], 'color': 'rgba(52,199,89,0.08)'}
                ]
            }
        )
    )
    fig.update_layout(template="anthill", margin=dict(l=0, r=0, t=28, b=0), height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)

def create_spider_chart(data, title, key=None):
    if not isinstance(data, dict) or not data:
        data = {"Market": 5, "Execution": 5, "Technology": 5, "Regulatory": 5, "Competition": 5}
    fig = go.Figure(
        go.Scatterpolar(
            r=list(data.values()), theta=list(data.keys()), fill='toself',
            line=dict(color=PALETTE["primary"], width=2), fillcolor='rgba(10,132,255,0.15)'
        )
    )
    fig.update_layout(template="anthill", title=dict(text=title), polar=dict(bgcolor='rgba(255,255,255,0)', radialaxis=dict(visible=True, range=[0, 10])), height=260)
    st.plotly_chart(fig, use_container_width=True, key=key)

def create_area_traffic_chart(traffic: List[int], key=None):
    if not traffic:
        return
    df = pd.DataFrame({"Month": list(range(1, len(traffic)+1)), "Traffic": traffic})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Traffic"], fill='tozeroy', line=dict(color=PALETTE["primary"], width=2), name="Visits"))
    fig.update_layout(template="anthill", title=dict(text="Web Traffic (last 12 months)"), height=240, xaxis_title="Month", yaxis_title="Visits")
    st.plotly_chart(fig, use_container_width=True, key=key)

def create_rule40_waterfall(annual_growth_pct: float, gross_margin_pct: float, key=None):
    total = float(annual_growth_pct) + float(gross_margin_pct)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Growth %", "Gross Margin %", "Rule of 40"],
        y=[annual_growth_pct, gross_margin_pct, 0],
        text=[f"{annual_growth_pct:.0f}%", f"{gross_margin_pct:.0f}%", f"{total:.0f}%"],
        textposition="outside",
        connector={"line": {"color": PALETTE["muted"]}},
        decreasing={"marker":{"color": PALETTE["danger"]}},
        increasing={"marker":{"color": PALETTE["success"]}},
        totals={"marker":{"color": PALETTE["accent"]}}
    ))
    fig.update_layout(template="anthill", title=dict(text="Rule of 40 Components"), height=240, yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True, key=key)

def create_risk_bar_chart(risks: Dict[str, float], key=None):
    if not risks:
        return
    items = sorted(risks.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=["Risk", "Score"])
    fig = go.Figure(go.Bar(x=df["Score"], y=df["Risk"], orientation="h", marker_color=PALETTE["warn"]))
    fig.update_layout(template="anthill", title=dict(text="Top Risk Drivers"), height=240, xaxis_title="Score (0–10)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True, key=key)

# FIX: draw label inside the bullet figure’s white “pill”
def create_bullet_indicator(title: str, current: float, target: float, suffix: str = "", invert: bool = False, key=None):
    # invert=False => higher is better; invert=True => lower is better
    perf = float(current or 0.0)
    tgt = float(target or 0.0)
    axis_max = max(1.0, perf, tgt) * 1.25

    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=perf,
        number={"suffix": suffix, "font": {"size": 16}},
        delta={"reference": tgt,
               "increasing": {"color": PALETTE["success"] if invert else PALETTE["danger"]},
               "decreasing": {"color": PALETTE["danger"] if invert else PALETTE["success"]}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, axis_max]},
            "bar": {"color": PALETTE["primary"]},
            "threshold": {"line": {"color": PALETTE["accent"], "width": 2}, "thickness": 0.75, "value": tgt}
        },
    ))

    # Label placed inside top whitespace (the “pill”)
    fig.add_annotation(
        x=0.0, y=1.06, xref="paper", yref="paper",
        text=title, showarrow=False, align="left",
        font=dict(size=12, color="#2f334d")
    )

    fig.update_layout(template="anthill", height=120, margin=dict(l=6, r=6, t=30, b=6))
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})

def fmt_usd_m(amount_abs_usd: float) -> str:
    try:
        return f"${amount_abs_usd/1_000_000:,.2f}M"
    except Exception:
        return "N/A"

def annualize_growth(monthly_pct: float) -> float:
    g = monthly_pct / 100.0
    return (pow(1.0 + g, 12) - 1.0) * 100.0

# =========================
# Research helpers
# =========================
def _render_paragraphs(title: str, content: str):
    if not content: return
    if title: st.markdown(f"**{title}**")
    parts = [p.strip() for p in content.replace("\r\n", "\n").split("\n\n") if p.strip()]
    for p in parts:
        st.markdown(p)

def display_research(md: Dict[str, Any]):
    ext = md.get("external_research", {}) or {}

    section_heading("Research & Sources", "Live web, filings, and public signals")
    if "notice" in ext:
        st.info(ext["notice"]); return
    if "error" in ext:
        st.info("External research was unavailable. Please try again."); return

    if ext.get("summary"):
        with card(title="Summary", accent="accent"):
            _render_paragraphs("", ext["summary"])

    sections = ext.get("sections", {}) or {}
    if sections:
        colL, colR = st.columns(2, gap="medium")
        left = ["overview", "products", "business_model", "gtm", "unit_economics", "funding", "investors", "leadership"]
        right = ["hiring", "traction", "customers", "pricing", "competitors", "moat", "partnerships", "regulatory", "risks", "tech_stack", "roadmap"]
        with colL:
            for k in left:
                if sections.get(k):
                    with card(title=k.replace("_"," ").title()):
                        _render_paragraphs("", sections[k])
        with colR:
            for k in right:
                if sections.get(k):
                    with card(title=k.replace("_"," ").title()):
                        _render_paragraphs("", sections[k])

    sources = ext.get("sources", []) or []
    if sources:
        with card(title="Citations", accent="primary"):
            for i, s in enumerate(sources[:20], start=1):
                title = s.get("title", "Source"); url = s.get("url", "#")
                snippet = s.get("snippet", ""); conf = s.get("confidence", None)
                conf_str = f" · conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
                st.markdown(f"- [{i}] [{title}]({url}) — {snippet}{conf_str}")

# =========================
# PDF quick path (no comps)
# =========================
def render_pdf_quick_analysis():
    section_heading("Quick Analysis From PDF", "Upload a deck and generate the full evaluation")
    with card(title="Upload & Run", accent="primary"):
        uploaded = st.file_uploader("Upload PDF deck", type=["pdf"], key="pdf_uploader_firstpage")
        run_clicked = st.button("Run Analysis from PDF", use_container_width=True, key="pdf_run_btn")
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
    section_heading("Step 1 — Company Profile", "This unlocks the metrics step")
    with card(title="Company & Founder", accent="primary"):
        inputs: Dict[str, Any] = {}
        c1, c2 = st.columns(2, gap="medium")
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
        col_next, _ = st.columns([1, 5], gap="medium")
        if col_next.button("Next →", disabled=not is_valid, use_container_width=True, key="pi_next"):
            if is_valid:
                st.session_state.profile_inputs = inputs
                st.session_state.wizard_step = 1
                st.rerun()
        if not is_valid:
            st.caption("Fill all required fields (including founder LinkedIn) to continue.")
    return inputs

def render_ssq_deep_dive_inputs() -> Dict[str, Any]:
    section_heading("Speed Scaling Deep‑Dive", "Factors used in SSQ deep-dive")
    f: Dict[str, Any] = {}
    with card(title="Market & Moat", accent="accent"):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        f["maximum_market_size_usd"]   = c1.number_input("Max Market Size (USD)", value=500_000_000, min_value=0, key="ssq_max_market_size_usd")
        f["market_growth_rate_pct"]    = c2.number_input("Market Growth Rate (%)", value=25.0, min_value=-100.0, max_value=500.0, step=0.5, key="ssq_market_growth_rate_pct")
        f["economic_condition_index"]  = c3.slider("Economic Condition (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_economic_condition_index")
        f["readiness_index"]           = c4.slider("Readiness (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_readiness_index")
        c5, c6, c7, c8 = st.columns(4, gap="medium")
        f["originality_index"]         = c5.slider("Originality (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_originality_index")
        f["need_index"]                = c6.slider("Need (0–10)", 0.0, 10.0, 8.0, 0.1, key="ssq_need_index")
        f["testing_index"]             = c7.slider("Testing (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_testing_index")
        f["pmf_index"]                 = c8.slider("PMF (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_pmf_index")
        c9, c10, c11, c12 = st.columns(4, gap="medium")
        f["scalability_index"]         = c9.slider("Scalability (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_scalability_index")
        f["technology_duplicacy_index"]= c10.slider("Tech Duplicacy (0–10)", 0.0, 10.0, 4.0, 0.1, key="ssq_technology_duplicacy_index")
        f["execution_duplicacy_index"] = c11.slider("Execution Duplicacy (0–10)", 0.0, 10.0, 4.0, 0.1, key="ssq_execution_duplicacy_index")
        f["first_mover_advantage_index"]= c12.slider("First Mover Advantage (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_fma_index")

    with card(title="GTM, Revenue & Ops", accent="primary"):
        c13, c14, c15, c16 = st.columns(4, gap="medium")
        f["barriers_to_entry_index"]   = c13.slider("Barriers to Entry (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_barriers_to_entry_index")
        f["num_close_competitors"]     = c14.number_input("Close Competitors (count)", value=5, min_value=0, key="ssq_num_close_competitors")
        f["price_advantage_pct"]       = c15.number_input("Price Advantage (%)", value=10.0, min_value=-100.0, max_value=100.0, step=0.5, key="ssq_price_advantage_pct")
        f["channels_of_promotion"]     = c16.number_input("Channels of Promotion (count)", value=3, min_value=0, max_value=25, key="ssq_channels_of_promotion")

        g1, g2, g3, g4 = st.columns(4, gap="medium")
        f["mrr_inr"]                    = g1.number_input("MRR (₹)", value=6_000_000, min_value=0, key="ssq_mrr_inr")
        f["sales_growth_pct"]           = g2.number_input("Sales Growth (YoY, %)", value=80.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_sales_growth_pct")
        f["lead_to_close_ratio_pct"]    = g3.number_input("Lead→Close Ratio (%)", value=12.0, min_value=0.0, max_value=100.0, step=0.1, key="ssq_lead_to_close_ratio_pct")
        f["marketing_spend_inr"]        = g4.number_input("Marketing Spend / mo (₹)", value=2_000_000, min_value=0, key="ssq_marketing_spend_inr")

        g5, g6, g7 = st.columns(3, gap="medium")
        f["ltv_cac_ratio"]              = g5.slider("LTV:CAC", 0.1, 10.0, 3.5, 0.1, key="ssq_ltv_cac_ratio")
        f["customer_growth_pct"]        = g6.number_input("Customer Growth (YoY, %)", value=100.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_customer_growth_pct")
        f["repurchase_ratio_pct"]       = g7.number_input("Repurchase Ratio (%)", value=25.0, min_value=0.0, max_value=100.0, step=0.5, key="ssq_repurchase_ratio_pct")

    with card(title="Team & Finance", accent="success"):
        t1, t2, t3, t4 = st.columns(4, gap="medium")
        f["domain_experience_years"]    = t1.number_input("Domain Experience (yrs)", value=6, min_value=0, max_value=40, key="ssq_domain_experience_years")
        f["quality_of_experience_index"]= t2.slider("Quality of Experience (0–10)", 0.0, 10.0, 7.0, 0.1, key="ssq_quality_of_experience_index")
        f["team_size"]                  = t3.number_input("Team Size", value=50, min_value=1, key="ssq_team_size")
        f["avg_salary_inr"]             = t4.number_input("Avg Salary / yr (₹)", value=1_500_000, min_value=0, key="ssq_avg_salary_inr")

        t5, t6, t7, t8 = st.columns(4, gap="medium")
        f["equity_dilution_pct"]        = t5.number_input("Equity Dilution to Date (%)", value=22.0, min_value=0.0, max_value=100.0, step=0.5, key="ssq_equity_dilution_pct")
        f["runway_months"]              = t6.number_input("Runway (months)", value=12.0, min_value=0.0, max_value=120.0, step=0.5, key="ssq_runway_months")
        f["cac_inr"]                    = t7.number_input("CAC (₹)", value=8_000, min_value=0, key="ssq_cac_inr")
        f["variance_analysis_index"]    = t8.slider("Variance Analysis (0–10)", 0.0, 10.0, 6.0, 0.1, key="ssq_variance_analysis_index")

        f1, f2, f3, f4 = st.columns(4, gap="medium")
        f["mrr_growth_rate_pct"]        = f1.number_input("MRR Growth Rate (YoY, %)", value=90.0, min_value=-100.0, max_value=1000.0, step=0.5, key="ssq_mrr_growth_rate_pct")
        f["de_ratio"]                   = f2.number_input("D/E Ratio", value=0.2, min_value=0.0, max_value=10.0, step=0.05, key="ssq_de_ratio")
        f["gpm_pct"]                    = f3.number_input("Gross Profit Margin (%)", value=60.0, min_value=-100.0, max_value=100.0, step=0.5, key="ssq_gpm_pct")
        f["nr_inr"]                     = f4.number_input("Net Revenue (₹)", value=120_000_000, min_value=0, key="ssq_nr_inr")
        f5, f6, f7 = st.columns(3, gap="medium")
        f["net_income_ratio_pct"]       = f5.number_input("Net Income Ratio (%)", value=5.0, min_value=-200.0, max_value=200.0, step=0.5, key="ssq_net_income_ratio_pct")
        f["filed_patents"]              = f6.number_input("Filed Patents (count)", value=2, min_value=0, max_value=200, key="ssq_filed_patents")
        f["approved_patents"]           = f7.number_input("Approved Patents (count)", value=0, min_value=0, max_value=200, key="ssq_approved_patents")
    return f

def render_outcomes_section() -> Dict[str, Any]:
    section_heading("Desired Outcomes from Investment", "Targets feed the memo and bullets")
    with card(title="Targets", accent="primary"):
        outcomes: Dict[str, Any] = {}
        c1, c2, c3 = st.columns(3, gap="medium")
        outcomes["target_arr_usd"]         = c1.number_input("Target ARR (USD)", value=5_000_000, min_value=0, key="outcome_target_arr_usd")
        outcomes["target_burn_multiple"]   = c2.number_input("Target Burn Multiple", value=1.5, min_value=0.0, max_value=20.0, step=0.1, key="outcome_target_burn_multiple")
        outcomes["target_nrr_pct"]         = c3.number_input("Target NRR (%)", value=115.0, min_value=0.0, max_value=500.0, step=0.5, key="outcome_target_nrr_pct")
        c4, c5, c6 = st.columns(3, gap="medium")
        outcomes["target_gm_pct"]          = c4.number_input("Target Gross Margin (%)", value=70.0, min_value=-100.0, max_value=100.0, step=0.5, key="outcome_target_gm_pct")
        outcomes["target_runway_months"]   = c5.number_input("Target Runway (months)", value=18.0, min_value=0.0, max_value=120.0, step=0.5, key="outcome_target_runway_months")
        outcomes["milestones"]             = c6.text_input("Top 3 Milestones (comma-separated)", "Marquee logos, New geography, Platform launch", key="outcome_milestones")
        return outcomes

def render_metrics_step(profile: Dict[str, Any]) -> Dict[str, Any]:
    section_heading("Step 2 — Metrics & Financials", "Quantitative context")
    inputs: Dict[str, Any] = {}
    with card(title="Core Metrics", accent="primary"):
        c1, c2, c3 = st.columns(3, gap="medium")
        inputs['founded_year']          = c1.number_input("Founded Year", 2010, 2025, profile.get('founded_year', 2022), key="metrics_founded_year")
        inputs['total_funding_usd']     = c2.number_input("Total Funding (USD)", value=5_000_000, min_value=0, key="metrics_total_funding_usd")
        inputs['team_size']             = c3.number_input("Team Size", value=50, min_value=1, key="metrics_team_size")

        c4, c5, c6, c7 = st.columns(4, gap="medium")
        inputs['product_stage_score']   = c4.slider("Product Stage (0-10)", 0.0, 10.0, 8.0, key="metrics_product_stage_score")
        inputs['team_score']            = c5.slider("Execution (0-10)", 0.0, 10.0, 8.0, key="metrics_team_score")
        inputs['moat_score']            = c6.slider("Moat (0-10)", 0.0, 10.0, 7.0, key="metrics_moat_score")
        inputs['investor_quality_score']= c7.slider("Investor Quality (1-10)", 1.0, 10.0, 7.0, key="metrics_investor_quality_score")

        st.markdown("**AI Scoring Options (subjectives)**")
        a1, a2 = st.columns(2, gap="medium")
        inputs["ai_score_team_execution"]   = a1.checkbox("Use AI to score Team Execution", value=False, key="metrics_ai_score_team_execution")
        inputs["ai_score_investor_quality"] = a2.checkbox("Use AI to score Investor Quality", value=False, key="metrics_ai_score_investor_quality")
        e1, e2 = st.columns(2, gap="medium")
        inputs["team_ai_evidence"]          = e1.text_area("Team Evidence (links, bios, achievements)", "", height=70, key="metrics_team_ai_evidence")
        inputs["investor_ai_evidence"]      = e2.text_area("Investor Evidence (cap table, fund brands, track record)", "", height=70, key="metrics_investor_ai_evidence")

        c8, c9, c10 = st.columns(3, gap="medium")
        inputs['ltv_cac_ratio']         = c8.slider("LTV:CAC", 0.1, 10.0, 3.5, 0.1, key="metrics_ltv_cac_ratio")
        inputs['gross_margin_pct']      = c9.slider("Gross Margin (%)", 0.0, 95.0, 60.0, 1.0, key="metrics_gross_margin_pct")
        inputs['monthly_churn_pct']     = c10.slider("Monthly Churn (%)", 0.0, 20.0, 2.0, 0.1, key="metrics_monthly_churn_pct")

        c11, c12, c13 = st.columns(3, gap="medium")
        inputs['arr']                   = c11.number_input("Current ARR (₹)", value=80_000_000, min_value=0, key="metrics_arr")
        inputs['burn']                  = c12.number_input("Monthly Burn (₹)", value=10_000_000, min_value=0, key="metrics_burn")
        inputs['cash']                  = c13.number_input("Cash Reserves (₹)", value=90_000_000, min_value=0, key="metrics_cash")

        c14, c15, c16 = st.columns(3, gap="medium")
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

        inputs["use_ai_valuation_assist"] = st.checkbox("Use AI to assist valuation multiples (sector/stage peers)", value=True, key="metrics_use_ai_valuation_assist")

    inputs["ssq_deep_dive_factors"]   = render_ssq_deep_dive_inputs()
    inputs["use_ai_ssq_weights"]      = st.checkbox("Use AI to weigh SSQ factors", value=True, key="metrics_use_ai_ssq_weights")
    inputs["desired_outcomes"]        = render_outcomes_section()

    with card(title="Proceed", accent="primary"):
        col_back, col_run = st.columns([1, 2], gap="medium")
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
# Deep Dive helpers
# =========================
def _compute_contributions(deep: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per = (deep.get("per_factor_scores") or {})
    weights = (deep.get("weights") or {})
    rows: List[Dict[str, Any]] = []
    for k, score in per.items():
        w = float(weights.get(k, 0.0))
        rows.append({"Factor": k, "Score (0–10)": float(score), "Weight": w, "Weighted": float(score) * w})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.sort_values("Weighted", ascending=False).reset_index(drop=True)
    strengths = df.head(10).copy()
    watchouts = df.sort_values(["Score (0–10)", "Weighted"], ascending=[True, True]).head(10).copy()
    return strengths, watchouts

# =========================
# Executive Dashboard and Deep Dive
# =========================
def render_summary_dashboard(report):
    memo = report.get('investment_memo', {}) or {}
    risk = report.get('risk_matrix', {}) or {}
    verdict = report.get('final_verdict', {}) or {}
    ssq = report.get('ssq_report', {}) or {}

    conviction = memo.get('conviction', 'Low')
    rec_icon = {"High": "🚀", "Medium": "👀", "Low": "✋"}.get(conviction, "❓")

    section_heading("Executive Dashboard", "Top‑level KPIs and risk profile")
    row1 = st.columns([1.1, 0.9, 1.0], gap="medium")
    with row1[0]:
        with card(title=f"{rec_icon} Recommendation", subtitle=f"Conviction: {conviction}", accent="primary"):
            st.metric("Valuation", verdict.get("predicted_valuation_range_usd", "N/A"))
            p = verdict.get('success_probability_percent', 0.0); p = p/100.0 if p > 1 else p
            st.metric("Success Probability", f"{p:.1%}")
    with row1[1]:
        with card(title="SSQ Snapshot", accent="accent"):
            st.metric("Score", f"{ssq.get('ssq_score', 0.0)} / 10")
            st.progress((ssq.get('ssq_score', 0.0) or 0.0) / 10.0)
            st.caption(f"Momentum {ssq.get('momentum',0)} · Efficiency {ssq.get('efficiency',0)} · Scalability {ssq.get('scalability',0)}")
    with row1[2]:
        with card(title="Deal Score & Risks", accent="warn"):
            create_gauge_chart(
                ({"High":10,"Medium":6,"Low":3}.get(conviction,5)*0.4 + (ssq.get('ssq_score', 0) or 0)*0.6),
                "Overall Deal Score",
                key="gauge_overall_deal_score"
            )
            create_risk_bar_chart(risk, key="risk_bar_chart")

    inputs = st.session_state.get("inputs", {}) or {}
    fx = float(os.environ.get("FX_INR_PER_USD", 83) or 83)
    arr_inr = float(inputs.get("arr", 0) or 0.0)
    arr_usd = arr_inr / (fx if fx > 0 else 83.0)
    gm = float(inputs.get("gross_margin_pct", 60.0) or 60.0)
    growth_ann = annualize_growth(float(inputs.get("expected_monthly_growth_pct", 5.0) or 5.0))
    traffic = inputs.get("monthly_web_traffic", [])

    section_heading("Key Visuals", "Traffic, Rule of 40, and risk clarity")
    row2 = st.columns(3, gap="medium")
    with row2[0]:
        with card(title="Web Traffic", subtitle="Last 12 months", accent="primary"):
            create_area_traffic_chart(traffic, key="traffic_area")
    with row2[1]:
        with card(title="Rule of 40", subtitle="Growth + Margin", accent="success"):
            create_rule40_waterfall(growth_ann, gm, key="rule40_waterfall")
    with row2[2]:
        with card(title="Radar Risk Profile", accent="warn"):
            create_spider_chart(report.get('risk_matrix', {}), "Risk Profile (Radar)", key="radar_risk")

    desired = inputs.get("desired_outcomes", {}) or {}
    if desired:
        section_heading("Targets vs Current", "Bullet gauges with deltas")
        burn = float(inputs.get("burn", 0.0))
        burn_mult = (burn * 12.0) / (arr_inr + 1e-6) if arr_inr > 0 else 0.0
        churn_m = float(inputs.get("monthly_churn_pct", 2.0))
        annual_ret = pow(1.0 - max(0, min(50.0, churn_m))/100.0, 12) * 100.0
        colb1, colb2, colb3, colb4 = st.columns(4, gap="medium")
        # No external card titles here; label is inside the figure
        with colb1:
            with card():
                create_bullet_indicator("ARR (USD)", current=arr_usd, target=float(desired.get("target_arr_usd", arr_usd)), suffix="", key="bullet_arr")
        with colb2:
            with card():
                create_bullet_indicator("Burn Multiple", current=burn_mult, target=float(desired.get("target_burn_multiple", 1.5)), invert=True, key="bullet_burn")
        with colb3:
            with card():
                create_bullet_indicator("NRR proxy (from churn)", current=annual_ret, target=float(desired.get("target_nrr_pct", 110.0)), suffix="%", key="bullet_nrr")
        with colb4:
            with card():
                create_bullet_indicator("Gross Margin", current=gm, target=float(desired.get("target_gm_pct", gm)), suffix="%", key="bullet_gm")

def render_deep_dive_tab(report: Dict[str, Any]):
    deep = report.get("ssq_deep_dive", {}) or {}
    ssq = report.get("ssq_report", {}) or {}
    founder = report.get("founder_profile", {}) or {}
    ai_diag = (report.get("ai_diagnostics", {}) or {}).get("subjectives", {}) or {}
    desired = (st.session_state.get("inputs", {}) or {}).get("desired_outcomes", {}) or {}

    section_heading("Deep Dive", "Founder signals, Speed Scaling factors, and target outcomes")

    snap1, snap2, snap3, snap4 = st.columns(4, gap="medium")
    with snap1: st.metric("SSQ (Final)", ssq.get("ssq_score", "N/A"))
    with snap2: st.metric("SSQ Deep‑Dive", deep.get("ssq_deep_dive_score", "N/A"))
    with snap3: st.metric("Execution Signals (0–10)", founder.get("execution_signals", "N/A"))
    with snap4: st.metric("Network Strength (0–10)", founder.get("network_strength", "N/A"))

    with card(title="Founder Profile (LinkedIn + X)", accent="primary"):
        if founder:
            st.write(founder.get("summary", ""))
            cols = st.columns(4, gap="medium")
            cols[0].write(f"Experience: {founder.get('experience_years','N/A')} yrs")
            cols[1].write(f"Leadership roles: {', '.join(founder.get('leadership_roles', [])[:3]) or 'N/A'}")
            cols[2].write(f"Domain expertise: {', '.join(founder.get('domain_expertise', [])[:3]) or 'N/A'}")
            cols[3].write(f"Functional expertise: {', '.join(founder.get('functional_expertise', [])[:3]) or 'N/A'}")

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                exits = founder.get("exits", []) or []
                if exits:
                    st.markdown("**Exits**")
                    for e in exits[:6]:
                        st.write(f"- {e.get('company','')} — {e.get('type','').upper()} {e.get('year','')}")
                ach = founder.get("notable_achievements", []) or []
                if ach:
                    st.markdown("**Achievements**")
                    for a in ach[:6]: st.write(f"- {a}")
                fr = founder.get("fundraises_led", []) or []
                if fr:
                    st.markdown("**Fundraises Led**")
                    for r in fr[:6]:
                        amt = r.get('amount_usd')
                        amt_str = f"${amt:,}" if isinstance(amt, (int, float)) else "—"
                        st.write(f"- {r.get('company','')} {r.get('round','')} {amt_str}")
            with c2:
                edu = founder.get("education", []) or []
                if edu:
                    st.markdown("**Education**")
                    for e in edu[:6]:
                        st.write(f"- {e.get('school','')} — {e.get('degree','')} {e.get('year','')}")
                risks = founder.get("risk_flags", []) or []
                if risks:
                    st.markdown("**Founder Risk Flags**")
                    for r in risks[:8]: st.write(f"- {r}")
            srcs = founder.get("sources", []) or []
            if srcs:
                st.markdown("**Founder Sources**")
                for i, s in enumerate(srcs[:8], start=1):
                    st.markdown(f"- [{i}] [{s.get('title','Source')}]({s.get('url','#')}) — {s.get('snippet','')}")
        else:
            st.info("No founder profile analysis available. Provide a LinkedIn URL in the Company Profile step.")

    with card(title="Speed Scaling Deep‑Dive", accent="accent"):
        if deep and deep.get("per_factor_scores"):
            strengths, watchouts = _compute_contributions(deep)
            contrib_df = strengths.copy()
            contrib_df["Factor"] = contrib_df["Factor"].astype(str)
            fig = go.Figure(go.Bar(x=contrib_df["Weighted"], y=contrib_df["Factor"], orientation="h", marker_color=PALETTE["primary"]))
            fig.update_layout(template="anthill", title=dict(text="Top Drivers (weighted)"), height=300, yaxis_title="", xaxis_title="Weighted contribution")
            st.plotly_chart(fig, use_container_width=True, key="chart_top_drivers")

            two = st.columns(2, gap="medium")
            with two[0]:
                st.markdown("**Strengths (Top 10)**")
                for _, row in strengths.iterrows():
                    st.write(f"- {row['Factor']}: score {row['Score (0–10)']:.1f}, weight {row['Weight']:.3f}")
            with two[1]:
                st.markdown("**Watchouts (Bottom 10 by score)**")
                for _, row in watchouts.iterrows():
                    st.write(f"- {row['Factor']}: score {row['Score (0–10)']:.1f}, weight {row['Weight']:.3f}")

            with st.expander("Per‑factor scores and weights (full)"):
                st.dataframe(pd.DataFrame({
                    "Factor": list((deep.get("per_factor_scores") or {}).keys()),
                    "Score (0–10)": list((deep.get("per_factor_scores") or {}).values()),
                    "Weight": [ (deep.get("weights") or {}).get(k, 0.0) for k in (deep.get("per_factor_scores") or {}).keys() ]
                }).sort_values("Weight", ascending=False), use_container_width=True)
            st.caption(f"AI Adjustment applied to SSQ deep-dive: {deep.get('ai_adjustment', 0)}")
        else:
            st.info("Deep-dive factors were not provided. Fill in the 'Speed Scaling Deep‑Dive' inputs before running analysis.")

    if ai_diag and (ai_diag.get("rationale") or ai_diag.get("red_flags")):
        with card(title="AI Subjective Assessment", accent="success"):
            if ai_diag.get("rationale"):
                st.write(ai_diag["rationale"])
            if ai_diag.get("red_flags"):
                st.markdown("**Subjective Red Flags**")
                st.write(ai_diag["red_flags"])

    if desired:
        with card(title="Desired Outcomes (next 12–18 months)", accent="primary"):
            c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
            c1.metric("Target ARR (USD)", f"${desired.get('target_arr_usd', 0):,.0f}")
            c2.metric("Target Burn Multiple", desired.get("target_burn_multiple", "—"))
            c3.metric("Target NRR (%)", desired.get("target_nrr_pct", "—"))
            c4.metric("Target GM (%)", desired.get("target_gm_pct", "—"))
            c5.metric("Runway (months)", desired.get("target_runway_months", "—"))
            if desired.get("milestones"):
                st.caption(f"Milestones: {desired.get('milestones')}")

# =========================
# Memo utils and Analysis page
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

def render_analysis_area(report):
    prof = report.get('profile', None)
    company_name = getattr(prof, 'company_name', 'Company') if prof else 'Company'
    header_cols = st.columns([3, 1], gap="medium")
    header_cols[0].markdown(f"## Diagnostic Report: {company_name}")
    if header_cols[1].button("New Analysis", use_container_width=True, key="new_analysis_btn"):
        keys_to_clear = ['report', 'inputs', 'wizard_step', 'profile_inputs']
        st.session_state.view = 'input'
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.rerun()

    render_summary_dashboard(report)

    tabs = st.tabs(["Investment Memo", "Deep Dive", "Research & Sources", "Inputs", "Forecast", "ML", "Hallucination Audit"])
    memo = report.get('investment_memo', {}) or {}
    sim_res = report.get('simulation', {}) or {}
    md = report.get('market_deep_dive', {}) or {}

    with tabs[0]:
        section_heading("Investment Memo", "IC‑grade narrative and rationale")
        with card(title="Executive Summary", accent="primary"):
            st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)
        with st.expander("Full memo"):
            cols = st.columns(2, gap="medium")
            detail_fields = ["investment_thesis","market","product","traction","unit_economics","gtm","competition","team","risks","catalysts","round_dynamics","use_of_proceeds","valuation_rationale","kpis_next_12m","exit_paths"]
            for i, k in enumerate(detail_fields):
                if memo.get(k):
                    with cols[i % 2]:
                        with card(title=k.replace('_',' ').title()):
                            st.write(memo[k])
        st.download_button("Download Memo (Markdown)", data=memo_to_markdown(memo), file_name=f"{company_name}_Investment_Memo.md", mime="text/markdown", key="memo_dl_btn")

        with st.expander("Financial Runway (quick view)"):
            ts = sim_res.get('time_series_data', pd.DataFrame())
            if not ts.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts["Month"], y=ts["Cash Reserves (₹)"], name="Cash Reserves", line=dict(color=PALETTE["accent"])))
                fig.add_trace(go.Scatter(x=ts["Month"], y=ts["Monthly Revenue (₹)"], name="Monthly Revenue", line=dict(color=PALETTE["primary"])))
                fig.update_layout(template="anthill", height=260, xaxis_title="Month")
                st.plotly_chart(fig, use_container_width=True, key="runway_dual_axis")

    with tabs[1]:
        render_deep_dive_tab(report)

    with tabs[2]:
        display_research(md)

    with tabs[3]:
        with card(title="Raw Inputs", accent="accent"):
            st.json(st.session_state.get('inputs', {}))

    with tabs[4]:
        forecast = report.get('fundraise_forecast', {}) or {}
        with card(title="Fundraise Forecast", accent="primary"):
            col1, col2, col3 = st.columns(3, gap="medium")
            col1.metric("Round Likelihood (6m)", f"{forecast.get('round_likelihood_6m', 0.0):.1%}")
            col2.metric("Round Likelihood (12m)", f"{forecast.get('round_likelihood_12m', 0.0):.1%}")
            col3.metric("Time to Next Round", f"{forecast.get('expected_time_to_next_round_months', 0.0):.1f} months")

    with tabs[5]:
        ml = report.get("ml_predictions", {}) or {}
        online = ml.get("online", {}) or {}
        with card(title="ML Predictions", accent="success"):
            col1, col2, col3 = st.columns(3, gap="medium")
            col1.metric("Online Likelihood (12m)", f"{online.get('round_probability_12m', 0.0):.1%}")
            col2.metric("Online Likelihood (6m)", f"{online.get('round_probability_6m', 0.0):.1%}")
            val = online.get("predicted_valuation_usd", None)
            col3.metric("Next Valuation (USD)", f"${val:,.0f}" if isinstance(val, (int, float)) and val else "N/A")

    with tabs[6]:
        section_heading("Hallucination Audit", "VC‑aware hallucination score with diagnostics")
        audit = report.get("hallucination_audit", {}) or {}
        if audit and not audit.get("error"):
            score = float(audit.get("vc_ahe_score", 0.0))
            with card(title="Overall VC AHE Score", accent="primary"):
                # scale 0..1 to 0..10 gauge for consistency with other gauges
                create_gauge_chart(score * 10.0, "0–10 scale (scaled from 0–1)", key="ahe_gauge_overall")
                c1, c2, c3, c4 = st.columns(4, gap="medium")
                c1.metric("Bias Risk", audit.get("bias_risk", "N/A"))
                c2.metric("Valuation Risk", audit.get("valuation_risk", "N/A"))
                c3.metric("Uncertainty", f"{audit.get('uncertainty', 0.0):.2f}")
                c4.metric("Trend Relevance", f"{audit.get('trend_relevance', 0.0):.2f}")
            with card(title="Components", accent="accent"):
                comp_rows = [{
                    "ROUGE-L": audit.get("rouge_l", 0.0),
                    "BERT F1 (or trigram F1)": audit.get("bert_f", 0.0),
                    "1 − Uncertainty": 1.0 - float(audit.get("uncertainty", 0.0)),
                    "Math/Science": audit.get("math_score", 0.0),
                    "Finance": audit.get("finance_score", 0.0),
                    "Trend": audit.get("trend_relevance", 0.0),
                }]
                st.dataframe(comp_rows, use_container_width=True)
        else:
            st.info("No hallucination audit available yet.")

# =========================
# Main
# =========================
async def main():
    engine = load_engine()

    if st.session_state.view == 'input':
        render_pdf_quick_analysis()
        step_cols = st.columns(2, gap="medium")
        step_cols[0].markdown("**1) Company Profile** ✅" if st.session_state.wizard_step == 1 else "**1) Company Profile**")
        step_cols[1].markdown("**2) Metrics & Financials**" + (" ✅" if st.session_state.wizard_step == 1 else ""))

        if st.session_state.wizard_step == 0:
            render_company_profile_step()
        else:
            render_metrics_step(st.session_state.get("profile_inputs", {}))

    elif st.session_state.view == 'analysis':
        with st.spinner("Running deep research, founder profiling, AI scoring, hallucination audit, and valuation..."):
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
                if st.button("Try Again", key="try_again_btn"):
                    st.session_state.view = 'input'
                    st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
