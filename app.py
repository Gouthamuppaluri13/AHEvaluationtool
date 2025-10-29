import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
import os
import logging
from typing import Dict, Any

from next_gen_vc_engine import NextGenVCEngine, FocusArea, FounderPersonality, AdvancedStartupProfile

st.set_page_config(page_title="Anthill AI+ Evaluation", layout="wide", initial_sidebar_state="collapsed")

# Allow Streamlit Secrets to supply Hugging Face env for ModelRegistry
for k in ["HUGGINGFACE_MODEL_ID", "MODEL_ARTIFACT_NAME", "MODEL_ASSET_URL"]:
    try:
        val = st.secrets.get(k)
        if val:
            os.environ[k] = str(val)
    except Exception:
        pass

# --- UI Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg-color: #0D1117; --primary-text: #E6EDF3; --secondary-text: #7D8590;
    --accent-color: #E60023; --accent-glow: rgba(230, 0, 35, 0.4);
    --border-color: #30363D; --container-bg: #161B22; --input-bg: #0D1117;
    --font-primary: 'Inter', sans-serif; --font-mono: 'Roboto Mono', monospace;
    --green: #28a745; --amber: #ffc107; --red: #dc3545;
}
body, .stApp { background-color: var(--bg-color); color: var(--primary-text); font-family: var(--font-primary); }
.st-emotion-cache-16txtl3 { display: none; }
h1, h2, h3, h4 { font-family: var(--font-mono); font-weight: 700; color: var(--primary-text); }
h1 { font-size: 2.5rem; } 
h2 { font-size: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-bottom: 1rem;}
h3 { font-size: 1.25rem; color: var(--accent-color); }
h4 { font-size: 1.1rem; color: var(--primary-text); margin-top: 1.5rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border-color); padding-bottom: 0.25rem; }
.main .block-container { padding: 1.5rem 2.5rem; }
.stButton > button { background: var(--accent-color); color: #ffffff; border: 1px solid var(--accent-color); border-radius: 4px; padding: 0.75rem 2rem; font-weight: 700; font-family: var(--font-mono); }
.stButton > button:hover { box-shadow: 0 0 25px var(--accent-glow); transform: scale(1.02); border: 1px solid #fff; }
.stTextInput > div > div > input, .stTextArea textarea, .stNumberInput > div > div > input, .stSelectbox > div { background-color: var(--input-bg); color: var(--primary-text); border: 1px solid var(--border-color); }
.stMetric { background-color: var(--container-bg); border: 1px solid var(--border-color); border-left: 5px solid var(--accent-color); border-radius: 4px; padding: 1rem; }
.recommendation-card { padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-color); }
.recommendation-card h2 { font-size: 1.8rem; margin-bottom: 0.5rem; }
.recommendation-card p { font-size: 1.1rem; color: var(--secondary-text); font-family: var(--font-mono); }
.high-conviction { background-color: rgba(40, 167, 69, 0.1); border-left: 5px solid var(--green); }
.medium-conviction { background-color: rgba(255, 193, 7, 0.1); border-left: 5px solid var(--amber); }
.low-conviction { background-color: rgba(220, 53, 69, 0.1); border-left: 5px solid var(--red); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    try:
        engine = NextGenVCEngine(st.secrets["TAVILY_API_KEY"], st.secrets["ALPHA_VANTAGE_KEY"], st.secrets["GEMINI_API_KEY"]); return engine
    except Exception as e: st.error(f"🔴 CRITICAL ERROR: Could not initialize engine. Check API keys and model files. Details: {e}"); st.stop()

def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = score, title = {'text': title, 'font': {'family': 'Roboto Mono', 'size': 16}}, gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': '#E60023'}, 'bgcolor': '#161B22', 'bordercolor': '#30363D'})); return fig

def create_spider_chart(data, title):
    fig = go.Figure(go.Scatterpolar(r=list(data.values()), theta=list(data.keys()), fill='toself', line=dict(color='#E60023'), fillcolor='rgba(230, 0, 35, 0.2)')); fig.update_layout(title=dict(text=title, x=0.5), polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False, paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", font=dict(color="#E6EDF3")); return fig

def display_deep_dive_intel(data: Dict[str, Any]):
    st.subheader("Market Deep-Dive");
    if not isinstance(data, dict) or "error" in data: st.error(data.get("error", "Market analysis data is not available.")); return
    key_map = {"total_addressable_market": "🎯 Total Addressable Market", "competitive_landscape": "⚔️ Competitive Landscape", "moat_analysis": "🏰 Moat Analysis", "valuation_trends": "💹 Valuation Trends", "regulatory_outlook": "📜 Regulatory Outlook", "supply_demand_dynamics": "⚖️ Supply-Demand Dynamics"}
    for key, title in key_map.items():
        if key in data:
            st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True); content = data[key]
            if isinstance(content, dict):
                for sub_key, value in content.items():
                    if isinstance(value, list):
                        st.markdown(f"**{sub_key.replace('_', ' ').title()}:**");
                        for item in value: st.markdown(f"- {item}")
                    else: st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {value}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        with st.container(border=True):
                            st.markdown(f"**{item.get('name', 'N/A')}** ({item.get('estimated_market_share', 'N/A')} Share)")
                            for sub_key, value in item.items():
                                if sub_key != 'name' and sub_key != 'estimated_market_share': st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {value}")
            else: st.markdown(content)

def render_input_area():
    col1, col2 = st.columns([1, 5]); col1.image("Anthill Logo-Falcon.png", width=120); col2.title("Anthill AI+ Evaluation"); st.markdown("---"); inputs = {}
    focus_area_map = {"Enhance Urban Lifestyle": ['E-commerce & D2C', 'Consumer Services', 'FinTech', 'PropTech', 'Logistics & Supply Chain', 'Travel & Hospitality', 'Media & Entertainment', 'Gaming'], "Live Healthy": ['Digital Health', 'MedTech', 'BioTech', 'Wellness & Fitness'], "Mitigate Climate Change": ['Clean Energy', 'ClimateTech', 'AgriTech', 'Waste Management', 'WaterTech']}
    tab1, tab2 = st.tabs(["📝 Company Profile", "📊 Metrics & Financials"])
    with tab1:
        st.subheader("Qualitative Information"); c1, c2 = st.columns(2, gap="large")
        with c1:
            inputs['company_name'] = st.text_input("Company Name", "InnoTech Bharat"); inputs['focus_area'] = st.selectbox("Investment Focus Area", list(focus_area_map.keys())); inputs['sector'] = st.selectbox("Sector", focus_area_map[inputs['focus_area']]); inputs['hq_location'] = st.text_input("HQ Location", "Bengaluru, India")
        with c2:
            inputs['stage'] = st.selectbox("Funding Stage", ["Pre-Seed", "Seed", "Series A", "Series B"], index=2); inputs['founder_type'] = st.selectbox("Founder Archetype", [p.value for p in FounderPersonality]); inputs['is_dpiit_recognized'] = 1 if st.checkbox("DPIIT Recognized?", value=True) else 0
        inputs['founder_bio'] = st.text_area("Founder Bio", "An alumni of an IIT, the founder previously worked at a unicorn startup and has 8 years of experience in the FinTech space.", height=100)
        inputs['product_desc'] = st.text_area("Product Description", "Our AI-powered platform provides a novel solution for the quick commerce sector in India.", height=100)
    with tab2:
        st.subheader("Quantitative Metrics"); c1, c2, c3 = st.columns(3)
        with c1:
            inputs['founded_year'] = st.number_input("Founded Year", 2018, 2025, 2022); inputs['age'] = 2025 - inputs['founded_year']
            inputs['total_funding_usd'] = st.number_input("Total Funding (USD)", value=5000000, min_value=0)
            inputs['team_size'] = st.number_input("Team Size", value=50, min_value=1); inputs['num_investors'] = st.number_input("Number of Investors", value=5, min_value=0)
        with c2:
            inputs['product_stage_score'] = st.slider("Product Stage Score (0-10)", 0.0, 10.0, 8.0); inputs['team_score'] = st.slider("Team Score (Execution)", 0.0, 10.0, 8.0)
            inputs['moat_score'] = st.slider("Moat Score", 0.0, 10.0, 7.0)
            inputs['investor_quality_score'] = st.slider("Investor Quality", 1.0, 10.0, 7.0)
        with c3:
            inputs['ltv_cac_ratio'] = st.slider("LTV:CAC Ratio", 0.1, 10.0, 3.5, 0.1)
            inputs['gross_margin_pct'] = st.slider("Gross Margin (%)", 0.0, 95.0, 60.0, 1.0)
            inputs['monthly_churn_pct'] = st.slider("Monthly Revenue Churn (%)", 0.0, 20.0, 2.0, 0.1)
        st.markdown("---"); st.subheader("Financials & Traction"); c1, c2, c3 = st.columns(3)
        inputs['arr'] = c1.number_input("Current ARR (₹)", value=80000000, min_value=0); inputs['burn'] = c2.number_input("Monthly Burn (₹)", value=10000000, min_value=0); inputs['cash'] = c3.number_input("Cash (₹)", value=120000000, min_value=0)
        st.markdown("— Growth and Funnel Assumptions —")
        c4, c5, c6 = st.columns(3)
        inputs['expected_monthly_growth_pct'] = c4.number_input("Expected Monthly Growth (%)", value=5.0, min_value=-50.0, max_value=200.0, step=0.5)
        inputs['growth_volatility_pct'] = c5.number_input("Growth Volatility (σ, %)", value=3.0, min_value=0.0, max_value=100.0, step=0.5)
        inputs['lead_to_customer_conv_pct'] = c6.number_input("Lead → Customer Conversion (%)", value=5.0, min_value=0.1, max_value=100.0, step=0.1)

        traffic_string = st.text_input("Last 12 Months Web Traffic (comma-separated)", "5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000")
        try:
            inputs['monthly_web_traffic'] = [int(x.strip()) for x in traffic_string.split(',') if x.strip()]
            if len(inputs['monthly_web_traffic']) != 12: st.warning("Please ensure you enter exactly 12 comma-separated numbers for web traffic.", icon="⚠️")
        except ValueError: st.error("Invalid web traffic. Please enter only comma-separated numbers.", icon="🛑"); st.stop()
    st.markdown("---"); c1, c2 = st.columns([4, 1.4]); comps_ticker = c1.text_input("Public Comps Ticker", "ZOMATO.BSE"); 
    if c2.button("🚀 Run AI+ Analysis"):
        st.session_state.view = 'analysis'; st.session_state.inputs = inputs; st.session_state.comps_ticker = comps_ticker; st.rerun()

def render_summary_dashboard(report):
    memo = report.get('investment_memo', {}); profile = report.get('profile'); risk = report.get('risk_matrix', {}); verdict = report.get('final_verdict', {}); ssq = report.get('ssq_report', {})
    conviction = memo.get('conviction', 'Low'); rec_icon = {"High": "🚀", "Medium": "👀", "Low": "✋"}.get(conviction, "❓")
    st.header("Executive Dashboard"); col1, col2, col3 = st.columns([1.5, 1.2, 1.5])
    with col1:
        st.markdown(f'<div class="recommendation-card {conviction.lower()}-conviction"><h2>{rec_icon} {memo.get("recommendation", "Error")}</h2><p>Conviction: {conviction}</p></div>', unsafe_allow_html=True)
        st.metric("🎯 Predicted Valuation", verdict.get("predicted_valuation_range_usd", "N/A"))
        probability = verdict.get('success_probability_percent', 0.0)
        if probability > 1: probability /= 100.0
        st.metric("📈 Success Probability", f"{probability:.1%}")
    with col2:
        st.metric("⚡ Speedscaling Quotient (SSQ)", f"{ssq.get('ssq_score', 0.0)} / 10")
        with st.container(border=True):
            st.markdown(f"<small>Momentum: **{ssq.get('momentum', 0)}**</small>", unsafe_allow_html=True); st.progress(ssq.get('momentum', 0) / 10.0)
            st.markdown(f"<small>Efficiency: **{ssq.get('efficiency', 0)}**</small>", unsafe_allow_html=True); st.progress(ssq.get('efficiency', 0) / 10.0)
            st.markdown(f"<small>Scalability: **{ssq.get('scalability', 0)}**</small>", unsafe_allow_html=True); st.progress(ssq.get('scalability', 0) / 10.0)
    with col3:
        highest_risk, highest_risk_score = max(risk.items(), key=lambda item: item[1]) if risk else ("N/A", 0)
        rec_score = {"High": 10, "Medium": 6, "Low": 3}.get(conviction, 0); deal_score = (rec_score * 0.4) + (ssq.get('ssq_score', 0) * 0.6)
        st.plotly_chart(create_gauge_chart(deal_score, "Overall Deal Score"), use_container_width=True)
        st.metric("🛡️ Highest Risk Factor", f"{highest_risk} ({highest_risk_score:.1f}/10)", delta_color="inverse")

def render_analysis_area(report):
    col1, col2 = st.columns([1, 5]); col1.image("Anthill Logo-Falcon.png", width=120); col2.title("Anthill AI+ Evaluation"); st.markdown("---")
    c1, c2 = st.columns([3, 1]); c1.header(f"Diagnostic Report: {report.get('profile').company_name}")
    if c2.button("⬅️ New Analysis"):
        keys_to_clear = ['report', 'inputs', 'comps_ticker']; st.session_state.view = 'input'
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
    render_summary_dashboard(report); st.markdown("---"); st.header("Deep Dive Analysis")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Investment Memo", 
        "📈 Risk & Financial Simulation", 
        "🌐 Market Deep-Dive & Comps", 
        "📥 Submitted Inputs",
        "💸 Fundraise Forecast",
        "🤖 ML Predictions"
    ])
    memo = report.get('investment_memo', {}); sim_res = report.get('simulation', {})

    with tab1:
        st.subheader("Executive Summary"); st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)
        st.markdown("---"); c1, c2 = st.columns(2, gap="large")
        with c1: st.subheader("Bull Case"); st.markdown(memo.get('bull_case_narrative', 'No bull case generated.'), unsafe_allow_html=True)
        with c2: st.subheader("Bear Case"); st.markdown(memo.get('bear_case_narrative', 'No bear case generated.'), unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2, gap="large")
        with c1: st.plotly_chart(create_spider_chart(report.get('risk_matrix', {}), "Heuristic Risk Profile"), use_container_width=True)
        with c2:
            st.subheader("Financial Runway Simulation"); st.info(sim_res.get('narrative_summary', 'Simulation not available.'))
            if not sim_res.get('time_series_data', pd.DataFrame()).empty: st.line_chart(sim_res['time_series_data'].set_index('Month'), color=["#E60023", "#7D8590"])

    with tab3:
        display_deep_dive_intel(report.get('market_deep_dive', {})); st.markdown("---"); st.subheader("Public Comparables")
        comps = report.get('public_comps', {})
        if "Error" in comps: st.error(comps["Error"])
        elif comps: st.metric(label=f"Company: {comps.get('Company')}", value=comps.get('Price (₹)'), delta=f"Exchange: {comps.get('Market', 'N/A')}")
        else: st.warning("No public comparable data available.")

        st.markdown("---"); st.subheader("Funding Trends (India) — Web News")
        trends = report.get('market_deep_dive', {}).get('indian_funding_trends', {})
        if trends:
            c1, c2, c3 = st.columns(3)
            c1.metric("Median Round (INR)", f"₹ {trends.get('median_round_size_inr', 0):,.0f}" if trends.get('median_round_size_inr') else "N/A")
            c2.metric("Total Rounds Parsed", sum(trends.get('round_counts_by_stage', {}).values()))
            c3.metric("Distinct Top Investors", len(trends.get('top_investors', [])))
            st.json(trends.get('round_counts_by_stage', {}))
        else:
            st.info("No India funding trend data.")

        st.markdown("---"); st.subheader("Recent News")
        news = report.get('market_deep_dive', {}).get('recent_news', {}).get('news', [])
        if news:
            for n in news[:8]:
                st.markdown(f"- [{n.get('title','(untitled)')}]({n.get('url','#')}) — {n.get('published_date','')}")
        else:
            st.info("No recent news available.")

        st.markdown("---"); st.subheader("India Funding Dataset Context (Kaggle)")
        india_ctx = report.get('market_deep_dive', {}).get('india_funding_dataset_context', {})
        if india_ctx:
            c1, c2 = st.columns(2)
            c1.metric("Median Round (INR)", f"₹ {india_ctx.get('median_amount_inr', 0):,.0f}" if india_ctx.get('median_amount_inr') else "N/A")
            c2.metric("Total Rounds (Dataset)", f"{india_ctx.get('rounds_total', 0):,}")
            st.caption("Top Investors:")
            invs = india_ctx.get("top_investors", [])[:12]
            if invs:
                st.write(", ".join(i["name"] for i in invs))
            st.caption("Yearly Rounds:")
            st.json(india_ctx.get("yearly_rounds", [])[-10:])
        else:
            st.info("Kaggle-based India funding context not built yet. Run build_india_funding_context.py to generate data/india_funding_index.json.")

    with tab4:
        st.json(st.session_state.get('inputs', {}))

    with tab5:
        forecast = report.get('fundraise_forecast', {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Round Likelihood (6 months)", f"{forecast.get('round_likelihood_6m', 0.0):.1%}")
        c2.metric("Round Likelihood (12 months)", f"{forecast.get('round_likelihood_12m', 0.0):.1%}")
        c3.metric("Expected Time to Next Round", f"{forecast.get('expected_time_to_next_round_months', 0.0):.1f} months")
        st.caption("Probabilities calibrated when model bundle is available; otherwise robust heuristics are applied.")

    with tab6:
        st.subheader("Online Model Results")
        ml = report.get("ml_predictions", {})
        online = ml.get("online", {}) or {}
        legacy = ml.get("legacy", {}) or {}

        c1, c2, c3 = st.columns(3)
        c1.metric("Round Likelihood (12m, online)", f"{online.get('round_probability_12m', 0.0):.1%}")
        c2.metric("Round Likelihood (6m, online)", f"{online.get('round_probability_6m', 0.0):.1%}")
        val = online.get("predicted_valuation_usd", None)
        c3.metric("Predicted Next Valuation (USD, online)", f"${val:,.0f}" if isinstance(val, (int, float)) and val else "N/A")

        st.caption(f"Model source: {online.get('meta', {}).get('source', 'unknown')}; AUC: {online.get('meta', {}).get('auc', 'n/a')}")
        imps = online.get("feature_importances", [])
        if imps:
            df_imp = pd.DataFrame(imps).sort_values("importance", ascending=True).tail(20)
            import plotly.express as px
            fig = px.bar(df_imp, x="importance", y="feature", orientation="h", template="plotly_dark", color_discrete_sequence=["#E60023"])
            fig.update_layout(paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", font=dict(color="#E6EDF3"), xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available from the online model.")

        st.markdown("---")
        st.subheader("Legacy Hybrid Model (fallback)")
        c4, c5 = st.columns(2)
        c4.metric("Success Probability (legacy)", f"{legacy.get('success_probability', 0.0):.1%}")
        v2 = legacy.get("predicted_next_valuation_usd", None)
        c5.metric("Predicted Next Valuation (USD, legacy)", f"${v2:,.0f}" if isinstance(v2, (int, float)) and v2 else "N/A")

async def main():
    if 'view' not in st.session_state: st.session_state.view = 'input'
    engine = load_engine()
    if st.session_state.view == 'input': render_input_area()
    elif st.session_state.view == 'analysis':
        with st.spinner("Calculating Speedscaling Quotient... Performing market deep-dive... Generating investment memo..."):
            try:
                report = await engine.comprehensive_analysis(st.session_state.inputs, st.session_state.comps_ticker)
                st.session_state.report = report; render_analysis_area(report)
            except Exception as e:
                st.error(f"A critical error occurred during analysis: {e}"); logging.error(f"Analysis failed: {e}", exc_info=True)
                if st.button("⬅️ Try Again"): st.session_state.view = 'input'; st.rerun()

if __name__ == "__main__": asyncio.run(main())
