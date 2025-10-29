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
h4 { font-size: 1.1rem; color: var(--primary-text); margin-top: 1.5rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem;}
.main .block-container { padding: 1.5rem 2.5rem; }
.stButton > button { background: var(--accent-color); color: #ffffff; border: 1px solid var(--accent-color); border-radius: 4px; padding: 0.75rem 2rem; font-weight: 700; font-family: var(--font-mono); font-size: 1.1rem; transition: all 0.3s ease; box-shadow: 0 0 15px var(--accent-glow); }
.stButton > button:hover { box-shadow: 0 0 25px var(--accent-glow); transform: scale(1.02); border: 1px solid #fff; }
.stTextInput > div > div > input, .stTextArea textarea, .stNumberInput > div > div > input, .stSelectbox > div { background-color: var(--input-bg); color: var(--primary-text); border: 1px solid var(--border-color); border-radius: 4px; }
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
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = score, title = {'text': title, 'font': {'family': 'Roboto Mono', 'size': 16}}, gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "#E60023"}, 'bgcolor': "#161B22", 'steps' : [{'range': [0, 4], 'color': 'rgba(220, 53, 69, 0.5)'},{'range': [4, 7], 'color': 'rgba(255, 193, 7, 0.5)'},{'range': [7, 10], 'color': 'rgba(40, 167, 69, 0.5)'}]})); fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color':"#E6EDF3", 'family':"Roboto Mono"}, height=250, margin=dict(l=30, r=30, t=40, b=20)); return fig
def create_spider_chart(data, title):
    fig = go.Figure(go.Scatterpolar(r=list(data.values()), theta=list(data.keys()), fill='toself', line=dict(color='#E60023'), fillcolor='rgba(230, 0, 35, 0.2)')); fig.update_layout(title=dict(text=title, font=dict(family='Roboto Mono', size=16, color='#E60023')), polar=dict(bgcolor='rgba(22, 27, 34, 0.5)', radialaxis=dict(range=[0, 10], color='#7D8590'), angularaxis=dict(color='#E6EDF3')), showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Roboto Mono', color='#E6EDF3')); return fig

def display_deep_dive_intel(data: Dict[str, Any]):
    st.subheader("Market Deep-Dive")
    if not isinstance(data, dict) or "error" in data:
        st.error(data.get("error", "Market analysis data is not available."))
        return
    
    key_map = {
        "total_addressable_market": "🎯 Total Addressable Market",
        "competitive_landscape": "⚔️ Competitive Landscape",
        "moat_analysis": "🏰 Moat Analysis",
        "valuation_trends": "💲 Valuation & Funding Trends",
        "macro_factors": "🌍 Macro Factors"
    }
    
    for key, title in key_map.items():
        if key in data:
            st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
            content = data[key]
            if isinstance(content, dict):
                for sub_key, value in content.items():
                    if isinstance(value, list):
                        st.markdown(f"**{sub_key.replace('_', ' ').title()}:**")
                        for item in value:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {value}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        with st.container(border=True):
                            st.markdown(f"**{item.get('name', 'N/A')}** ({item.get('estimated_market_share', 'N/A')} Share)")
                            for sub_key, value in item.items():
                                if sub_key != 'name' and sub_key != 'estimated_market_share':
                                    st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {value}")
            else:
                st.markdown(content)
    
    # Add India-specific sections
    if "indian_funding_trends" in data and data["indian_funding_trends"]:
        st.markdown("---")
        st.markdown("<h4>🇮🇳 India Funding Trends (News-Derived)</h4>", unsafe_allow_html=True)
        trends = data["indian_funding_trends"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            median_inr = trends.get('median_round_size_inr', 0)
            st.metric("Median Round Size", f"₹{median_inr/10000000:.1f} Cr" if median_inr > 0 else "N/A")
        with col2:
            stage_counts = trends.get('round_counts_by_stage', {})
            total_rounds = sum(stage_counts.values())
            st.metric("Total Rounds (News)", total_rounds)
        with col3:
            top_invs = trends.get('top_investors', [])
            st.metric("Top Investors Found", len(top_invs))
        
        if stage_counts:
            st.markdown("**Round Distribution by Stage:**")
            for stage, count in sorted(stage_counts.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- {stage}: {count}")
        
        if top_invs:
            st.markdown("**Top Investors:**")
            st.markdown(", ".join(top_invs[:5]))
    
    if "recent_news" in data and data["recent_news"]:
        st.markdown("---")
        st.markdown("<h4>📰 Recent News</h4>", unsafe_allow_html=True)
        for news in data["recent_news"][:5]:
            with st.expander(news.get('title', 'Untitled')):
                st.markdown(news.get('content', 'No content'))
                if news.get('url'):
                    st.markdown(f"[Read more]({news['url']})")
    
    if "india_funding_dataset_context" in data and data["india_funding_dataset_context"]:
        st.markdown("---")
        st.markdown("<h4>📊 India Funding Dataset Context (Kaggle)</h4>", unsafe_allow_html=True)
        context = data["india_funding_dataset_context"]
        
        col1, col2 = st.columns(2)
        with col1:
            median_inr = context.get('median_amount_inr', 0)
            st.metric("Median Round Size (Dataset)", f"₹{median_inr/10000000:.1f} Cr" if median_inr > 0 else "N/A")
            st.metric("Total Rounds (Dataset)", context.get('rounds_total', 0))
        with col2:
            yearly = context.get('yearly_rounds', {})
            if yearly:
                st.markdown("**Yearly Distribution:**")
                for year, count in sorted(yearly.items(), key=lambda x: x[0], reverse=True)[:5]:
                    st.markdown(f"- {year}: {count} rounds")
        
        top_invs = context.get('top_investors', [])
        if top_invs:
            st.markdown("**Top Investors (Historical):**")
            st.markdown(", ".join(top_invs[:5]))
        
        round_mix = context.get('round_mix', {})
        if round_mix:
            st.markdown("**Round Type Mix:**")
            for round_type, count in sorted(round_mix.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.markdown(f"- {round_type}: {count}")

def render_input_area():
    col1, col2 = st.columns([1, 5]); col1.image("Anthill Logo-Falcon.png", width=120); col2.title("Anthill AI+ Evaluation"); st.markdown("---"); inputs = {}
    focus_area_map = {"Enhance Urban Lifestyle": ['E-commerce & D2C', 'Consumer Services', 'FinTech', 'PropTech', 'Logistics & Supply Chain', 'Travel & Hospitality', 'Media & Entertainment', 'Gaming'], "Live Healthy": ['HealthTech', 'AgriTech', 'CleanTech & EV', 'Consumer Services'], "Mitigate Climate Change": ['CleanTech & EV', 'AgriTech', 'Deep Tech & Robotics', 'GovTech & Social Enterprise']}
    tab1, tab2 = st.tabs(["📝 Company Profile", "📊 Metrics & Financials"])
    with tab1:
        st.subheader("Qualitative Information"); c1, c2 = st.columns(2, gap="large")
        with c1:
            inputs['company_name'] = st.text_input("Company Name", "InnoTech Bharat"); inputs['focus_area'] = st.selectbox("Investment Focus Area", list(focus_area_map.keys())); inputs['sector'] = st.selectbox("Primary Sector", focus_area_map[inputs['focus_area']])
        with c2:
            inputs['stage'] = st.selectbox("Funding Stage", ["Pre-Seed", "Seed", "Series A", "Series B"], index=2); inputs['founder_type'] = st.selectbox("Founder Archetype", [p.value for p in FounderPersonality]); inputs['location'] = st.radio("Headquarters Location", ['Metro', 'Tier-2/3'], horizontal=True)
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
            inputs['moat_score'] = st.slider("Moat Score", 0.0, 10.0, 7.0); inputs['is_dpiit_recognized'] = 1 if st.checkbox("DPIIT Recognized?", value=True) else 0
        with c3:
            inputs['investor_quality_score'] = st.slider("Investor Quality", 1.0, 10.0, 7.0); inputs['advisor_network_strength'] = st.slider("Advisor Network", 1.0, 10.0, 5.0)
            if inputs['focus_area'] == "Enhance Urban Lifestyle": inputs['ltv_cac_ratio'] = st.slider("LTV:CAC Ratio", 0.1, 10.0, 3.5, 0.1)
            elif inputs['focus_area'] == "Live Healthy": inputs['clinical_evidence'] = st.radio("Clinical Evidence", ["None", "Pre-clinical", "Phase I/II", "Phase III/Approved"])
            elif inputs['focus_area'] == "Mitigate Climate Change": inputs['trl'] = st.slider("Tech Readiness (TRL)", 1, 9, 6)
        st.markdown("---"); st.subheader("Financials & Traction"); c1, c2, c3 = st.columns(3)
        inputs['arr'] = c1.number_input("Current ARR (₹)", value=80000000, min_value=0); inputs['burn'] = c2.number_input("Monthly Burn (₹)", value=10000000, min_value=0); inputs['cash'] = c3.number_input("Cash on Hand (₹)", value=200000000, min_value=0)
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
    render_summary_dashboard(report)
    st.markdown("---")
    st.header("Deep Dive Analysis")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Investment Memo",
        "📈 Risk & Financial Simulation",
        "🌐 Market Deep-Dive & Comps",
        "💸 Fundraise Forecast",
        "🤖 ML Predictions",
        "📥 Submitted Inputs"
    ])
    
    memo = report.get('investment_memo', {})
    sim_res = report.get('simulation', {})
    
    with tab1:
        st.subheader("Executive Summary")
        st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)
        st.markdown("---")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Bull Case")
            st.markdown(memo.get('bull_case_narrative', 'No bull case generated.'), unsafe_allow_html=True)
        with c2:
            st.subheader("Bear Case")
            st.markdown(memo.get('bear_case_narrative', 'No bear case generated.'), unsafe_allow_html=True)
    
    with tab2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.plotly_chart(create_spider_chart(report.get('risk_matrix', {}), "Heuristic Risk Profile"), use_container_width=True)
        with c2:
            st.subheader("Financial Runway Simulation")
            st.info(sim_res.get('narrative_summary', 'Simulation not available.'))
            if not sim_res.get('time_series_data', pd.DataFrame()).empty:
                st.line_chart(sim_res['time_series_data'].set_index('Month'), color=["#E60023", "#7D8590"])
    
    with tab3:
        display_deep_dive_intel(report.get('market_deep_dive', {}))
        st.markdown("---")
        st.subheader("Public Comparables")
        comps = report.get('public_comps', {})
        if "Error" in comps:
            st.error(comps["Error"])
        elif comps:
            st.metric(label=f"Company: {comps.get('Company')}", value=comps.get('Price (₹)'), delta=f"Exchange: {comps.get('Market', 'N/A')}")
        else:
            st.warning("No public comparable data available.")
    
    with tab4:
        st.subheader("💸 Fundraise Forecast")
        forecast = report.get('fundraise_forecast')
        
        if forecast:
            col1, col2, col3 = st.columns(3)
            with col1:
                prob_6m = forecast.get('round_likelihood_6m', 0)
                st.metric("Round Probability (6 months)", f"{prob_6m:.1%}")
            with col2:
                prob_12m = forecast.get('round_likelihood_12m', 0)
                st.metric("Round Probability (12 months)", f"{prob_12m:.1%}")
            with col3:
                expected_time = forecast.get('expected_time_to_next_round_months', 0)
                st.metric("Expected Time to Next Round", f"{expected_time:.1f} months")
            
            st.markdown("---")
            st.info("These predictions are based on calibrated models trained on historical funding data and current company metrics.")
        else:
            st.warning("Fundraise forecast not available. This may indicate missing training data or model initialization issues.")
    
    with tab5:
        st.subheader("🤖 ML Predictions")
        
        online_pred = report.get('online_ml_prediction')
        legacy_pred = report.get('legacy_ml_prediction')
        
        if online_pred:
            st.markdown("### Online Model Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                prob_12m = online_pred.get('round_probability_12m', 0)
                st.metric("Round Probability (12m)", f"{prob_12m:.1%}")
            with col2:
                prob_6m = online_pred.get('round_probability_6m', 0)
                st.metric("Round Probability (6m)", f"{prob_6m:.1%}")
            with col3:
                valuation = online_pred.get('predicted_valuation_usd')
                if valuation:
                    st.metric("Predicted Valuation", f"${valuation:,.0f}")
                else:
                    st.metric("Predicted Valuation", "N/A")
            
            # Feature importances
            importances = online_pred.get('feature_importances', {})
            if importances:
                st.markdown("### Feature Importances (Top 20)")
                
                # Create horizontal bar chart
                import plotly.graph_objects as go
                
                features = list(importances.keys())[:20]
                values = [importances[f] for f in features]
                
                fig = go.Figure(go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker=dict(color='#E60023')
                ))
                
                fig.update_layout(
                    title="Most Important Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(22, 27, 34, 0.5)',
                    font=dict(family='Roboto Mono', color='#E6EDF3')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model metadata
            meta = online_pred.get('meta', {})
            st.markdown(f"**Model Type:** {meta.get('model_type', 'Unknown')}")
            st.markdown(f"**Source:** {meta.get('source', 'Unknown')}")
        
        if legacy_pred:
            st.markdown("---")
            st.markdown("### Legacy Model Results (For Comparison)")
            
            col1, col2 = st.columns(2)
            with col1:
                success_prob = legacy_pred.get('success_probability', 0)
                st.metric("Success Probability", f"{success_prob:.1%}")
            with col2:
                next_val = legacy_pred.get('predicted_next_valuation_usd', 0)
                st.metric("Next Valuation", f"${next_val:,.0f}")
        
        if not online_pred and not legacy_pred:
            st.warning("ML predictions not available.")
    
    with tab6:
        st.json(st.session_state.get('inputs', {}))

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