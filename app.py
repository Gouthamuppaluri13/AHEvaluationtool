import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
import os
import logging
from typing import Dict, Any

from next_gen_vc_engine import NextGenVCEngine, FocusArea, FounderPersonality, AdvancedStartupProfile
from ui_theme import apply_theme, hero, section_heading, card  # Glass theme

# Apply Apple-like glass theme
apply_theme(page_title="Anthill AI+ Evaluation", page_icon="🦅")

# Map Streamlit secrets into env for ModelRegistry
for k in ["HUGGINGFACE_MODEL_ID", "MODEL_ARTIFACT_NAME", "MODEL_ASSET_URL"]:
    try:
        val = st.secrets.get(k)
        if val:
            os.environ[k] = str(val)
    except Exception:
        pass

# Hero
hero(
    "Anthill AI+ Evaluation",
    "Minimal, elegant, glass‑themed VC copilot."
)

@st.cache_resource
def load_engine():
    try:
        engine = NextGenVCEngine(st.secrets["TAVILY_API_KEY"], st.secrets["ALPHA_VANTAGE_KEY"], st.secrets["GEMINI_API_KEY"])
        return engine
    except Exception as e:
        st.error(f"🔴 CRITICAL ERROR: Could not initialize engine. Check API keys and model files. Details: {e}")
        st.stop()

# Charts tuned for light glass
def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'family': 'ui-monospace, SFMono-Regular', 'size': 16, 'color': '#0F172A'}},
        number={'font': {'color': '#0F172A'}},
        gauge={
            'axis': {'range': [None, 10], 'tickcolor': '#94A3B8'},
            'bar': {'color': "#0A84FF"},
            'bgcolor': "rgba(255,255,255,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 3.3], 'color': 'rgba(255,59,48,0.10)'},
                {'range': [3.3, 6.6], 'color': 'rgba(255,204,0,0.10)'},
                {'range': [6.6, 10], 'color': 'rgba(52,199,89,0.10)'}
            ]
        }
    ))
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#0F172A")
    )
    return fig

def create_spider_chart(data, title):
    fig = go.Figure(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        line=dict(color='#0A84FF', width=2),
        fillcolor='rgba(10,132,255,0.15)'
    ))
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, font=dict(family='-apple-system, SF Pro Display, Inter', size=20, color='#0F172A')),
        polar=dict(
            bgcolor='rgba(255,255,255,0)',
            radialaxis=dict(visible=True, range=[0, 10], gridcolor='#E2E8F0', linecolor='#CBD5E1', tickfont=dict(color='#475569')),
            angularaxis=dict(gridcolor='#E2E8F0', linecolor='#CBD5E1', tickfont=dict(color='#475569'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def display_deep_dive_intel(data: Dict[str, Any]):
    section_heading("🌐 Market Deep-Dive")
    if not isinstance(data, dict) or "error" in data:
        st.error(data.get("error", "Market analysis data is not available."))
        return
    key_map = {
        "total_addressable_market": "🎯 Total Addressable Market",
        "competitive_landscape": "⚔️ Competitive Landscape",
        "moat_analysis": "🏰 Moat Analysis",
        "valuation_trends": "💹 Valuation Trends",
        "regulatory_outlook": "📜 Regulatory Outlook",
        "supply_demand_dynamics": "⚖️ Supply vs Demand"
    }
    for key, title in key_map.items():
        if key in data:
            st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
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
                # Card per competitive item
                for item in content:
                    if isinstance(item, dict):
                        with card():
                            st.markdown(f"**{item.get('name', 'N/A')}** ({item.get('estimated_market_share', 'N/A')} Share)")
                            for sub_key, value in item.items():
                                if sub_key not in ('name', 'estimated_market_share'):
                                    st.markdown(f"**{sub_key.replace('_', ' ').title()}:** {value}")
            else:
                st.markdown(str(content))

def render_input_area():
    with card():
        col1, col2 = st.columns([1, 5])
        col1.image("Anthill Logo-Falcon.png", width=120)
        col2.markdown("<h1>Anthill AI+ Evaluation</h1>", unsafe_allow_html=True)

    st.markdown("---")
    inputs = {}

    focus_area_map = {
        "Enhance Urban Lifestyle": ['E-commerce & D2C', 'Consumer Services', 'FinTech', 'PropTech', 'Logistics & Supply Chain', 'Travel & Hospitality', 'Media & Entertainment', 'Gaming'],
        "Live Healthy": ['Digital Health', 'MedTech', 'BioTech'],
        "Mitigate Climate Change": ['Clean Energy', 'EV Mobility', 'AgriTech']
    }

    tab1, tab2 = st.tabs(["📝 Company Profile", "📊 Metrics & Financials"])

    with tab1:
        section_heading("Qualitative Information", "Founders, product, and context")
        with card():
            c1, c2 = st.columns(2, gap="large")
            with c1:
                inputs['company_name'] = st.text_input("Company Name", "InnoTech Bharat")
                inputs['focus_area'] = st.selectbox("Investment Focus Area", list(focus_area_map.keys()))
                inputs['sector'] = st.selectbox("Sector", focus_area_map[inputs['focus_area']])
                inputs['location'] = st.selectbox("Location", ["India", "US", "SEA", "MENA", "EU"], index=0)
            with c2:
                inputs['stage'] = st.selectbox("Funding Stage", ["Pre-Seed", "Seed", "Series A", "Series B"], index=2)
                inputs['founder_type'] = st.selectbox("Founder Archetype", [p.value for p in FounderPersonality])
        with card():
            inputs['founder_bio'] = st.text_area(
                "Founder Bio",
                "An alumni of an IIT, the founder previously worked at a unicorn startup and has 8 years of experience in the FinTech space.",
                height=100
            )
            inputs['product_desc'] = st.text_area(
                "Product Description",
                "Our AI-powered platform provides a novel solution for the quick commerce sector in India.",
                height=100
            )

    with tab2:
        section_heading("Quantitative Metrics", "Score inputs and assumptions")
        with card():
            c1, c2, c3 = st.columns(3)
            with c1:
                inputs['founded_year'] = st.number_input("Founded Year", 2018, 2025, 2022)
                inputs['age'] = 2025 - inputs['founded_year']
                inputs['total_funding_usd'] = st.number_input("Total Funding (USD)", value=5_000_000, min_value=0)
                inputs['team_size'] = st.number_input("Team Size", value=50, min_value=1)
                inputs['num_investors'] = st.number_input("Number of Investors", value=5, min_value=0)
            with c2:
                inputs['product_stage_score'] = st.slider("Product Stage Score (0-10)", 0.0, 10.0, 8.0)
                inputs['team_score'] = st.slider("Team Score (Execution)", 0.0, 10.0, 8.0)
                inputs['moat_score'] = st.slider("Moat Score", 0.0, 10.0, 7.0)
                inputs['investor_quality_score'] = st.slider("Investor Quality", 1.0, 10.0, 7.0)
            with c3:
                inputs['ltv_cac_ratio'] = st.slider("LTV:CAC Ratio", 0.1, 10.0, 3.5, 0.1)
                inputs['gross_margin_pct'] = st.slider("Gross Margin (%)", 0.0, 95.0, 60.0, 1.0)
                inputs['monthly_churn_pct'] = st.slider("Monthly Revenue Churn (%)", 0.0, 20.0, 2.0, 0.1)

        st.markdown("---")
        section_heading("Financials & Traction", "ARR, burn, cash, and growth funnel")
        with card():
            c1, c2, c3 = st.columns(3)
            inputs['arr'] = c1.number_input("Current ARR (₹)", value=80_000_000, min_value=0)
            inputs['burn'] = c2.number_input("Monthly Burn (₹)", value=10_000_000, min_value=0)
            inputs['cash'] = c3.number_input("Cash Reserves (₹)", value=90_000_000, min_value=0)

            st.caption("— Growth and Funnel Assumptions —")
            c4, c5, c6 = st.columns(3)
            inputs['expected_monthly_growth_pct'] = c4.number_input("Expected Monthly Growth (%)", value=5.0, min_value=-50.0, max_value=200.0, step=0.5)
            inputs['growth_volatility_pct'] = c5.number_input("Growth Volatility (σ, %)", value=3.0, min_value=0.0, max_value=100.0, step=0.5)
            inputs['lead_to_customer_conv_pct'] = c6.number_input("Lead → Customer Conversion (%)", value=5.0, min_value=0.1, max_value=100.0, step=0.1)

            traffic_string = st.text_input(
                "Last 12 Months Web Traffic (comma-separated)",
                "5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000"
            )
            try:
                inputs['monthly_web_traffic'] = [int(x.strip()) for x in traffic_string.split(',') if x.strip()]
                if len(inputs['monthly_web_traffic']) != 12:
                    st.warning("Please ensure you enter exactly 12 comma-separated numbers for web traffic.", icon="⚠️")
            except ValueError:
                st.error("Invalid web traffic. Please enter only comma-separated numbers.", icon="🛑")
                st.stop()

    st.markdown("---")
    with card():
        c1, c2 = st.columns([4, 1.4])
        comps_ticker = c1.text_input("Public Comps Ticker", "ZOMATO.BSE")
        if c2.button("🚀 Run AI+ Analysis"):
            st.session_state.view = 'analysis'
            st.session_state.inputs = inputs
            st.session_state.comps_ticker = comps_ticker
            st.rerun()

def render_summary_dashboard(report):
    memo = report.get('investment_memo', {}) or {}
    profile = report.get('profile', None)
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
                f'<h2>{rec_icon} {memo.get("recommendation", "Error")}</h2>'
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
            highest_risk, highest_risk_score = max(risk.items(), key=lambda item: item[1]) if risk else ("N/A", 0)
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
    company_name = getattr(prof, 'company_name', 'Company')
    c1, c2 = st.columns([3, 1])
    c1.header(f"Diagnostic Report: {company_name}")
    if c2.button("⬅️ New Analysis"):
        keys_to_clear = ['report', 'inputs', 'comps_ticker']
        st.session_state.view = 'input'
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    render_summary_dashboard(report)
    st.markdown("---")
    section_heading("Deep Dive Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Investment Memo",
        "📈 Risk & Financial Simulation",
        "🌐 Market Deep-Dive & Comps",
        "📥 Submitted Inputs",
        "💸 Fundraise Forecast",
        "🤖 ML Predictions"
    ])

    memo = report.get('investment_memo', {}) or {}
    sim_res = report.get('simulation', {}) or {}

    with tab1:
        section_heading("Executive Summary")
        with card():
            st.markdown(memo.get('executive_summary', 'Not available.'), unsafe_allow_html=True)

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
            display_deep_dive_intel(report.get('market_deep_dive', {}))

        st.markdown("---")
        with card():
            st.subheader("Public Comparables")
            comps = report.get('public_comps', {})
            if "Error" in comps:
                st.error(comps["Error"])
            elif comps:
                st.metric(label=f"Company: {comps.get('Company')}", value=comps.get('Price (₹)'), delta=f"Exchange: {comps.get('Market', 'N/A')}")
            else:
                st.warning("No public comparable data available.")

        st.markdown("---")
        with card():
            st.subheader("Funding Trends (India) — Web News")
            trends = report.get('market_deep_dive', {}).get('indian_funding_trends', {})
            if trends:
                c1, c2, c3 = st.columns(3)
                c1.metric("Median Round (INR)", f"₹ {trends.get('median_round_size_inr', 0):,.0f}" if trends.get('median_round_size_inr') else "N/A")
                c2.metric("Total Rounds Parsed", sum(trends.get('round_counts_by_stage', {}).values()))
                c3.metric("Distinct Top Investors", len(trends.get('top_investors', [])))
                st.json(trends.get('round_counts_by_stage', {}))
            else:
                st.info("No India funding trend data.")

        st.markdown("---")
        with card():
            st.subheader("Recent News")
            news = report.get('market_deep_dive', {}).get('recent_news', {}).get('news', [])
            if news:
                for n in news[:8]:
                    st.markdown(f"- [{n.get('title','(untitled)')}]({n.get('url','#')}) — {n.get('published_date','')}")
            else:
                st.info("No recent news available.")

        st.markdown("---")
        with card():
            st.subheader("India Funding Dataset Context (Kaggle)")
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
        with card():
            st.json(st.session_state.get('inputs', {}))

    with tab5:
        with card():
            st.subheader("Fundraise Forecast")
            forecast = report.get('fundraise_forecast', {}) or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Round Likelihood (6 months)", f"{forecast.get('round_likelihood_6m', 0.0):.1%}")
            c2.metric("Round Likelihood (12 months)", f"{forecast.get('round_likelihood_12m', 0.0):.1%}")
            c3.metric("Expected Time to Next Round", f"{forecast.get('expected_time_to_next_round_months', 0.0):.1f} months")
            st.caption("Probabilities calibrated when model bundle is available; otherwise robust heuristics are applied.")

    with tab6:
        with card():
            st.subheader("Online Model Results")
            ml = report.get("ml_predictions", {}) or {}
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
                import plotly.express as px
                df_imp = pd.DataFrame(imps).sort_values("importance", ascending=True).tail(20)
                fig = px.bar(
                    df_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    template="simple_white",
                    color_discrete_sequence=["#0A84FF"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A"),
                    xaxis_title="Importance",
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
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
    if 'view' not in st.session_state:
        st.session_state.view = 'input'
    engine = load_engine()
    if st.session_state.view == 'input':
        render_input_area()
    elif st.session_state.view == 'analysis':
        with st.spinner("Calculating Speedscaling Quotient... Performing market deep-dive... Generating investment memo..."):
            try:
                report = await engine.comprehensive_analysis(st.session_state.inputs, st.session_state.comps_ticker)
                st.session_state.report = report
                render_analysis_area(report)
            except Exception as e:
                st.error(f"A critical error occurred during analysis: {e}")
                logging.error(f"Analysis failed: {e}", exc_info=True)
                if st.button("⬅️ Try Again"):
                    st.session_state.view = 'input'
                    st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
