# --- replace the entire render_ssq_deep_dive_inputs() with this version ---
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


# --- replace the entire render_outcomes_section() with this version (adds keys for safety) ---
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


# --- replace the entire render_metrics_step() with this version (adds keys to avoid any collisions) ---
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
