"""
VC engine with research-amplified IC memo and calibrated valuation/probability.
- External DeepResearchService (internally Grok-compatible) powers deep-dive + memo generation/refinement.
- Deterministic valuation: stage/sector bands + SSQ, growth, margin, retention, burn efficiency, FX-normalized ARR.
- Probability: logistic over SSQ, burn multiple, churn, Rule of 40, and stage.
"""

import logging
import os
import math
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

# Optional: Alpha Vantage
try:
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
    HAS_ALPHA = True
except Exception:
    TimeSeries = None  # type: ignore
    HAS_ALPHA = False

# Optional app-local services (guarded)
try:
    from services.model_registry import ModelRegistry  # type: ignore
except Exception:
    ModelRegistry = None  # type: ignore
try:
    from models.online_predictor import OnlinePredictor  # type: ignore
except Exception:
    OnlinePredictor = None  # type: ignore
try:
    from services.fundraise_forecast import FundraiseForecastService  # type: ignore
except Exception:
    FundraiseForecastService = None  # type: ignore
try:
    from services.data_enrichment import DataEnrichmentService  # type: ignore
except Exception:
    DataEnrichmentService = None  # type: ignore
# Deep research (provider-agnostic; UI is neutral)
try:
    from services.deep_research import DeepResearchService  # type: ignore
except Exception:
    DeepResearchService = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FocusArea(Enum):
    LIVE_HEALTHY = "Live Healthy"
    MITIGATE_CLIMATE_CHANGE = "Mitigate Climate Change"
    ENHANCE_URBAN_LIFESTYLE = "Enhance Urban Lifestyle"


class FounderPersonality(Enum):
    TECHNICAL = "technical"
    VISIONARY = "visionary"
    EXECUTOR = "executor"


@dataclass
class AdvancedStartupProfile:
    company_name: str
    stage: str
    sector: str
    focus_area: FocusArea
    annual_revenue: float           # ARR in INR by convention
    monthly_burn: float             # INR
    cash_reserves: float            # INR
    team_size: int
    founder_personality_type: FounderPersonality
    product_maturity_score: float
    competitive_advantage_score: float
    adaptability_score: float
    investor_quality_score: float
    market_risk_score: float = 5.0
    technology_risk_score: float = 5.0
    regulatory_risk_score: float = 5.0


class SpeedscaleQuotientCalculator:
    def __init__(self, inputs: Dict[str, Any], profile: AdvancedStartupProfile):
        self.inputs = inputs
        self.profile = profile
        self.weights = self._weights_for_focus()

    def _weights_for_focus(self) -> Dict[str, float]:
        if self.profile.focus_area == FocusArea.ENHANCE_URBAN_LIFESTYLE:
            return {"momentum": 0.5, "efficiency": 0.3, "scalability": 0.2}
        elif self.profile.focus_area == FocusArea.LIVE_HEALTHY:
            return {"momentum": 0.3, "efficiency": 0.3, "scalability": 0.4}
        elif self.profile.focus_area == FocusArea.MITIGATE_CLIMATE_CHANGE:
            return {"momentum": 0.2, "efficiency": 0.3, "scalability": 0.5}
        return {"momentum": 0.33, "efficiency": 0.33, "scalability": 0.34}

    def _momentum(self) -> float:
        traffic = self.inputs.get("monthly_web_traffic", [0, 0])
        growth = (
            (traffic[-1] - traffic[0]) / (traffic[0] + 1e-6)
            if isinstance(traffic, (list, tuple)) and len(traffic) > 1 and isinstance(traffic[0], (int, float)) and traffic[0] > 0
            else 0.0
        )
        normalized = min(max(growth, 0) / 5.0, 1.0) * 10.0
        return (normalized * 0.6) + (float(self.profile.product_maturity_score) * 0.4)

    def _efficiency(self) -> float:
        ltv_cac = float(self.inputs.get("ltv_cac_ratio", 1.0))
        normalized_ltv_cac = min(max(ltv_cac, 0) / 5.0, 1.0) * 10.0
        burn = float(self.inputs.get("burn", 0))
        arr = float(self.inputs.get("arr", 0))
        burn_multiple = (burn * 12.0) / (arr + 1e-6) if arr > 0 else 99.0
        normalized_burn = max(0.0, 1.0 - (burn_multiple / 3.0)) * 10.0
        return (normalized_ltv_cac * 0.5) + (normalized_burn * 0.5)

    def _scalability(self) -> float:
        return (
            float(self.profile.adaptability_score) * 0.4
            + float(self.profile.competitive_advantage_score) * 0.4
            + float(self.profile.investor_quality_score) * 0.2
        )

    def calculate(self) -> Dict[str, float]:
        m = self._momentum()
        e = self._efficiency()
        s = self._scalability()
        ssq = m * self.weights["momentum"] + e * self.weights["efficiency"] + s * self.weights["scalability"]
        return {"ssq_score": round(ssq, 1), "momentum": round(m, 1), "efficiency": round(e, 1), "scalability": round(s, 1)}


class ExternalDataIntegrator:
    def __init__(self, key: Optional[str]):
        self.enabled = bool(key) and HAS_ALPHA and TimeSeries is not None
        self.ts = TimeSeries(key=key, output_format="pandas") if self.enabled else None  # type: ignore
        if not self.enabled:
            logger.info("[engine] Public comps API disabled or key missing.")

    def get_public_comps(self, ticker: str) -> Dict[str, Any]:
        if not self.enabled or self.ts is None:
            return {"notice": "Public comps unavailable.", "ticker": ticker}
        try:
            data, meta = self.ts.get_quote_endpoint(symbol=ticker)  # type: ignore
            price = float(data["05. price"].iloc[-1])
            return {
                "Company": meta.get("2. Symbol", ticker),
                "Price (₹)": f"{price:,.2f}",
                "Market": meta.get("1. symbol", ticker).split(".")[-1],
            }
        except Exception as e:
            return {"Error": f"API call failed for {ticker}: {e}"}


class NextGenVCEngine:
    def __init__(self, tavily_key: Optional[str], av_key: Optional[str], _unused: Optional[str], research_key: Optional[str] = None):
        self.online_predictor = None
        if ModelRegistry is not None and OnlinePredictor is not None:
            try:
                reg = ModelRegistry()
                joblib_path = reg.load_joblib_path()
                self.online_predictor = OnlinePredictor(joblib_path)
            except Exception as e:
                logger.warning(f"[engine] OnlinePredictor init failed: {e}")

        self.data_integrator = ExternalDataIntegrator(av_key)

        self.enrichment = None
        if DataEnrichmentService is not None:
            try:
                self.enrichment = DataEnrichmentService(tavily_key)
            except Exception as e:
                logger.warning(f"[engine] Enrichment disabled: {e}")

        self.fundraise_forecast = None
        if FundraiseForecastService is not None:
            try:
                self.fundraise_forecast = FundraiseForecastService()
            except Exception as e:
                logger.warning(f"[engine] Fundraise forecast disabled: {e}")

        self.research = None
        if DeepResearchService is not None:
            try:
                self.research = DeepResearchService(research_key or os.getenv("RESEARCH_API_KEY") or os.getenv("GROK_API_KEY"))
            except Exception as e:
                logger.warning(f"[engine] DeepResearchService disabled: {e}")

    # ---------- Helpers ----------
    def _fx_rate(self) -> float:
        try:
            return float(os.environ.get("FX_INR_PER_USD") or "") or 83.0
        except Exception:
            return 83.0

    @staticmethod
    def _fmt_usd_m(amount_abs_usd: float) -> str:
        return f"${amount_abs_usd/1_000_000:,.2f}M"

    def _burn_multiple(self, arr_inr: float, burn_inr: float) -> float:
        return (burn_inr * 12.0) / (arr_inr + 1e-6) if arr_inr > 0 else 99.0

    def _annualized_growth(self, monthly_pct: float) -> float:
        g = monthly_pct / 100.0
        return (pow(1.0 + g, 12) - 1.0) * 100.0

    def _rule_of_40(self, annual_growth_pct: float, gross_margin_pct: float) -> float:
        return float(annual_growth_pct + gross_margin_pct)

    def _retention_factor(self, monthly_churn_pct: float) -> float:
        churn = max(0.0, min(50.0, monthly_churn_pct)) / 100.0
        annual_ret = pow(1.0 - churn, 12)
        return float(0.5 + annual_ret * 0.65)  # ~0.5..1.15

    def _efficiency_factor(self, burn_multiple: float) -> float:
        if burn_multiple <= 1.0:
            return 1.15
        if burn_multiple <= 1.5:
            return 1.08
        if burn_multiple <= 2.0:
            return 1.0
        if burn_multiple <= 3.0:
            return 0.9
        return 0.75

    def _growth_factor(self, annual_growth_pct: float) -> float:
        if annual_growth_pct <= 0:
            return 0.85
        if annual_growth_pct < 50:
            return 0.95
        if annual_growth_pct < 100:
            return 1.05
        if annual_growth_pct < 200:
            return 1.15
        return 1.25

    def _margin_factor(self, gross_margin_pct: float) -> float:
        if gross_margin_pct < 30:
            return 0.85
        if gross_margin_pct < 50:
            return 0.95
        if gross_margin_pct < 70:
            return 1.05
        if gross_margin_pct < 85:
            return 1.12
        return 1.20

    def _ssq_factor(self, ssq: float) -> float:
        return float(0.8 + (max(0.0, min(10.0, ssq)) / 10.0) * 0.4)

    def _sector_stage_bands(self, sector: str, stage: str) -> Tuple[float, float]:
        st = (stage or "").lower()
        s = (sector or "").lower()
        if "pre-seed" in st:
            base = (2.5, 7.0)
        elif "seed" in st:
            base = (3.5, 9.0)
        elif "series a" in st:
            base = (4.5, 11.0)
        elif "series b" in st:
            base = (4.0, 9.5)
        else:
            base = (3.5, 9.0)
        if "fintech" in s:
            base = (base[0] + 0.5, base[1] + 0.8)
        elif "bio" in s or "medtech" in s or "digital health" in s:
            base = (base[0] - 0.3, base[1] + 0.3)
        elif "clean" in s or "ev" in s or "climate" in s or "agri" in s:
            base = (base[0] - 0.2, base[1] + 0.4)
        elif "gaming" in s or "media" in s:
            base = (base[0] - 0.4, base[1] + 0.2)
        return base

    def _success_probability(self, ssq: float, burn_mult: float, churn_pct: float, rule40: float, stage: str) -> float:
        x_ssq = (ssq - 5.0) / 2.0
        x_burn = -(burn_mult - 2.0)
        x_churn = -(churn_pct - 3.0) / 3.0
        x_r40 = (rule40 - 40.0) / 20.0
        x_stage = {"pre-seed": -0.6, "seed": -0.3, "series a": 0.0, "series b": 0.2}.get(stage.lower(), -0.1)
        z = 0.1 + 0.65*x_ssq + 0.25*x_r40 + 0.20*x_stage + 0.18*x_burn + 0.12*x_churn
        p = 1.0 / (1.0 + math.exp(-z))
        return float(max(0.03, min(0.97, p)))

    def _valuation_range_abs_usd(self, inputs: Dict[str, Any], sector: str, stage: str, ssq: float) -> Tuple[float, float]:
        fx = self._fx_rate()
        arr_in = float(inputs.get("arr", 0.0))
        currency = str(inputs.get("currency", "INR")).upper()
        arr_usd_abs = arr_in if currency == "USD" else (arr_in / (fx if fx > 0 else 83.0))

        growth_ann = self._annualized_growth(float(inputs.get("expected_monthly_growth_pct", 5.0)))
        gm = float(inputs.get("gross_margin_pct", 60.0))
        churn = float(inputs.get("monthly_churn_pct", 2.0))
        burn_mult = self._burn_multiple(arr_in, float(inputs.get("burn", 0.0)))

        low_m, high_m = self._sector_stage_bands(sector, stage)
        mult = ((low_m + high_m) / 2.0) \
               * self._ssq_factor(ssq) \
               * self._growth_factor(growth_ann) \
               * self._margin_factor(gm) \
               * self._retention_factor(churn) \
               * self._efficiency_factor(burn_mult)

        mid = mult
        low = max(low_m, mid * 0.78)
        high = min(high_m * 1.15, mid * 1.25)

        low_abs = max(0.5e6, arr_usd_abs * low)
        high_abs = max(low_abs + 0.25e6, arr_usd_abs * high)
        return float(low_abs), float(high_abs)

    # ---------- Main ----------
    async def comprehensive_analysis(self, inputs: Dict[str, Any], comps_ticker: str) -> Dict[str, Any]:
        # Focus area
        fa = FocusArea.ENHANCE_URBAN_LIFESTYLE
        fa_val = str(inputs.get("focus_area", "")).strip()
        for x in FocusArea:
            if fa_val.lower() in {x.name.lower(), x.value.lower()}:
                fa = x
                break

        # Strategy features
        if fa == FocusArea.LIVE_HEALTHY:
            product_stage_score = float(inputs.get("product_stage_score", 5.0) * 0.3 + {"None":1,"Pre-clinical":3,"Phase I/II":6,"Phase III":8,"Approved":9}.get(str(inputs.get("regulatory_stage","None")),1)*0.7)
        elif fa == FocusArea.MITIGATE_CLIMATE_CHANGE:
            trl = float(inputs.get("trl_level", inputs.get("trl", 6)))
            product_stage_score = float(inputs.get("product_stage_score", 5.0) * 0.5 + (max(1.0,min(9.0,trl))/9.0)*10.0*0.5)
        else:
            product_stage_score = float(inputs.get("product_stage_score", 5.0))

        profile = AdvancedStartupProfile(
            company_name=str(inputs.get("company_name", "Untitled Company")),
            stage=str(inputs.get("stage", "Series A")),
            sector=str(inputs.get("sector", "")),
            focus_area=fa,
            annual_revenue=float(inputs.get("arr", 0.0)),
            monthly_burn=float(inputs.get("burn", 0.0)),
            cash_reserves=float(inputs.get("cash", 0.0)),
            team_size=int(inputs.get("team_size", 0) or 0),
            founder_personality_type=FounderPersonality.EXECUTOR if str(inputs.get("founder_type",'executor')).lower() not in {"technical","visionary","executor"} else FounderPersonality(str(inputs.get("founder_type")).lower()),
            product_maturity_score=product_stage_score,
            competitive_advantage_score=float(inputs.get("moat_score", 5.0)),
            adaptability_score=float(inputs.get("team_score", 5.0)),
            investor_quality_score=float(inputs.get("investor_quality_score", 5.0)),
        )

        ssq_report = SpeedscaleQuotientCalculator(inputs, profile).calculate()
        ssq = float(ssq_report.get("ssq_score", 5.0))

        # Optional tasks
        async def _run_online() -> Dict[str, Any]:
            if self.online_predictor is None:
                return {}
            try:
                return await asyncio.to_thread(self.online_predictor.predict, inputs)
            except Exception as e:
                logger.warning(f"[engine] Online predictor failed: {e}")
                return {}

        async def _run_enrichment() -> Dict[str, Any]:
            if self.enrichment is None:
                return {}
            try:
                return await asyncio.to_thread(self.enrichment.enrich, inputs.get("sector", ""), inputs.get("company_name", ""))
            except Exception as e:
                logger.warning(f"[engine] Enrichment failed: {e}")
                return {}

        async def _run_research() -> Dict[str, Any]:
            if self.research is None:
                return {"notice": "External research service not available. Add RESEARCH_API_KEY."}
            try:
                return await asyncio.to_thread(
                    self.research.research,
                    profile.company_name,
                    profile.sector,
                    str(inputs.get("location", "")),
                    str(inputs.get("product_desc", "")),
                )
            except Exception as e:
                logger.warning(f"[engine] Deep research failed: {e}")
                return {"error": "Research unavailable right now."}

        online_task = asyncio.create_task(_run_online())
        enrich_task = asyncio.create_task(_run_enrichment())
        research_task = asyncio.create_task(_run_research())

        online_pred, enrichment, research_res = await asyncio.gather(online_task, enrich_task, research_task)

        market_analysis: Dict[str, Any] = {}
        if isinstance(research_res, dict):
            market_analysis["external_research"] = research_res
        if isinstance(enrichment, dict) and enrichment:
            market_analysis["indian_funding_trends"] = enrichment.get("indian_funding_trends", {})
            market_analysis["recent_news"] = enrichment.get("recent_news", {})
            market_analysis["india_funding_dataset_context"] = enrichment.get("india_funding_dataset_context", {})

        # Derived KPIs
        burn_mult = self._burn_multiple(profile.annual_revenue, profile.monthly_burn)
        growth_ann = self._annualized_growth(float(inputs.get("expected_monthly_growth_pct", 5.0)))
        ro40 = self._rule_of_40(growth_ann, float(inputs.get("gross_margin_pct", 60.0)))
        churn_pct = float(inputs.get("monthly_churn_pct", 2.0))
        fx = self._fx_rate()
        arr_usd_abs = (profile.annual_revenue / (fx if fx > 0 else 83.0))

        # Valuation + probability
        low_abs, high_abs = self._valuation_range_abs_usd(inputs, profile.sector, profile.stage, ssq)
        valuation_range_str = f"{self._fmt_usd_m(low_abs)} - {self._fmt_usd_m(high_abs)}"
        prob = self._success_probability(ssq, burn_mult, churn_pct, ro40, profile.stage)

        # Simulation + comps + forecast
        def _simulate():
            cash = float(profile.cash_reserves)
            rev = float(profile.annual_revenue) / 12.0
            burn = float(profile.monthly_burn)
            hist: List[Dict[str, float]] = []
            for _ in range(36):
                rev *= 1.05
                cash += (rev - burn)
                hist.append({"Month": len(hist) + 1, "Cash Reserves (₹)": cash, "Monthly Revenue (₹)": rev})
                if cash <= 0:
                    break
            summary = "survives beyond the 36-month horizon." if hist and hist[-1]["Cash Reserves (₹)"] > 0 else f"runs out of cash in Month {len(hist)}."
            return {"time_series_data": pd.DataFrame(hist), "narrative_summary": f"Based on current financials and 5% monthly revenue growth, the company {summary}"}

        sim_task = asyncio.to_thread(_simulate)
        comps_task = asyncio.to_thread(self.data_integrator.get_public_comps, comps_ticker)

        async def _run_forecast() -> Dict[str, Any]:
            if self.fundraise_forecast is None:
                return {}
            try:
                return await asyncio.to_thread(self.fundraise_forecast.predict, inputs)
            except Exception as e:
                logger.warning(f"[engine] Fundraise forecast failed: {e}")
                return {}

        forecast_task = asyncio.create_task(_run_forecast())
        sim_res, comps_res, forecast = await asyncio.gather(sim_task, comps_task, forecast_task)

        risk = {
            "Market": float(profile.market_risk_score),
            "Execution": float(10 - profile.adaptability_score),
            "Technology": float(profile.technology_risk_score),
            "Regulatory": float(profile.regulatory_risk_score),
            "Competition": float(10 - profile.competitive_advantage_score),
        }

        final_verdict = {
            "predicted_valuation_range_usd": valuation_range_str,
            "success_probability_percent": round(prob * 100.0, 1),
        }

        # Research-amplified IC memo: use provider to refine if possible
        memo: Dict[str, Any] = {}
        ext = market_analysis.get("external_research", {}) if market_analysis else {}
        ext_memo = ext.get("memo", {}) if isinstance(ext, dict) else {}
        if isinstance(self.research, DeepResearchService):
            # Build a compact context for memo refinement
            memo_ctx = {
                "company": {
                    "name": profile.company_name,
                    "sector": profile.sector,
                    "stage": profile.stage,
                    "focus_area": profile.focus_area.value,
                },
                "valuation": {
                    "range_usd": valuation_range_str,
                    "low_abs_usd": round(low_abs, 2),
                    "high_abs_usd": round(high_abs, 2),
                },
                "kpis": {
                    "ssq": ssq,
                    "burn_multiple": round(burn_mult, 2),
                    "annual_growth_pct": round(growth_ann, 1),
                    "rule_of_40": round(ro40, 1),
                    "gross_margin_pct": float(inputs.get("gross_margin_pct", 60.0)),
                    "monthly_churn_pct": churn_pct,
                    "arr_usd": round(arr_usd_abs, 2),
                    "team_size": profile.team_size,
                },
                "inputs": {
                    "ltv_cac_ratio": float(inputs.get("ltv_cac_ratio", 3.5)),
                    "investor_quality_score": float(inputs.get("investor_quality_score", 7.0)),
                    "product_desc": str(inputs.get("product_desc", ""))[:800],
                    "founder_bio": str(inputs.get("founder_bio", ""))[:800],
                },
                "research": {
                    "summary": ext.get("summary", ""),
                    "sections": ext.get("sections", {}),
                    "sources": ext.get("sources", [])[:20],
                },
                "probability": final_verdict["success_probability_percent"],
            }
            try:
                refined = await asyncio.to_thread(self.research.build_ic_memo, memo_ctx)
                if isinstance(refined, dict) and refined.get("executive_summary"):
                    memo = {
                        "executive_summary": refined.get("executive_summary", ""),
                        "bull_case_narrative": refined.get("catalysts", "") or refined.get("investment_thesis", ""),
                        "bear_case_narrative": refined.get("risks", ""),
                        "recommendation": refined.get("recommendation", "Watchlist"),
                        "conviction": refined.get("conviction", "Medium"),
                        **{k: v for k, v in refined.items() if k not in {"executive_summary"}}
                    }
            except Exception as e:
                logger.warning(f"[engine] Memo refinement failed: {e}")

        # If still empty, fallback to external memo or a deterministic memo
        if not memo:
            if isinstance(ext_memo, dict) and ext_memo.get("executive_summary"):
                memo = {
                    "executive_summary": ext_memo.get("executive_summary", ""),
                    "bull_case_narrative": ext_memo.get("catalysts", "") or ext_memo.get("investment_thesis", ""),
                    "bear_case_narrative": ext_memo.get("risks", ""),
                    "recommendation": ext_memo.get("recommendation", "Watchlist"),
                    "conviction": ext_memo.get("conviction", "Medium"),
                    **{k: v for k, v in ext_memo.items() if k not in {"executive_summary"}}
                }
            else:
                memo = {
                    "executive_summary": (
                        f"{profile.company_name} — {profile.sector}, {profile.stage}. "
                        f"ARR ≈ {self._fmt_usd_m(arr_usd_abs)} | SSQ {ssq:.1f} | Rule of 40 {ro40:.0f} | "
                        f"Burn multiple {burn_mult:.2f}. Valuation: {valuation_range_str}."
                    ),
                    "investment_thesis": "Compelling wedge and secular tailwinds; path to premium multiples requires sustained growth and capital efficiency.",
                    "market": "Attractive demand vectors and room for share capture within adjacent categories.",
                    "product": "Differentiation via execution pace and defensible UX/infra; roadmap should harden moat.",
                    "traction": "Improving traffic and conversion; opportunity to formalize lighthouse logos.",
                    "unit_economics": f"LTV/CAC ≈ {float(inputs.get('ltv_cac_ratio', 3.5)):.1f}, churn ≈ {churn_pct:.1f}%/mo, gross margin ≈ {float(inputs.get('gross_margin_pct',60)):.0f}%.",
                    "gtm": "Mix of PLG and partner-led; expand channels with disciplined CAC thresholds.",
                    "competition": "Fragmented field; incumbents strong in distribution but slower in product velocity.",
                    "team": str(inputs.get("founder_bio", ""))[:800],
                    "risks": "Competitive intensity, GTM efficiency, macro fundraising window.",
                    "catalysts": "12–18m: feature launches, enterprise deals, geo expansion.",
                    "round_dynamics": f"Stage {profile.stage}; investor quality {float(inputs.get('investor_quality_score',7)):.1f}/10.",
                    "use_of_proceeds": "Strengthen GTM, product, data infra; selective leadership hires.",
                    "valuation_rationale": f"Sector/stage ARR multiples adjusted by quality factors yield {valuation_range_str}.",
                    "kpis_next_12m": "Rule of 40 > 50; burn multiple < 1.5; NRR > 110%; ≥2x ARR.",
                    "exit_paths": "Strategic M&A optionality; IPO feasible with scale and margins.",
                    "bull_case_narrative": "Premium multiples via sustained growth, improving margins, and durable retention.",
                    "bear_case_narrative": "Slower growth or capital intensity caps multiple; dilution risk.",
                    "recommendation": "Invest" if prob >= 0.68 else ("Watchlist" if prob >= 0.52 else "Pass"),
                    "conviction": "High" if prob >= 0.78 else ("Medium" if prob >= 0.58 else "Low"),
                }

        return {
            "final_verdict": final_verdict,
            "investment_memo": memo,
            "risk_matrix": risk,
            "simulation": sim_res,
            "market_deep_dive": market_analysis,
            "public_comps": comps_res,
            "profile": profile,
            "ssq_report": ssq_report,
            "fundraise_forecast": forecast,
            "ml_predictions": {"online": online_pred, "legacy": {}},
        }
