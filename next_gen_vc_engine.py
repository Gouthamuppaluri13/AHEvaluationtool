"""
VC engine with provider-agnostic deep research and IC-grade memo.
- Uses an external DeepResearchService (X + web) to build deep-dive and memo.
- Deterministic valuation synthesis (FX-aware) and probability from SSQ + risk.
- External services (predictor, comps, enrichment, forecast) are optional and guarded.
"""

import logging
import os
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List

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
# Deep research (no provider branding in UI)
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
    annual_revenue: float           # INR by convention from UI/PDF
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
        self.weights = self._get_contextual_weights()

    def _get_contextual_weights(self) -> Dict[str, float]:
        if self.profile.focus_area == FocusArea.ENHANCE_URBAN_LIFESTYLE:
            return {"momentum": 0.5, "efficiency": 0.3, "scalability": 0.2}
        elif self.profile.focus_area == FocusArea.LIVE_HEALTHY:
            return {"momentum": 0.3, "efficiency": 0.3, "scalability": 0.4}
        elif self.profile.focus_area == FocusArea.MITIGATE_CLIMATE_CHANGE:
            return {"momentum": 0.2, "efficiency": 0.3, "scalability": 0.5}
        return {"momentum": 0.33, "efficiency": 0.33, "scalability": 0.34}

    def _calculate_momentum_score(self) -> float:
        traffic = self.inputs.get("monthly_web_traffic", [0, 0])
        growth = (
            (traffic[-1] - traffic[0]) / (traffic[0] + 1e-6)
            if isinstance(traffic, (list, tuple)) and len(traffic) > 1 and isinstance(traffic[0], (int, float)) and traffic[0] > 0
            else 0.0
        )
        normalized_growth = min(max(growth, 0) / 5.0, 1.0) * 10.0
        return (normalized_growth * 0.6) + (float(self.profile.product_maturity_score) * 0.4)

    def _calculate_efficiency_score(self) -> float:
        ltv_cac = float(self.inputs.get("ltv_cac_ratio", 1.0))
        normalized_ltv_cac = min(max(ltv_cac, 0) / 5.0, 1.0) * 10.0
        burn = float(self.inputs.get("burn", 0))
        arr = float(self.inputs.get("arr", 0))
        burn_multiple = (burn * 12.0) / (arr + 1e-6) if arr > 0 else 99.0
        normalized_burn = max(0.0, 1.0 - (burn_multiple / 3.0)) * 10.0
        return (normalized_ltv_cac * 0.5) + (normalized_burn * 0.5)

    def _calculate_scalability_score(self) -> float:
        return (
            float(self.profile.adaptability_score) * 0.4
            + float(self.profile.competitive_advantage_score) * 0.4
            + float(self.profile.investor_quality_score) * 0.2
        )

    def calculate(self) -> Dict[str, float]:
        momentum = self._calculate_momentum_score()
        efficiency = self._calculate_efficiency_score()
        scalability = self._calculate_scalability_score()
        ssq = (
            momentum * self.weights["momentum"]
            + efficiency * self.weights["efficiency"]
            + scalability * self.weights["scalability"]
        )
        return {
            "ssq_score": round(ssq, 1),
            "momentum": round(momentum, 1),
            "efficiency": round(efficiency, 1),
            "scalability": round(scalability, 1),
        }


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


class FocusAreaStrategy:
    def process_inputs(self, i: Dict) -> Dict:
        raise NotImplementedError


class EnhanceUrbanLifestyleStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict:
        return {
            "product_maturity_score": float(i.get("product_stage_score", 5.0)),
            "competitive_advantage_score": float(i.get("moat_score", 5.0) * 0.4 + i.get("ltv_cac_ratio", 1.0) * 1.5),
            "adaptability_score": float(i.get("team_score", 5.0)),
        }


class LiveHealthyStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict:
        ce_map = {"None": 1, "Pre-clinical": 3, "Phase I/II": 6, "Phase III/Approved": 9}
        c = ce_map.get(str(i.get("clinical_evidence", "None")), 1)
        return {
            "product_maturity_score": float(i.get("product_stage_score", 5.0) * 0.3 + c * 0.7),
            "competitive_advantage_score": float(i.get("moat_score", 5.0)),
            "adaptability_score": float(i.get("team_score", 5.0)),
            "specialized_metrics": {"ClinicalEvidenceScore": float(c)},
        }


class MitigateClimateChangeStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict:
        return {
            "product_maturity_score": float(i.get("product_stage_score", 5.0) * 0.5 + i.get("trl", 1) * 0.5),
            "competitive_advantage_score": float(i.get("moat_score", 5.0)),
            "adaptability_score": float(i.get("team_score", 5.0)),
        }


class NextGenVCEngine:
    def __init__(self, tavily_key: Optional[str], av_key: Optional[str], _unused: Optional[str], research_key: Optional[str] = None):
        # Optional predictor
        self.online_predictor = None
        if ModelRegistry is not None and OnlinePredictor is not None:
            try:
                registry = ModelRegistry()
                joblib_path = registry.load_joblib_path()
                self.online_predictor = OnlinePredictor(joblib_path)
                logger.info("[engine] OnlinePredictor initialized.")
            except Exception as e:
                logger.warning(f"[engine] OnlinePredictor init failed: {e}")

        # Services
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

        # Deep research (provider-agnostic)
        self.research = None
        if DeepResearchService is not None:
            try:
                # Accept RESEARCH_API_KEY or GROK_API_KEY from env/secrets
                self.research = DeepResearchService(research_key or os.getenv("RESEARCH_API_KEY") or os.getenv("GROK_API_KEY"))
            except Exception as e:
                logger.warning(f"[engine] DeepResearchService disabled: {e}")

        # Focus strategies
        self.strategies: Dict[str, FocusAreaStrategy] = {
            FocusArea.LIVE_HEALTHY.value: LiveHealthyStrategy(),
            FocusArea.MITIGATE_CLIMATE_CHANGE.value: MitigateClimateChangeStrategy(),
            FocusArea.ENHANCE_URBAN_LIFESTYLE.value: EnhanceUrbanLifestyleStrategy(),
        }

    @staticmethod
    def _coerce_focus_area(value: Any) -> FocusArea:
        if isinstance(value, FocusArea):
            return value
        s = str(value or "").strip()
        for fa in FocusArea:
            if s.lower() in {fa.name.lower(), fa.value.lower()}:
                return fa
        return FocusArea.ENHANCE_URBAN_LIFESTYLE

    @staticmethod
    def _coerce_personality(value: Any) -> FounderPersonality:
        if isinstance(value, FounderPersonality):
            return value
        s = str(value or "").strip().lower()
        for fp in FounderPersonality:
            if s in {fp.name.lower(), fp.value.lower()}:
                return fp
        return FounderPersonality.EXECUTOR

    # ---------- Deterministic valuation + probability ----------
    def _fx_rate(self) -> float:
        try:
            return float(os.environ.get("FX_INR_PER_USD") or "") or 83.0
        except Exception:
            return 83.0

    def _stage_multiple_band(self, stage: str) -> tuple[float, float]:
        s = (stage or "").lower()
        if "pre-seed" in s:
            return (2.0, 6.0)
        if "seed" in s:
            return (3.0, 8.0)
        if "series a" in s:
            return (4.0, 10.0)
        if "series b" in s:
            return (3.5, 8.5)
        return (3.0, 8.0)

    def _prob_from_ssq_risk(self, ssq: float, risks: Dict[str, float]) -> float:
        ssq_norm = max(0.0, min(1.0, ssq / 10.0))
        avg_risk = (sum(risks.values()) / (len(risks) * 10.0)) if risks else 0.5
        base = 0.1 + 0.85 * ssq_norm - 0.25 * avg_risk
        return float(max(0.03, min(0.97, base)))

    def _deterministic_verdict(self, inputs: Dict[str, Any], profile: AdvancedStartupProfile, ssq: Dict[str, float], risks: Dict[str, float]) -> Dict[str, Any]:
        fx = self._fx_rate()
        arr_in = float(inputs.get("arr", 0.0))
        currency = str(inputs.get("currency", "INR")).upper()
        arr_usd = arr_in if currency == "USD" else (arr_in / (fx if fx > 0 else 83.0))

        low_mult, high_mult = self._stage_multiple_band(profile.stage)
        ssq_adj = max(0.75, min(1.25, 0.9 + 0.06 * (float(ssq.get("ssq_score", 5.0)) - 5.0)))
        g_adj = max(0.9, min(1.2, 1.0 + (float(inputs.get("expected_monthly_growth_pct", 5.0)) - 5.0) * 0.01))
        m_adj = max(0.85, min(1.15, 0.95 + (float(inputs.get("gross_margin_pct", 60.0)) - 60.0) * 0.003))
        low_val = max(0.5, arr_usd * low_mult * ssq_adj * g_adj * m_adj)
        high_val = max(low_val + 0.5, arr_usd * high_mult * ssq_adj * g_adj * m_adj)
        prob = self._prob_from_ssq_risk(float(ssq.get("ssq_score", 5.0)), risks) * 100.0
        return {
            "predicted_valuation_range_usd": f"${low_val:,.1f}M - ${high_val:,.1f}M",
            "success_probability_percent": round(prob, 1),
        }

    async def comprehensive_analysis(self, inputs: Dict[str, Any], comps_ticker: str) -> Dict[str, Any]:
        # Strategy + profile
        fa = self._coerce_focus_area(inputs.get("focus_area"))
        strategy = self.strategies.get(fa.value, EnhanceUrbanLifestyleStrategy())
        processed = strategy.process_inputs(inputs)

        profile = AdvancedStartupProfile(
            company_name=str(inputs.get("company_name", "Untitled Company")),
            stage=str(inputs.get("stage", "Series A")),
            sector=str(inputs.get("sector", "")),
            focus_area=fa,
            annual_revenue=float(inputs.get("arr", 0.0)),
            monthly_burn=float(inputs.get("burn", 0.0)),
            cash_reserves=float(inputs.get("cash", 0.0)),
            team_size=int(inputs.get("team_size", 0) or 0),
            founder_personality_type=self._coerce_personality(inputs.get("founder_type")),
            product_maturity_score=float(processed.get("product_maturity_score", 5.0)),
            competitive_advantage_score=float(processed.get("competitive_advantage_score", 5.0)),
            adaptability_score=float(processed.get("adaptability_score", 5.0)),
            investor_quality_score=float(inputs.get("investor_quality_score", 5.0)),
        )

        # SSQ
        ssq = SpeedscaleQuotientCalculator(inputs, profile).calculate()

        # Optional online predictor
        async def _run_online() -> Dict[str, Any]:
            if self.online_predictor is None:
                return {}
            try:
                return await asyncio.to_thread(self.online_predictor.predict, inputs)
            except Exception as e:
                logger.warning(f"[engine] Online predictor failed: {e}")
                return {}

        # Enrichment (optional) and Deep Research in parallel
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
                    str(inputs.get("sector", "")),
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

        # Market deep-dive: primarily from research; enrichment attached if present.
        market_analysis: Dict[str, Any] = {}
        if isinstance(research_res, dict):
            market_analysis["external_research"] = research_res
        if isinstance(enrichment, dict) and enrichment:
            market_analysis["indian_funding_trends"] = enrichment.get("indian_funding_trends", {})
            market_analysis["recent_news"] = enrichment.get("recent_news", {})
            market_analysis["india_funding_dataset_context"] = enrichment.get("india_funding_dataset_context", {})

        # Risk + verdict
        risk = {
            "Market": float(profile.market_risk_score),
            "Execution": float(10 - profile.adaptability_score),
            "Technology": float(profile.technology_risk_score),
            "Regulatory": float(profile.regulatory_risk_score),
            "Competition": float(10 - profile.competitive_advantage_score),
        }
        final_verdict = self._deterministic_verdict(inputs, profile, ssq, risk)

        # Simulation + comps + fundraising forecast (optional)
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
            return {
                "time_series_data": pd.DataFrame(hist),
                "narrative_summary": f"Based on current financials and 5% monthly revenue growth, the company {summary}",
            }

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

        # Build memo: use research_res["memo"] if present; else synthesize from sections + numeric context.
        research_memo = (market_analysis.get("external_research", {}) or {}).get("memo", {}) if market_analysis else {}
        memo = {}
        if isinstance(research_memo, dict) and research_memo:
            memo = {
                "executive_summary": research_memo.get("executive_summary", ""),
                "bull_case_narrative": research_memo.get("catalysts", "") or research_memo.get("investment_thesis", ""),
                "bear_case_narrative": research_memo.get("risks", ""),
                "recommendation": research_memo.get("recommendation", "Watchlist"),
                "conviction": research_memo.get("conviction", "Medium"),
                # include the full IC memo fields for downstream use
                **{k: v for k, v in research_memo.items() if k not in {"executive_summary"}}
            }
        else:
            sections = (market_analysis.get("external_research", {}) or {}).get("sections", {}) if market_analysis else {}
            ex_summary = (market_analysis.get("external_research", {}) or {}).get("summary", "")
            prob = final_verdict.get("success_probability_percent", 50.0)
            recommendation = "Invest" if prob >= 62.0 else ("Watchlist" if prob >= 45.0 else "Pass")
            conviction = "High" if prob >= 75.0 else ("Medium" if prob >= 55.0 else "Low")
            memo = {
                "executive_summary": ex_summary or "Deep research unavailable; see quantitative summary and risks.",
                "investment_thesis": sections.get("overview", ""),
                "market": sections.get("business_model", "") or "",
                "product": sections.get("products", ""),
                "traction": sections.get("traction", ""),
                "unit_economics": sections.get("unit_economics", ""),
                "gtm": sections.get("gtm", ""),
                "competition": sections.get("competitors", ""),
                "team": sections.get("leadership", ""),
                "risks": sections.get("risks", ""),
                "catalysts": sections.get("roadmap", "") or sections.get("partnerships", ""),
                "round_dynamics": sections.get("funding", ""),
                "use_of_proceeds": "",
                "valuation_rationale": "",
                "kpis_next_12m": "",
                "exit_paths": "",
                "bull_case_narrative": sections.get("moat", "") or sections.get("roadmap", ""),
                "bear_case_narrative": sections.get("risks", ""),
                "recommendation": recommendation,
                "conviction": conviction,
            }

        return {
            "final_verdict": final_verdict,
            "investment_memo": memo,
            "risk_matrix": risk,
            "simulation": sim_res,
            "market_deep_dive": market_analysis,
            "public_comps": comps_res,
            "profile": profile,
            "ssq_report": ssq,
            "fundraise_forecast": forecast,
            "ml_predictions": {"online": online_pred, "legacy": {}},
        }
