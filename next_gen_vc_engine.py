"""
Next-generation VC engine with robust fallbacks:
- Guards optional deps: alpha_vantage, google-generativeai, torch, transformers
- Skips heavy BERT/PyTorch path unless fully available AND local files exist
- Uses OnlinePredictor via ModelRegistry first; legacy AIPlusPredictor is optional
- Gemini calls gracefully degrade to deterministic summaries when API/package missing
- External data (Alpha Vantage) guarded and non-fatal
"""

import logging
import os
import json
import re
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

# Optional deps: alpha_vantage
try:
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
    HAS_ALPHA = True
except Exception:
    TimeSeries = None  # type: ignore
    HAS_ALPHA = False

# Optional deps: Google Gemini
try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    genai = None  # type: ignore
    HAS_GEMINI = False

# Optional deps: torch + transformers (heavy)
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

try:
    from transformers import BertTokenizer, BertModel  # type: ignore
    HAS_TRANSFORMERS = True
except Exception:
    BertTokenizer = None  # type: ignore
    BertModel = None  # type: ignore
    HAS_TRANSFORMERS = False

# App-local services (guarded)
try:
    from services.data_enrichment import DataEnrichmentService  # type: ignore
except Exception:
    DataEnrichmentService = None  # type: ignore

try:
    from services.fundraise_forecast import FundraiseForecastService  # type: ignore
except Exception:
    FundraiseForecastService = None  # type: ignore

try:
    from services.model_registry import ModelRegistry  # type: ignore
except Exception:
    ModelRegistry = None  # type: ignore

try:
    from models.online_predictor import OnlinePredictor  # type: ignore
except Exception:
    OnlinePredictor = None  # type: ignore

# For legacy AIPlus preprocessor loading
try:
    import pickle  # noqa: F401
    HAS_PICKLE = True
except Exception:
    HAS_PICKLE = False

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
    annual_revenue: float
    monthly_burn: float
    cash_reserves: float
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
            if isinstance(traffic, (list, tuple))
            and len(traffic) > 1
            and isinstance(traffic[0], (int, float))
            and traffic[0] > 0
            else 0
        )
        normalized_growth = min(max(growth, 0) / 5.0, 1.0) * 10
        return (normalized_growth * 0.6) + (float(self.profile.product_maturity_score) * 0.4)

    def _calculate_efficiency_score(self) -> float:
        ltv_cac = float(self.inputs.get("ltv_cac_ratio", 1.0))
        normalized_ltv_cac = min(max(ltv_cac, 0) / 5.0, 1.0) * 10
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


# -------------------- Optional legacy AI+ model (Torch/BERT) --------------------

AI_PLUS_READY = HAS_TORCH and HAS_TRANSFORMERS

if AI_PLUS_READY:
    class AIPlusModel(nn.Module):  # type: ignore
        def __init__(self, num_numeric_features, tab_e=64, txt_e=64, ts_e=32):
            super().__init__()
            self.tabular_encoder = TabularEncoder(num_numeric_features, tab_e)
            self.text_encoder = TextEncoder(txt_e)
            self.ts_encoder = TimeSeriesEncoder(1, 64, 4, ts_e)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(tab_e + txt_e + ts_e, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            self.success_head = nn.Linear(256, 1)
            self.valuation_head = nn.Linear(256, 1)

        def forward(self, n, i, a, t):
            f = torch.cat(
                [self.tabular_encoder(n), self.text_encoder(i, a), self.ts_encoder(t)],
                dim=1,
            )
            fused = self.fusion_mlp(f)
            return torch.cat([self.success_head(fused), self.valuation_head(fused)], dim=1)

    class TabularEncoder(nn.Module):  # type: ignore
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(i, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, o),
            )

        def forward(self, x):
            return self.net(x)

    class TextEncoder(nn.Module):  # type: ignore
        def __init__(self, o):
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.fc = nn.Linear(self.bert.config.hidden_size, o)

        def forward(self, i, a):
            out = self.bert(input_ids=i, attention_mask=a).pooler_output
            return self.fc(out)

    class TimeSeriesEncoder(nn.Module):  # type: ignore
        def __init__(self, i, e, n, o):
            super().__init__()
            self.embedding = nn.Linear(i, e)
            el = nn.TransformerEncoderLayer(d_model=e, nhead=n, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(el, num_layers=2)
            self.fc = nn.Linear(e, o)

        def forward(self, x):
            h = self.embedding(x)
            h = self.transformer_encoder(h)
            return self.fc(h.mean(dim=1))

    class AIPlusPredictor:
        def __init__(self, model_path: str, preprocessor_path: str):
            # Only attempt if files exist
            if not (os.path.exists(model_path) and os.path.exists(preprocessor_path)):
                raise FileNotFoundError("AIPlus model/preprocessor not found.")
            with open(preprocessor_path, "rb") as f:
                pre = pickle.load(f)  # type: ignore
            self.preprocessor = pre
            try:
                num_features = len(self.preprocessor.get_feature_names_out())
            except Exception:
                # Fall back if preprocessor doesn't expose names
                num_features = getattr(self.preprocessor, "n_features_in_", 32)
            self.model = AIPlusModel(num_numeric_features=num_features)  # type: ignore
            self.model.load_state_dict(  # type: ignore
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            self.model.eval()  # type: ignore
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # type: ignore

        def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["funding_per_investor"] = df["total_funding_usd"] / (df["num_investors"] + 1e-6)
            df["funding_per_employee"] = df["total_funding_usd"] / (df["team_size"] + 1e-6)
            df["founder_has_iit_iim_exp"] = df["founder_bio"].fillna("").str.contains(
                "IIT|IIM", case=False, regex=True
            ).astype(int)
            df["avg_web_traffic"] = df["monthly_web_traffic"].apply(
                lambda x: float(np.mean(x)) if isinstance(x, (list, tuple)) and len(x) else 0.0
            )
            def g(x):
                if isinstance(x, (list, tuple)) and len(x) > 1 and isinstance(x[0], (int, float)) and x[0] > 0:
                    return (x[-1] - x[0]) / (x[0] + 1e-6)
                return 0.0
            df["web_traffic_growth"] = df["monthly_web_traffic"].apply(g)
            return df

        def predict(self, inputs: dict) -> Dict[str, float]:
            # Numeric/categorical path
            df = pd.DataFrame([inputs])
            df_f = self._feature_engineer(df)
            numeric_features = [
                "age", "total_funding_usd", "num_investors", "team_size",
                "is_dpiit_recognized", "funding_per_investor", "funding_per_employee",
                "founder_has_iit_iim_exp", "avg_web_traffic", "web_traffic_growth",
                "product_stage_score", "team_score", "moat_score", "ltv_cac_ratio",
            ]
            categorical_features = ["sector", "location"]
            cols = [c for c in numeric_features + categorical_features if c in df_f.columns]
            X = self.preprocessor.transform(df_f[cols])  # type: ignore

            # Text tokens
            founder_bio = str(df_f.get("founder_bio", [""])[0] or "")
            product_desc = str(df_f.get("product_desc", [""])[0] or "")
            text = founder_bio + " " + product_desc
            toks = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")  # type: ignore

            # Time series
            ts = df_f.get("monthly_web_traffic", [[0.0]])[0]
            if not isinstance(ts, (list, tuple)) or len(ts) == 0:
                ts = [0.0]
            t = torch.tensor(ts, dtype=torch.float32).unsqueeze(1).unsqueeze(0)  # type: ignore

            with torch.no_grad():  # type: ignore
                out = self.model(  # type: ignore
                    n=torch.tensor(X, dtype=torch.float32),  # type: ignore
                    i=toks["input_ids"],  # type: ignore
                    a=toks["attention_mask"],  # type: ignore
                    t=t,  # type: ignore
                )
            return {
                "success_probability": float(torch.sigmoid(out[:, 0]).item()),  # type: ignore
                "predicted_next_valuation_usd": float(out[:, 1].item()),  # type: ignore
            }
else:
    AIPlusPredictor = None  # type: ignore
    logger.info("[engine] AIPlusPredictor disabled (PyTorch/Transformers not available).")


# -------------------- External data and strategies --------------------

class ExternalDataIntegrator:
    def __init__(self, key: Optional[str]):
        self.key = key
        self.enabled = bool(key) and HAS_ALPHA and TimeSeries is not None
        self.ts = TimeSeries(key=key, output_format="pandas") if self.enabled else None  # type: ignore
        if not self.enabled:
            logger.info("[engine] Alpha Vantage disabled (missing package or key).")

    def get_public_comps(self, ticker: str) -> Dict[str, Any]:
        if not self.enabled or self.ts is None:
            return {"notice": "Alpha Vantage unavailable. Skipping comps.", "ticker": ticker}
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


# -------------------- Market Intel + Investment Memo (Gemini; guarded) --------------------

class MarketIntelligence:
    def __init__(self, gemini_key: Optional[str]):
        self.enabled = bool(gemini_key) and HAS_GEMINI and genai is not None
        if self.enabled:
            try:
                genai.configure(api_key=gemini_key)  # type: ignore
                self.model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore
            except Exception as e:
                logger.warning(f"[engine] Gemini init failed: {e}")
                self.enabled = False
                self.model = None
        else:
            self.model = None
            logger.info("[engine] Gemini disabled (missing package or key).")

    async def get_market_intel(self, sector: str, company: str, description: str) -> Dict[str, Any]:
        if not self.enabled or self.model is None:
            return {
                "notice": "Gemini unavailable; returning minimal market context.",
                "total_addressable_market": {"size": "N/A"},
                "competitive_landscape": [],
                "valuation_trends": {},
                "regulatory_outlook": {},
                "supply_demand_dynamics": {},
            }
        prompt = (
            'As a top-tier VC market analyst, perform a deep-dive analysis for a startup. '
            f'Startup: "{company}" in the "{sector}" sector. Description: "{description}". '
            "Task: Return JSON with: total_addressable_market, competitive_landscape, moat_analysis, "
            "valuation_trends, regulatory_outlook, supply_demand_dynamics."
        )
        try:
            r = await self.model.generate_content_async(prompt)  # type: ignore
            text = getattr(r, "text", "") or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {"notice": "No JSON found from LLM."}
        except Exception as e:
            logger.error(f"[engine] Market intelligence failed: {e}")
            return {"error": f"Failed to generate market deep-dive analysis. Details: {e}"}


class InvestmentThesisGenerator:
    def __init__(self, gemini_key: Optional[str]):
        self.enabled = bool(gemini_key) and HAS_GEMINI and genai is not None
        if self.enabled:
            try:
                genai.configure(api_key=gemini_key)  # type: ignore
                self.model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore
            except Exception as e:
                logger.warning(f"[engine] Gemini init for thesis failed: {e}")
                self.enabled = False
                self.model = None
        else:
            self.model = None
            logger.info("[engine] Gemini (memo) disabled (missing package or key).")

    async def generate(self, summary_text: str) -> Dict[str, Any]:
        if not self.enabled or self.model is None:
            msg = "AI memo disabled. Provide executive summary and recommendation manually."
            return {
                "executive_summary": msg,
                "bull_case_narrative": msg,
                "bear_case_narrative": msg,
                "recommendation": "Pending",
                "conviction": "Low",
            }
        prompt = (
            "As a VC Partner, write an investment memo as valid JSON with keys: "
            "executive_summary, bull_case_narrative, bear_case_narrative, "
            "recommendation ('Invest' or 'Pass'), conviction ('High'|'Medium'|'Low'). "
            f"Data summary: {summary_text}"
        )
        try:
            r = await self.model.generate_content_async(prompt)  # type: ignore
            text = getattr(r, "text", "") or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("No valid JSON object found in the LLM response for the investment memo.")
        except Exception as e:
            logger.error(f"[engine] Thesis generation failed: {e}")
            msg = "AI memo generation failed. Response could not be parsed."
            return {
                "executive_summary": msg,
                "bull_case_narrative": msg,
                "bear_case_narrative": msg,
                "recommendation": "Error",
                "conviction": "Error",
            }


# -------------------- Risk + Simulation --------------------

class ComprehensiveRiskMatrix:
    def assess(self, p: AdvancedStartupProfile) -> Dict[str, float]:
        return {
            "Market": float(p.market_risk_score),
            "Execution": float(10 - p.adaptability_score),
            "Technology": float(p.technology_risk_score),
            "Regulatory": float(p.regulatory_risk_score),
            "Competition": float(10 - p.competitive_advantage_score),
        }


class InteractiveSimulation:
    def run_simulation(self, p: AdvancedStartupProfile) -> Dict[str, Any]:
        cash = float(p.cash_reserves)
        rev = float(p.annual_revenue) / 12.0
        burn = float(p.monthly_burn)
        hist: List[Dict[str, float]] = []
        for _ in range(36):
            rev *= 1.05
            cash += (rev - burn)
            hist.append({"Month": len(hist) + 1, "Cash Reserves (₹)": cash, "Monthly Revenue (₹)": rev})
            if cash <= 0:
                break
        summary = (
            "survives beyond the 36-month horizon."
            if hist and hist[-1]["Cash Reserves (₹)"] > 0
            else f"runs out of cash in Month {len(hist)}."
        )
        return {
            "time_series_data": pd.DataFrame(hist),
            "narrative_summary": f"Based on current financials and a modest 5% monthly revenue growth, the company {summary}",
        }


# -------------------- Engine --------------------

class NextGenVCEngine:
    def __init__(self, tavily_key: Optional[str], av_key: Optional[str], gemini_key: Optional[str]):
        # Online predictor via registry (preferred)
        self.online_predictor = None
        if ModelRegistry is not None and OnlinePredictor is not None:
            try:
                registry = ModelRegistry()
                joblib_path = registry.load_joblib_path()
                self.online_predictor = OnlinePredictor(joblib_path)
                logger.info("[engine] OnlinePredictor initialized.")
            except Exception as e:
                logger.warning(f"[engine] OnlinePredictor init failed: {e}. Falling back.")

        # Legacy AIPlus predictor (optional; heavy)
        self.predictor = None
        if AIPlusPredictor is not None:
            try:
                # Only construct if files exist to avoid BERT download in environments that don't need it
                if os.path.exists("ai_plus_model.pth") and os.path.exists("preprocessor.pkl"):
                    self.predictor = AIPlusPredictor("ai_plus_model.pth", "preprocessor.pkl")  # type: ignore
                    logger.info("[engine] AIPlusPredictor initialized.")
                else:
                    logger.info("[engine] Skipping AIPlusPredictor (model files not found).")
            except Exception as e:
                logger.warning(f"[engine] AIPlusPredictor init skipped: {e}")

        # Services
        self.market_intel = MarketIntelligence(gemini_key)
        self.thesis_gen = InvestmentThesisGenerator(gemini_key)
        self.risk_matrix = ComprehensiveRiskMatrix()
        self.simulation = InteractiveSimulation()
        self.data_integrator = ExternalDataIntegrator(av_key)

        self.enrichment = None
        if DataEnrichmentService is not None:
            try:
                self.enrichment = DataEnrichmentService(tavily_key)
            except Exception as e:
                logger.warning(f"[engine] DataEnrichmentService disabled: {e}")

        self.fundraise_forecast = None
        if FundraiseForecastService is not None:
            try:
                self.fundraise_forecast = FundraiseForecastService()
            except Exception as e:
                logger.warning(f"[engine] FundraiseForecastService disabled: {e}")

        # Strategy map
        self.strategies: Dict[str, FocusAreaStrategy] = {
            FocusArea.LIVE_HEALTHY.value: LiveHealthyStrategy(),
            FocusArea.MITIGATE_CLIMATE_CHANGE.value: MitigateClimateChangeStrategy(),
            FocusArea.ENHANCE_URBAN_LIFESTYLE.value: EnhanceUrbanLifestyleStrategy(),
        }

        # Synthesis model (Gemini) used just for verdict; reuse the same capability
        self.synthesis_model = self.market_intel.model if self.market_intel.enabled else None

    @staticmethod
    def _coerce_focus_area(value: Any) -> FocusArea:
        # Accept either enum name or enum value (case-insensitive); default to ENHANCE_URBAN_LIFESTYLE
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

    @staticmethod
    def _c(o):
        if isinstance(o, dict):
            return {k: NextGenVCEngine._c(v) for k, v in o.items()}
        if isinstance(o, list):
            return [NextGenVCEngine._c(e) for e in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Enum):
            return o.value
        return o

    def _create_synthesis_prompt_text(self, data: Dict) -> str:
        profile = self._c(data.get("heuristic_profile", {}))
        verdict = self._c(data.get("final_verdict", {}))
        ssq = self._c(data.get("speedscale_quotient", {}))
        risks = self._c(data.get("risk_scores", {}))
        market = self._c(data.get("market", {}))

        india_ctx = market.get("india_funding_dataset_context", {})
        india_ctx_str = ""
        if isinstance(india_ctx, dict) and india_ctx:
            inv = ", ".join([i["name"] for i in india_ctx.get("top_investors", [])[:8] if "name" in i])
            yrs = [y.get("year") for y in india_ctx.get("yearly_rounds", []) if isinstance(y, dict)]
            india_ctx_str = (
                f"India Sector Context — Median Round (INR): {india_ctx.get('median_amount_inr','N/A')}; "
                f"Yearly Rounds: {yrs[-5:]}; Top Investors: {inv}"
            )

        summary = f"""- Company: {profile.get('company_name')} | Sector: {profile.get('sector')} | Stage: {profile.get('stage')}
### Final Verdict & SSQ
- Predicted Valuation: {verdict.get('predicted_valuation_range_usd', 'N/A')}
- Success Probability: {verdict.get('success_probability_percent', 0.0)}%
- SSQ: {ssq.get('ssq_score', 'N/A')} (Momentum: {ssq.get('momentum')}, Efficiency: {ssq.get('efficiency')}, Scalability: {ssq.get('scalability')})
### Risks
- Highest: {max(risks, key=risks.get) if risks else 'N/A'} at {max(risks.values()) if risks else 'N/A'}/10
### Market Context
- TAM: {market.get('total_addressable_market', {}).get('size', 'N/A')}
- Competitor: {market.get('competitive_landscape', [{}])[0].get('name', 'N/A') if market.get('competitive_landscape') else 'N/A'}
- Valuation Trends: {market.get('valuation_trends', {}).get('valuation_multiples', 'N/A')}
- {india_ctx_str}"""
        return summary.strip()

    async def _get_final_verdict(self, raw_pred: Dict[str, float], market_intel: Dict[str, Any], profile: AdvancedStartupProfile) -> Dict[str, Any]:
        # If Gemini is available, synthesize; else compute a simple deterministic verdict
        if self.synthesis_model is not None:
            prompt = (
                "As a VC Partner, review raw ML and market data. "
                f"Profile: Sector={profile.sector}, Stage={profile.stage}. "
                f"ML: {raw_pred}. Market: {market_intel}. "
                "Return JSON: 'predicted_valuation_range_usd', 'success_probability_percent' (0-100)."
            )
            try:
                r = await self.synthesis_model.generate_content_async(prompt)  # type: ignore
                text = getattr(r, "text", "") or ""
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception as e:
                logger.warning(f"[engine] Synthesis (Gemini) failed; falling back: {e}")

        # Deterministic fallback: scale valuation range from ARR and probability
        prob = float(raw_pred.get("success_probability", 0.0))
        arr = float(profile.annual_revenue)
        base_low = max(1.0, arr * 2.0 / 1e6)  # in $M
        base_high = max(base_low + 1.0, arr * 6.0 / 1e6)
        # Boost range modestly with higher probability
        adj = 1.0 + (prob / 200.0)
        low_m = round(base_low * adj, 1)
        high_m = round(base_high * adj, 1)
        return {
            "predicted_valuation_range_usd": f"${low_m}M - ${high_m}M",
            "success_probability_percent": round(prob * 100.0, 1) if prob <= 1.0 else round(prob, 1),
        }

    async def comprehensive_analysis(self, inputs: Dict[str, Any], comps_ticker: str) -> Dict[str, Any]:
        # Resolve strategy
        fa_val = inputs.get("focus_area")
        fa = self._coerce_focus_area(fa_val)
        strategy = self.strategies.get(fa.value, EnhanceUrbanLifestyleStrategy())

        processed = strategy.process_inputs(inputs)

        # Build profile
        profile = AdvancedStartupProfile(
            company_name=str(inputs.get("company_name", "")),
            stage=str(inputs.get("stage", "")),
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

        ssq = SpeedscaleQuotientCalculator(inputs, profile).calculate()

        # Predictions
        pred_tasks: List[asyncio.Future] = []
        async_results: List[Any] = []

        async def _run_online() -> Any:
            if self.online_predictor is None:
                return {"notice": "Online predictor unavailable."}
            try:
                return await asyncio.to_thread(self.online_predictor.predict, inputs)
            except Exception as e:
                logger.warning(f"[engine] Online predictor failed: {e}")
                return {"notice": f"Online predictor error: {e}"}

        async def _run_legacy() -> Any:
            if self.predictor is None:
                return {"notice": "Legacy AIPlus predictor unavailable."}
            try:
                return await asyncio.to_thread(self.predictor.predict, inputs)  # type: ignore
            except Exception as e:
                logger.warning(f"[engine] AIPlus predictor failed: {e}")
                return {"notice": f"Legacy predictor error: {e}"}

        # Schedule available predictors
        pred_tasks.append(asyncio.create_task(_run_online()))
        if self.predictor is not None:
            pred_tasks.append(asyncio.create_task(_run_legacy()))

        # Market intel + enrichment
        intel_task = asyncio.create_task(
            self.market_intel.get_market_intel(
                str(inputs.get("sector", "")),
                str(inputs.get("company_name", "")),
                str(inputs.get("product_desc", "")),
            )
        )

        async def _run_enrichment() -> Dict[str, Any]:
            if self.enrichment is None:
                return {}
            try:
                return await asyncio.to_thread(self.enrichment.enrich, inputs.get("sector", ""), inputs.get("company_name", ""))
            except Exception as e:
                logger.warning(f"[engine] Enrichment failed: {e}")
                return {}

        enrich_task = asyncio.create_task(_run_enrichment())

        # Await predictions (preserving order: online first, then legacy)
        preds = await asyncio.gather(*pred_tasks) if pred_tasks else []
        online_pred = preds[0] if preds else {}
        legacy_pred = preds[1] if len(preds) > 1 else {}

        # Await intel + enrichment
        base_market, enrichment = await asyncio.gather(intel_task, enrich_task)

        market_analysis: Dict[str, Any] = dict(base_market or {})
        if isinstance(enrichment, dict):
            market_analysis["indian_funding_trends"] = enrichment.get("indian_funding_trends", {})
            market_analysis["recent_news"] = enrichment.get("recent_news", {})
            market_analysis["india_funding_dataset_context"] = enrichment.get("india_funding_dataset_context", {})

        # Choose probability/valuation
        chosen_prob = None
        chosen_val = None
        if isinstance(online_pred, dict):
            chosen_prob = online_pred.get("round_probability_12m") or online_pred.get("success_probability")
            chosen_val = online_pred.get("predicted_valuation_usd") or online_pred.get("predicted_next_valuation_usd")
        if chosen_prob is None and isinstance(legacy_pred, dict):
            chosen_prob = legacy_pred.get("success_probability")
        if chosen_val is None and isinstance(legacy_pred, dict):
            chosen_val = legacy_pred.get("predicted_next_valuation_usd")

        try:
            prob_f = float(chosen_prob) if chosen_prob is not None else 0.0
        except Exception:
            prob_f = 0.0
        try:
            val_f = float(chosen_val) if chosen_val is not None else 0.0
        except Exception:
            val_f = 0.0

        final_verdict = await self._get_final_verdict(
            {"success_probability": prob_f, "predicted_next_valuation_usd": val_f},
            market_analysis,
            profile,
        )

        # Parallel: simulation, comps, fundraising forecast
        sim_task = asyncio.to_thread(self.simulation.run_simulation, profile)
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

        risk = self.risk_matrix.assess(profile)
        synthesis_data = {
            "final_verdict": final_verdict,
            "heuristic_profile": asdict(profile),
            "market": market_analysis,
            "risk_scores": risk,
            "speedscale_quotient": ssq,
        }
        summary_for_memo = self._create_synthesis_prompt_text(synthesis_data)
        investment_memo = await self.thesis_gen.generate(summary_for_memo)

        return {
            "final_verdict": final_verdict,
            "investment_memo": investment_memo,
            "risk_matrix": risk,
            "simulation": sim_res,
            "market_deep_dive": market_analysis,
            "public_comps": comps_res,
            "profile": profile,
            "ssq_report": ssq,
            "fundraise_forecast": forecast,
            "ml_predictions": {"online": online_pred, "legacy": legacy_pred},
        }
