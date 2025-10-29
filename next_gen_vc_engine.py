import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import json
import re
from alpha_vantage.timeseries import TimeSeries
import google.generativeai as genai
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle

# NEW imports
from services.data_enrichment import DataEnrichmentService
from services.fundraise_forecast import FundraiseForecastService
from services.model_registry import ModelRegistry
from models.online_predictor import OnlinePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusArea(Enum):
    LIVE_HEALTHY="Live Healthy"; MITIGATE_CLIMATE_CHANGE="Mitigate Climate Change"; ENHANCE_URBAN_LIFESTYLE="Enhance Urban Lifestyle"
class FounderPersonality(Enum):
    TECHNICAL="technical"; VISIONARY="visionary"; EXECUTOR="executor"

@dataclass
class AdvancedStartupProfile:
    company_name: str; stage: str; sector: str; focus_area: FocusArea; annual_revenue: float; monthly_burn: float; cash_reserves: float; team_size: int; founder_personality_type: FounderPersonality
    product_maturity_score: float; competitive_advantage_score: float; adaptability_score: float; investor_quality_score: float
    market_risk_score: float = 5.0; technology_risk_score: float = 5.0; regulatory_risk_score: float = 5.0

class SpeedscaleQuotientCalculator:
    def __init__(self, inputs: Dict[str, Any], profile: AdvancedStartupProfile):
        self.inputs = inputs; self.profile = profile; self.weights = self._get_contextual_weights()
    def _get_contextual_weights(self) -> Dict[str, float]:
        if self.profile.focus_area == FocusArea.ENHANCE_URBAN_LIFESTYLE: return {'momentum': 0.5, 'efficiency': 0.3, 'scalability': 0.2}
        elif self.profile.focus_area == FocusArea.LIVE_HEALTHY: return {'momentum': 0.3, 'efficiency': 0.3, 'scalability': 0.4}
        elif self.profile.focus_area == FocusArea.MITIGATE_CLIMATE_CHANGE: return {'momentum': 0.2, 'efficiency': 0.3, 'scalability': 0.5}
        return {'momentum': 0.33, 'efficiency': 0.33, 'scalability': 0.34}
    def _calculate_momentum_score(self) -> float:
        traffic_growth = self.inputs.get('monthly_web_traffic', [0, 0])
        growth_rate = ((traffic_growth[-1] - traffic_growth[0]) / (traffic_growth[0] + 1e-6)) if len(traffic_growth) > 1 and traffic_growth[0] > 0 else 0
        normalized_growth = min(growth_rate / 5.0, 1.0) * 10
        return (normalized_growth * 0.6) + (self.profile.product_maturity_score * 0.4)
    def _calculate_efficiency_score(self) -> float:
        ltv_cac = self.inputs.get('ltv_cac_ratio', 1.0); normalized_ltv_cac = min(ltv_cac / 5.0, 1.0) * 10
        burn = self.inputs.get('burn', 0); arr = self.inputs.get('arr', 0)
        burn_multiple = (burn * 12) / (arr + 1e-6) if arr > 0 else 99
        normalized_burn = max(0, 1 - (burn_multiple / 3.0)) * 10
        return (normalized_ltv_cac * 0.5) + (normalized_burn * 0.5)
    def _calculate_scalability_score(self) -> float:
        return (self.profile.adaptability_score * 0.4) + (self.profile.competitive_advantage_score * 0.4) + (self.profile.investor_quality_score * 0.2)
    def calculate(self) -> Dict[str, float]:
        momentum = self._calculate_momentum_score(); efficiency = self._calculate_efficiency_score(); scalability = self._calculate_scalability_score()
        ssq_score = (momentum * self.weights['momentum'] + efficiency * self.weights['efficiency'] + scalability * self.weights['scalability'])
        return {'ssq_score': round(ssq_score, 1), 'momentum': round(momentum, 1), 'efficiency': round(efficiency, 1), 'scalability': round(scalability, 1)}

class AIPlusModel(nn.Module):
    def __init__(self, num_numeric_features, tab_e=64, txt_e=64, ts_e=32):
        super().__init__(); self.tabular_encoder = TabularEncoder(num_numeric_features, tab_e); self.text_encoder = TextEncoder(txt_e); self.ts_encoder = TimeSeriesEncoder(1, 64, 4, ts_e)
        self.fusion_mlp = nn.Sequential(nn.Linear(tab_e + txt_e + ts_e, 256), nn.ReLU(), nn.Dropout(0.5)); self.success_head = nn.Linear(256, 1); self.valuation_head = nn.Linear(256, 1)
    def forward(self, n, i, a, t):
        f = torch.cat([self.tabular_encoder(n), self.text_encoder(i, a), self.ts_encoder(t)], dim=1); fused = self.fusion_mlp(f)
        return torch.cat([self.success_head(fused), self.valuation_head(fused)], dim=1)
class TabularEncoder(nn.Module):
    def __init__(self, i, o): super().__init__(); self.net = nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3), nn.Linear(128, o))
    def forward(self, x): return self.net(x)
class TextEncoder(nn.Module):
    def __init__(self, o): super().__init__(); self.bert = BertModel.from_pretrained('bert-base-uncased'); self.fc = nn.Linear(self.bert.config.hidden_size, o)
    def forward(self, i, a): return self.fc(self.bert(input_ids=i, attention_mask=a).pooler_output)
class TimeSeriesEncoder(nn.Module):
    def __init__(self, i, e, n, o):
        super().__init__(); self.embedding = nn.Linear(i, e); el = nn.TransformerEncoderLayer(d_model=e, nhead=n, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(el, num_layers=2); self.fc = nn.Linear(e, o)
    def forward(self, x): return self.fc(self.transformer_encoder(self.embedding(x)).mean(dim=1))
class AIPlusPredictor:
    def __init__(self, model_path, preprocessor_path):
        with open(preprocessor_path, 'rb') as f: self.preprocessor = pickle.load(f); num_features = len(self.preprocessor.get_feature_names_out())
        self.model = AIPlusModel(num_numeric_features=num_features); self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))); self.model.eval(); self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def _feature_engineer(self, df):
        df['funding_per_investor'] = df['total_funding_usd'] / (df['num_investors'] + 1e-6); df['funding_per_employee'] = df['total_funding_usd'] / (df['team_size'] + 1e-6)
        df['founder_has_iit_iim_exp'] = df['founder_bio'].str.contains('IIT|IIM', case=False, regex=True).astype(int); df['avg_web_traffic'] = df['monthly_web_traffic'].apply(np.mean)
        df['web_traffic_growth'] = df['monthly_web_traffic'].apply(lambda x: (x[-1] - x[0]) / (x[0] + 1e-6) if x and len(x) > 1 and x[0] > 0 else 0); return df
    def predict(self, inputs: dict):
        df = pd.DataFrame([inputs]); df_featured = self._feature_engineer(df)
        numeric_features = ['age', 'total_funding_usd', 'num_investors', 'team_size', 'is_dpiit_recognized', 'funding_per_investor', 'funding_per_employee', 'founder_has_iit_iim_exp', 'avg_web_traffic', 'web_traffic_growth', 'product_stage_score', 'team_score', 'moat_score', 'ltv_cac_ratio']
        categorical_features = ['sector', 'location']; numeric_cat_data = self.preprocessor.transform(df_featured[numeric_features + categorical_features])
        text = df_featured['founder_bio'].iloc[0] + " " + df_featured['product_desc'].iloc[0]
        tokenized_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        time_series_data = torch.tensor(df_featured['monthly_web_traffic'].iloc[0], dtype=torch.float32).unsqueeze(1).unsqueeze(0)
        with torch.no_grad(): outputs = self.model(n=torch.tensor(numeric_cat_data, dtype=torch.float32), i=tokenized_inputs['input_ids'], a=tokenized_inputs['attention_mask'], t=time_series_data)
        return {"success_probability": torch.sigmoid(outputs[:, 0]).item(), "predicted_next_valuation_usd": outputs[:, 1].item()}
class ExternalDataIntegrator:
    def __init__(self, key): self.ts = TimeSeries(key=key, output_format='pandas')
    def get_public_comps(self, ticker: str):
        try:
            data, meta = self.ts.get_quote_endpoint(symbol=ticker); price = float(data['05. price'].iloc[-1])
            return {"Company": meta.get('2. Symbol', ticker), "Price (₹)": f"{price:,.2f}", "Market": meta.get("1. symbol", ticker).split('.')[-1]}
        except: return {"Error": f"API call failed for ticker {ticker}."}
class FocusAreaStrategy(ABC):
    @abstractmethod
    def process_inputs(self, i: Dict) -> Dict: pass
class EnhanceUrbanLifestyleStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict: return {'product_maturity_score':i['product_stage_score'],'competitive_advantage_score':(i['moat_score']*0.4)+(i.get('ltv_cac_ratio',1.0)*1.5),'adaptability_score':i['team_score']}
class LiveHealthyStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict:
        c = {"None":1,"Pre-clinical":3,"Phase I/II":6,"Phase III/Approved":9}[i.get('clinical_evidence','None')]
        return {'product_maturity_score':(i['product_stage_score']*0.3)+(c*0.7),'competitive_advantage_score':i['moat_score'],'adaptability_score':i['team_score'],'specialized_metrics':{'ClinicalEvidenceScore':c}}
class MitigateClimateChangeStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict: return {'product_maturity_score':(i['product_stage_score']*0.5)+(i.get('trl',1)*0.5),'competitive_advantage_score':i['moat_score'],'adaptability_score':i['team_score']}
class MarketIntelligence:
    def __init__(self, key): genai.configure(api_key=key); self.model = genai.GenerativeModel('gemini-1.5-flash')
    async def get_market_intel(self, sector, company, description):
        p = f"""As a top-tier VC market analyst, perform a deep-dive analysis for a startup. Startup: "{company}" in the "{sector}" sector. Description: "{description}". Task: Return JSON with: total_addressable_market, competitive_landscape, moat_analysis, valuation_trends, regulatory_outlook, supply_demand_dynamics."""
        try:
            r = await self.model.generate_content_async(p); match = re.search(r'\{.*\}', r.text, re.DOTALL)
            if match: return json.loads(match.group(0))
            else: raise ValueError("No valid JSON found in market intelligence response.")
        except Exception as e: logger.error(f"Market intelligence failed: {e}"); return {"error": f"Failed to generate market deep-dive analysis. Details: {e}"}
class ComprehensiveRiskMatrix:
    def assess(self, p: AdvancedStartupProfile): return {'Market': p.market_risk_score, 'Execution': (10-p.adaptability_score), 'Technology': p.technology_risk_score, 'Regulatory': p.regulatory_risk_score, 'Competition': (10-p.competitive_advantage_score)}
class InteractiveSimulation:
    def run_simulation(self, p: AdvancedStartupProfile):
        cash, rev, burn = p.cash_reserves, p.annual_revenue/12, p.monthly_burn; hist=[]
        for _ in range(36):
            rev *= 1.05; cash += (rev - burn); hist.append({"Month":len(hist)+1, "Cash Reserves (₹)":cash, "Monthly Revenue (₹)":rev})
            if cash <= 0: break
        summary = "survives beyond the 36-month horizon." if hist and hist[-1]["Cash Reserves (₹)"] > 0 else f"runs out of cash in Month {len(hist)}."
        return {'time_series_data': pd.DataFrame(hist), 'narrative_summary': f"Based on current financials and a modest 5% monthly revenue growth, the company {summary}"}

class InvestmentThesisGenerator:
    def __init__(self, key):
        genai.configure(api_key=key); self.model = genai.GenerativeModel('gemini-1.5-flash')
    async def generate(self, summary_text: str):
        p = f"""As a VC Partner, write an investment memo as valid JSON with keys: executive_summary, bull_case_narrative, bear_case_narrative, recommendation ('Invest' or 'Pass'), conviction ('High'|'Medium'|'Low'). Data summary: {summary_text}"""
        try:
            r = await self.model.generate_content_async(p)
            match = re.search(r'\{.*\}', r.text, re.DOTALL)
            if match: return json.loads(match.group(0))
            else: raise ValueError("No valid JSON object found in the LLM response for the investment memo.")
        except Exception as e:
            logger.error(f"Thesis generation failed: {e}")
            msg = "AI memo generation failed. Response could not be parsed."
            return {"executive_summary": msg, "bull_case_narrative": msg, "bear_case_narrative": msg, "recommendation": "Error", "conviction": "Error"}

class NextGenVCEngine:
    def __init__(self, tavily_key, av_key, gemini_key):
        self.predictor = AIPlusPredictor('ai_plus_model.pth', 'preprocessor.pkl'); self.market_intel = MarketIntelligence(gemini_key)
        self.risk_matrix = ComprehensiveRiskMatrix(); self.simulation = InteractiveSimulation(); self.data_integrator = ExternalDataIntegrator(av_key)
        self.thesis_gen = InvestmentThesisGenerator(gemini_key); self.synthesis_model = genai.GenerativeModel('gemini-1.5-flash')
        self.strategies = {f.value: s() for f, s in zip(FocusArea, [LiveHealthyStrategy, MitigateClimateChangeStrategy, EnhanceUrbanLifestyleStrategy])}
        # NEW services
        self.enrichment = DataEnrichmentService(tavily_key)
        self.fundraise_forecast = FundraiseForecastService()
        # NEW: online model
        registry = ModelRegistry()
        joblib_path = registry.load_joblib_path()
        self.online_predictor = OnlinePredictor(joblib_path)

    def _c(self, o):
        if isinstance(o, dict): return {k: self._c(v) for k, v in o.items()}
        if isinstance(o, list): return [self._c(e) for e in o]
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, Enum): return o.value
        return o
    def _create_synthesis_prompt_text(self, data: Dict) -> str:
        profile = self._c(data['heuristic_profile']); verdict = self._c(data['final_verdict']); ssq = self._c(data['speedscale_quotient']); risks = self._c(data['risk_scores']); market = self._c(data['market'])
        india_ctx = market.get('india_funding_dataset_context', {})
        india_ctx_str = ""
        if india_ctx:
            inv = ", ".join([i['name'] for i in india_ctx.get('top_investors', [])[:8]])
            india_ctx_str = f"India Sector Context — Median Round (INR): {india_ctx.get('median_amount_inr','N/A')}; Yearly Rounds: {[y['year'] for y in india_ctx.get('yearly_rounds', [])][-5:]}; Top Investors: {inv}"
        summary = f"""- Company: {profile.get('company_name')} | Sector: {profile.get('sector')} | Stage: {profile.get('stage')}
### Final Verdict & SSQ
- Predicted Valuation: {verdict.get('predicted_valuation_range_usd', 'N/A')}
- Success Probability: {verdict.get('success_probability_percent', 0.0):.1%}
- SSQ: {ssq.get('ssq_score', 'N/A')} (Momentum: {ssq.get('momentum')}, Efficiency: {ssq.get('efficiency')}, Scalability: {ssq.get('scalability')})
### Risks
- Highest: {max(risks, key=risks.get) if risks else 'N/A'} at {max(risks.values()) if risks else 'N/A'}/10
### Market Context
- TAM: {market.get('total_addressable_market', {}).get('size', 'N/A')}
- Competitor: {market.get('competitive_landscape', [{}])[0].get('name', 'N/A')}
- Valuation Trends: {market.get('valuation_trends', {}).get('valuation_multiples', 'N/A')}
- {india_ctx_str}"""
        return summary.strip()
    async def _get_final_verdict(self, raw_pred, market_intel, profile):
        p = f"""As a VC Partner, review raw ML and market data. Profile: Sector={profile.sector}, Stage={profile.stage}. ML: {raw_pred}. Market: {market_intel}. Return JSON: 'predicted_valuation_range_usd', 'success_probability_percent' (0-100)."""
        try:
            r = await self.synthesis_model.generate_content_async(p); match = re.search(r'\{.*\}', r.text, re.DOTALL)
            if match: return json.loads(match.group(0))
            else: raise ValueError("No valid JSON found in final verdict response.")
        except: return {"predicted_valuation_range_usd": "N/A", "success_probability_percent": 0.0}
    async def comprehensive_analysis(self, inputs, comps_ticker):
        strategy = self.strategies[inputs['focus_area']]; processed_scores = strategy.process_inputs(inputs)
        profile = AdvancedStartupProfile(company_name=inputs['company_name'], stage=inputs['stage'], sector=inputs['sector'],focus_area=FocusArea(inputs['focus_area']), annual_revenue=inputs['arr'], monthly_burn=inputs['burn'], cash_reserves=inputs['cash'], team_size=inputs['team_size'], founder_personality_type=FounderPersonality(inputs['founder_type']), product_maturity_score=processed_scores['product_maturity_score'], competitive_advantage_score=processed_scores['competitive_advantage_score'], adaptability_score=processed_scores['adaptability_score'], investor_quality_score=inputs.get('investor_quality_score', 5.0))
        ssq_calculator = SpeedscaleQuotientCalculator(inputs, profile); ssq_report = ssq_calculator.calculate()

        preds_task = asyncio.gather(
            asyncio.to_thread(self.online_predictor.predict, inputs),
            asyncio.to_thread(self.predictor.predict, inputs)
        )
        intel_task = self.market_intel.get_market_intel(inputs['sector'], inputs['company_name'], inputs['product_desc'])
        enrichment_task = asyncio.to_thread(self.enrichment.enrich, inputs['sector'], inputs['company_name'])

        (online_pred, legacy_pred), (base_market, enrichment) = await asyncio.gather(preds_task, asyncio.gather(intel_task, enrichment_task))

        market_analysis = dict(base_market or {})
        if isinstance(enrichment, dict):
            market_analysis["indian_funding_trends"] = enrichment.get("indian_funding_trends", {})
            market_analysis["recent_news"] = enrichment.get("recent_news", {})
            market_analysis["india_funding_dataset_context"] = enrichment.get("india_funding_dataset_context", {})

        chosen_prob = online_pred.get("round_probability_12m") if isinstance(online_pred, dict) else None
        chosen_val = online_pred.get("predicted_valuation_usd") if isinstance(online_pred, dict) else None
        if chosen_prob is None and "success_probability" in (legacy_pred or {}):
            chosen_prob = float(legacy_pred["success_probability"])
        if chosen_val is None and "predicted_next_valuation_usd" in (legacy_pred or {}):
            chosen_val = float(legacy_pred["predicted_next_valuation_usd"])

        final_verdict = await self._get_final_verdict(
            {"success_probability": chosen_prob or 0.0, "predicted_next_valuation_usd": chosen_val or 0.0},
            market_analysis, profile
        )

        sim_task = asyncio.to_thread(self.simulation.run_simulation, profile)
        comps_task = asyncio.to_thread(self.data_integrator.get_public_comps, comps_ticker)
        fundraise_task = asyncio.to_thread(self.fundraise_forecast.predict, inputs)
        sim_res, comps_res, forecast = await asyncio.gather(sim_task, comps_task, fundraise_task)

        risk = self.risk_matrix.assess(profile)
        synthesis_data = {"final_verdict": final_verdict, "heuristic_profile": asdict(profile), "market": market_analysis, "risk_scores": risk, "speedscale_quotient": ssq_report}
        summary_for_memo = self._create_synthesis_prompt_text(synthesis_data)
        investment_memo = await self.thesis_gen.generate(summary_for_memo)

        return {
            'final_verdict': final_verdict,
            'investment_memo': investment_memo,
            'risk_matrix': risk,
            'simulation': sim_res,
            'market_deep_dive': market_analysis,
            'public_comps': comps_res,
            'profile': profile,
            'ssq_report': ssq_report,
            'fundraise_forecast': forecast,
            'ml_predictions': {"online": online_pred, "legacy": legacy_pred}
        }
