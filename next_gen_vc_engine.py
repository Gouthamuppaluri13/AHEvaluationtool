import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import json
import re
from alpha_vantage.timeseries import TimeSeries
import os
import google.generativeai as genai
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusArea(Enum):
    LIVE_HEALTHY="Live Healthy"; MITIGATE_CLIMATE_CHANGE="Mitigate Climate Change"; ENHANCE_URBAN_LIFESTYLE="Enhance Urban Lifestyle"
class FounderPersonality(Enum):
    TECHNICAL="technical"; VISIONARY="visionary"; EXECUTOR="executor"

@dataclass
class AdvancedStartupProfile:
    company_name: str; stage: str; sector: str; focus_area: FocusArea; annual_revenue: float; monthly_burn: float; cash_reserves: float; team_size: int; founder_personality_type: FounderPersonality; market_share_percent: float; investor_quality_score: float; advisor_network_strength: float; product_maturity_score: float=0.0; competitive_advantage_score: float=0.0; adaptability_score: float=0.0; technology_risk_score: float=5.0; regulatory_risk_score: float=5.0; market_risk_score: float=5.0; specialized_metrics: Dict[str, Any] = field(default_factory=dict)

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
        df['founder_has_iit_iim_exp'] = df['founder_bio'].str.contains('IIT|IIM').astype(int); df['avg_web_traffic'] = df['monthly_web_traffic'].apply(np.mean)
        df['web_traffic_growth'] = df['monthly_web_traffic'].apply(lambda x: (x[-1] - x[0]) / (x[0] + 1e-6) if x and len(x) > 1 and x[0] > 0 else 0); return df
    def predict(self, inputs: dict):
        df = pd.DataFrame([inputs]); df_featured = self._feature_engineer(df)
        numeric_features = ['age', 'total_funding_usd', 'num_investors', 'team_size', 'is_dpiit_recognized', 'funding_per_investor', 'funding_per_employee', 'founder_has_iit_iim_exp', 'avg_web_traffic', 'web_traffic_growth']
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
    def process_inputs(self, i: Dict) -> Dict: return {'product_maturity_score':i['product_stage_score'],'competitive_advantage_score':(i['moat_score']*0.4)+(i.get('ltv_cac_ratio',1.0)*1.5),'adaptability_score':i['team_score'],'specialized_metrics':{'LTV/CAC Ratio':i.get('ltv_cac_ratio',1.0)}}
class LiveHealthyStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict:
        c = {"None":1,"Pre-clinical":3,"Phase I/II":6,"Phase III/Approved":9}[i.get('clinical_evidence','None')]
        return {'product_maturity_score':(i['product_stage_score']*0.3)+(c*0.7),'competitive_advantage_score':i['moat_score'],'adaptability_score':i['team_score'],'specialized_metrics':{'Clinical Evidence':i.get('clinical_evidence','None')}}
class MitigateClimateChangeStrategy(FocusAreaStrategy):
    def process_inputs(self, i: Dict) -> Dict: return {'product_maturity_score':(i['product_stage_score']*0.5)+(i.get('trl',1)*0.5),'competitive_advantage_score':i['moat_score'],'adaptability_score':i['team_score'],'technology_risk_score':max(10-i.get('trl',1),1),'specialized_metrics':{'TRL':i.get('trl',1)}}
class MarketIntelligence:
    def __init__(self, key): genai.configure(api_key=key); self.model = genai.GenerativeModel('gemini-1.5-flash')
    async def get_market_intel(self, sector, company, description):
        p = f"""As a top-tier VC market analyst, perform a deep-dive analysis for a startup. Startup: "{company}" in the "{sector}" sector. Description: "{description}". Task: Conduct a thorough market analysis focused on the Indian context. Return ONLY a valid JSON object with keys: "total_addressable_market", "competitive_landscape", "moat_analysis", "valuation_trends", "macro_factors". - total_addressable_market: (Object with keys "size", "projected_growth_rate", "key_drivers" as a list of strings). - competitive_landscape: (List of objects, each with "name", "positioning", "estimated_market_share", "key_strengths", "key_weaknesses"). - moat_analysis: (Object with "type", "defensibility"). - valuation_trends: (Object with "funding_rounds", "valuation_multiples"). - macro_factors: (Object with "tailwinds" as list, "headwinds" as list). Ensure all string values are concise, professional, and well-written paragraphs or lists."""
        try:
            r = await self.model.generate_content_async(p); match = re.search(r'\{.*\}', r.text, re.DOTALL)
            if match: return json.loads(match.group(0))
            else: raise ValueError("No valid JSON found in market intelligence response.")
        except Exception as e: logger.error(f"Market intelligence failed: {e}"); return {"error": f"Failed to generate market deep-dive analysis. Details: {e}"}
class ComprehensiveRiskMatrix:
    def assess(self, p: AdvancedStartupProfile): return {'Market': p.market_risk_score, 'Execution': (10-p.adaptability_score), 'Technology': p.technology_risk_score, 'Regulatory': p.regulatory_risk_score}
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
    
    def _clean_json_string(self, s: str) -> str:
        s = re.sub(r'```json', '', s)
        s = re.sub(r'```', '', s)
        s = s.strip()
        s = s.replace('\\', '\\\\')
        return s

    async def generate(self, summary_text: str):
        p = f"""As a sharp, skeptical, and insightful Partner at a top-tier VC fund (e.g., Sequoia, a16z), analyze the following pre-digested data summary for a startup.
        DATA SUMMARY:
        {summary_text}
        YOUR TASK: Write a world-class investment memo based on the summary. Do not just list the data. Weave a compelling narrative. Synthesize the quantitative predictions with the qualitative market analysis to build a powerful, decisive argument. Your tone must be confident and precise.
        RETURN ONLY a valid JSON object with keys: "executive_summary", "bull_case_narrative", "bear_case_narrative", "recommendation" (must be 'Invest' or 'Pass'), and "conviction" (must be 'High', 'Medium', or 'Low').
        IMPORTANT INSTRUCTIONS FOR NARRATIVES:
        1. The "bull_case_narrative" and "bear_case_narrative" must be a single string containing well-written paragraphs.
        2. Structure these narratives using Markdown `####` subheadings for clarity. The subheadings must be: `#### Market Opportunity`, `#### Team & Execution`, `#### Competitive Landscape & Moat`, and `#### Financial Viability`.
        3. Under each subheading, write a paragraph that explicitly connects the data points from the summary to your strategic insight.
        """
        try:
            r = await self.model.generate_content_async(p)
            cleaned_text = self._clean_json_string(r.text)
            match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                raise ValueError("No valid JSON object found in the LLM response for the investment memo.")
        except Exception as e: 
            logger.error(f"Thesis generation failed: {e}")
            error_msg = f"AI memo generation failed. The model's response could not be parsed. Details: {str(e)}"
            return {"executive_summary": error_msg, "bull_case_narrative": error_msg, "bear_case_narrative": error_msg, "recommendation": "Error", "conviction": "Error"}

class NextGenVCEngine:
    def __init__(self, tavily_key, av_key, gemini_key):
        self.predictor = AIPlusPredictor('ai_plus_model.pth', 'preprocessor.pkl'); self.market_intel = MarketIntelligence(gemini_key)
        self.risk_matrix = ComprehensiveRiskMatrix(); self.simulation = InteractiveSimulation(); self.data_integrator = ExternalDataIntegrator(av_key)
        self.thesis_gen = InvestmentThesisGenerator(gemini_key); self.synthesis_model = genai.GenerativeModel('gemini-1.5-flash')
        self.strategies = {f.value: s() for f, s in zip(FocusArea, [LiveHealthyStrategy, MitigateClimateChangeStrategy, EnhanceUrbanLifestyleStrategy])}
    def _c(self, o):
        if isinstance(o, dict): return {k: self._c(v) for k, v in o.items()}
        if isinstance(o, list): return [self._c(e) for e in o]
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, Enum): return o.value
        return o
    def _create_synthesis_prompt_text(self, data: Dict) -> str:
        profile = self._c(data['heuristic_profile']); verdict = self._c(data['final_verdict']); ssq = self._c(data['speedscale_quotient']); risks = self._c(data['risk_scores']); market = self._c(data['detailed_market_analysis'])
        summary = f"""- Company: {profile.get('company_name')} | Sector: {profile.get('sector')} | Stage: {profile.get('stage')}
### Final Verdict & SSQ
- Predicted Valuation: {verdict.get('predicted_valuation_range_usd', 'N/A')}
- Success Probability: {verdict.get('success_probability_percent', 0.0):.1%}
- Speedscaling Quotient (SSQ): {ssq.get('ssq_score', 'N/A')} / 10 (Momentum: {ssq.get('momentum')}, Efficiency: {ssq.get('efficiency')}, Scalability: {ssq.get('scalability')})
### Key Scores (out of 10)
- Team Score: {profile.get('adaptability_score')}
- Product Maturity: {profile.get('product_maturity_score')}
- Competitive Moat: {profile.get('competitive_advantage_score')}
### Top Risk
- Highest Risk Factor: {max(risks, key=risks.get) if risks else 'N/A'} at {max(risks.values()) if risks else 'N/A'}/10
### Market Context
- TAM Summary: {market.get('total_addressable_market', {}).get('size', 'N/A')}
- Top Competitor: {market.get('competitive_landscape', [{}])[0].get('name', 'N/A')}
- Valuation Trends: {market.get('valuation_trends', {}).get('valuation_multiples', 'N/A')}"""
        return summary.strip()
    async def _get_final_verdict(self, raw_pred, market_intel, profile):
        p = f"""As a VC Partner, review a raw ML model prediction and market data to set a final, realistic valuation. Profile: Sector={profile.sector}, Stage={profile.stage}. Raw ML Prediction: Valuation=${raw_pred['predicted_next_valuation_usd']:,.0f}, Success Prob={raw_pred['success_probability']:.1%}. Market Valuation Trends: "{market_intel.get('valuation_trends', {'valuation_multiples': 'N/A'}).get('valuation_multiples', 'N/A')}". Task: Return ONLY a JSON object with the final, realistic numbers. Keys: "predicted_valuation_range_usd" (string, e.g., "$5M - $7M"), "success_probability_percent" (float)."""
        try:
            r = await self.synthesis_model.generate_content_async(p); match = re.search(r'\{.*\}', r.text, re.DOTALL)
            if match: return json.loads(match.group(0))
            else: raise ValueError("No valid JSON found in final verdict response.")
        except: return {"predicted_valuation_range_usd": "N/A", "success_probability_percent": 0.0}
    async def comprehensive_analysis(self, inputs, comps_ticker):
        strategy = self.strategies[inputs['focus_area']]; processed_scores = strategy.process_inputs(inputs)
        profile = AdvancedStartupProfile(company_name=inputs['company_name'], stage=inputs['stage'], sector=inputs['sector'],focus_area=FocusArea(inputs['focus_area']), annual_revenue=inputs['arr'], monthly_burn=inputs['burn'],cash_reserves=inputs['cash'], team_size=inputs['team_size'],founder_personality_type=FounderPersonality(inputs.get('founder_type', 'visionary')),investor_quality_score=inputs['investor_quality_score'], advisor_network_strength=inputs['advisor_network_strength'],market_share_percent=0, **processed_scores)
        ssq_calculator = SpeedscaleQuotientCalculator(inputs, profile); ssq_report = ssq_calculator.calculate()
        pred_task = asyncio.to_thread(self.predictor.predict, inputs)
        intel_task = self.market_intel.get_market_intel(inputs['sector'], inputs['company_name'], inputs['product_desc'])
        raw_prediction, market_analysis = await asyncio.gather(pred_task, intel_task)
        final_verdict = await self._get_final_verdict(raw_prediction, market_analysis, profile)
        sim_task = asyncio.to_thread(self.simulation.run_simulation, profile)
        comps_task = asyncio.to_thread(self.data_integrator.get_public_comps, comps_ticker)
        sim_res, comps_res = await asyncio.gather(sim_task, comps_task)
        risk = self.risk_matrix.assess(profile)
        synthesis_data = {"final_verdict": final_verdict, "heuristic_profile": asdict(profile), "detailed_market_analysis": market_analysis, "risk_scores": risk, "speedscale_quotient": ssq_report}
        summary_for_memo = self._create_synthesis_prompt_text(synthesis_data)
        investment_memo = await self.thesis_gen.generate(summary_for_memo)
        return {'final_verdict': final_verdict, 'investment_memo': investment_memo, 'risk_matrix': risk, 'simulation': sim_res, 'market_deep_dive': market_analysis, 'public_comps': comps_res, 'profile': profile, 'ssq_report': ssq_report}