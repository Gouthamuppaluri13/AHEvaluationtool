"""
Base Evaluator for Sector-Specific Startup Analysis
====================================================
Abstract base class that defines the interface and common functionality
for all sector-specific evaluators in the Anthill AI+ platform.

Author: Anthill AI+ Team
Date: 2025-01-30
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class StartupStage(Enum):
    """Startup development stages"""
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C_PLUS = "series_c_plus"
    GROWTH = "growth"


class MarketCondition(Enum):
    """Market environment conditions"""
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


@dataclass
class EvaluationMetric:
    """Individual evaluation metric with metadata"""
    name: str
    value: float
    weight: float
    explanation: str
    data_sources: List[str]
    confidence: float = 0.8


@dataclass
class SectorEvaluation:
    """Complete sector-specific evaluation result"""
    sector_name: str
    overall_score: float
    confidence_interval: Tuple[float, float]
    metrics: List[EvaluationMetric]
    strengths: List[str]
    weaknesses: List[str]
    red_flags: List[str]
    opportunities: List[str]
    moat_score: float
    market_timing_score: float
    competitive_position: str
    recommended_action: str
    detailed_analysis: Dict[str, Any]


class BaseSectorEvaluator(ABC):
    """
    Abstract base class for sector-specific startup evaluation.
    
    Each sector inherits from this class and implements its own
    evaluation logic based on sector-specific success factors.
    """
    
    def __init__(self, sector_name: str):
        self.sector_name = sector_name
        self.metrics_weights = {}
        self.critical_success_factors = []
        self.typical_challenges = []
        self.data_sources_required = []
        
    @abstractmethod
    def get_sector_specific_metrics(self) -> List[str]:
        """
        Return list of sector-specific metrics that matter.
        
        Example for Space Tech:
        - technology_readiness_level
        - launch_partnerships
        - orbital_analysis_score
        
        Example for E-commerce:
        - brand_sentiment_score
        - cac_ltv_ratio
        - viral_coefficient
        """
        pass
    
    @abstractmethod
    def calculate_sector_score(self, data: Dict[str, Any]) -> SectorEvaluation:
        """
        Calculate comprehensive sector-specific evaluation score.
        
        Args:
            data: Dictionary containing all available data about the startup
            
        Returns:
            SectorEvaluation object with detailed analysis
        """
        pass
    
    @abstractmethod
    def get_required_data_sources(self) -> List[str]:
        """
        Return list of data sources needed for this sector.
        
        Example:
        - Space: ["uspto_patents", "faa_filings", "arxiv_papers"]
        - E-commerce: ["social_media", "web_traffic", "brand_sentiment"]
        """
        pass
    
    @abstractmethod
    def identify_moat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify and quantify competitive moat for this sector.
        
        Returns:
            Dict with moat_type, strength, and explanation
        """
        pass
    
    @abstractmethod
    def assess_market_timing(self, data: Dict[str, Any]) -> float:
        """
        Assess if this is the right time for this startup in this sector.
        
        Returns:
            Score from 0.0 to 10.0
        """
        pass
    
    def adjust_for_stage(self, base_metrics: Dict[str, float], 
                         stage: StartupStage) -> Dict[str, float]:
        """
        Adjust metric weights based on startup stage.
        
        Pre-seed focuses on: Team, Market, Product concept
        Series A focuses on: Traction, Unit economics, Growth
        Series B+ focuses on: Scale, Efficiency, Market leadership
        """
        stage_adjustments = {
            StartupStage.PRE_SEED: {
                'team_quality': 1.5,
                'market_size': 1.3,
                'product_innovation': 1.4,
                'traction': 0.5,
                'revenue': 0.3
            },
            StartupStage.SEED: {
                'team_quality': 1.3,
                'market_validation': 1.3,
                'early_traction': 1.4,
                'product_market_fit': 1.5,
                'revenue': 0.7
            },
            StartupStage.SERIES_A: {
                'growth_rate': 1.5,
                'unit_economics': 1.4,
                'market_penetration': 1.3,
                'revenue': 1.3,
                'team_quality': 1.1
            },
            StartupStage.SERIES_B: {
                'scale_efficiency': 1.5,
                'market_leadership': 1.4,
                'revenue_growth': 1.4,
                'profitability_path': 1.3,
                'competitive_moat': 1.3
            },
            StartupStage.SERIES_C_PLUS: {
                'market_dominance': 1.5,
                'profitability': 1.5,
                'international_expansion': 1.3,
                'strategic_partnerships': 1.2,
                'exit_readiness': 1.4
            }
        }
        
        adjustments = stage_adjustments.get(stage, {})
        adjusted = base_metrics.copy()
        
        for metric, value in adjusted.items():
            if metric in adjustments:
                adjusted[metric] = value * adjustments[metric]
                
        return adjusted
    
    def adjust_for_market_condition(self, base_score: float, 
                                   condition: MarketCondition,
                                   sector_sensitivity: float = 1.0) -> float:
        """
        Adjust scores based on market conditions.
        
        Args:
            base_score: Original evaluation score
            condition: Current market condition
            sector_sensitivity: How sensitive this sector is to markets (0-2)
        """
        adjustments = {
            MarketCondition.BULL: 1.1,
            MarketCondition.NEUTRAL: 1.0,
            MarketCondition.BEAR: 0.9
        }
        
        adjustment = adjustments[condition]
        # Apply sector sensitivity
        adjustment = 1.0 + (adjustment - 1.0) * sector_sensitivity
        
        return base_score * adjustment
    
    def calculate_confidence_interval(self, score: float, 
                                     data_quality: float = 0.8) -> Tuple[float, float]:
        """
        Calculate confidence interval for the prediction.
        
        Args:
            score: Base score
            data_quality: Quality of available data (0-1)
        """
        # Lower data quality = wider confidence interval
        std_dev = (1.0 - data_quality) * 2.0
        lower = max(0.0, score - 1.96 * std_dev)
        upper = min(10.0, score + 1.96 * std_dev)
        
        return (lower, upper)
    
    def identify_red_flags(self, data: Dict[str, Any]) -> List[str]:
        """
        Identify potential red flags common across sectors.
        Sector-specific evaluators can override this for specialized checks.
        """
        red_flags = []
        
        # Financial red flags
        if data.get('burn_rate', 0) > data.get('cash', 0) / 3:
            red_flags.append("⚠️ High burn rate - runway less than 3 months")
        
        if data.get('revenue_growth_rate', 0) < 0:
            red_flags.append("⚠️ Negative revenue growth")
        
        # Team red flags
        if data.get('founder_experience', 0) < 2:
            red_flags.append("⚠️ Limited founder experience in domain")
        
        if data.get('team_turnover', 0) > 0.3:
            red_flags.append("⚠️ High team turnover rate (>30%)")
        
        # Market red flags
        if data.get('customer_concentration', 0) > 0.5:
            red_flags.append("⚠️ High customer concentration risk")
        
        if data.get('competitive_intensity', 0) > 8:
            red_flags.append("⚠️ Extremely competitive market")
        
        # Legal/Compliance red flags
        if data.get('pending_litigation', False):
            red_flags.append("🚨 Pending litigation detected")
        
        if data.get('regulatory_violations', 0) > 0:
            red_flags.append("🚨 Regulatory violations on record")
        
        return red_flags
    
    def identify_strengths(self, metrics: List[EvaluationMetric]) -> List[str]:
        """Identify top strengths based on metric performance"""
        strengths = []
        
        # Sort metrics by value * weight
        scored_metrics = [(m, m.value * m.weight) for m in metrics]
        scored_metrics.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3-5 strengths
        for metric, score in scored_metrics[:5]:
            if metric.value >= 7.0:
                strengths.append(f"✅ Strong {metric.name}: {metric.explanation}")
        
        return strengths
    
    def identify_weaknesses(self, metrics: List[EvaluationMetric]) -> List[str]:
        """Identify areas needing improvement"""
        weaknesses = []
        
        # Sort metrics by value * weight
        scored_metrics = [(m, m.value * m.weight) for m in metrics]
        scored_metrics.sort(key=lambda x: x[1])
        
        # Bottom 3-5 weaknesses
        for metric, score in scored_metrics[:5]:
            if metric.value < 5.0:
                weaknesses.append(f"⚠️ Weak {metric.name}: {metric.explanation}")
        
        return weaknesses
    
    def calculate_weighted_score(self, metrics: List[EvaluationMetric]) -> float:
        """Calculate overall weighted score from individual metrics"""
        if not metrics:
            return 5.0
        
        total_weight = sum(m.weight for m in metrics)
        if total_weight == 0:
            return 5.0
        
        weighted_sum = sum(m.value * m.weight for m in metrics)
        return weighted_sum / total_weight
    
    def generate_recommendation(self, overall_score: float, 
                               moat_score: float,
                               red_flags: List[str]) -> str:
        """Generate investment recommendation"""
        if len(red_flags) >= 3:
            return "❌ PASS - Multiple critical red flags"
        
        if overall_score >= 8.0 and moat_score >= 7.0:
            return "🚀 STRONG BUY - Exceptional opportunity"
        elif overall_score >= 7.0 and moat_score >= 6.0:
            return "✅ BUY - Solid investment opportunity"
        elif overall_score >= 6.0:
            return "🤔 HOLD - Monitor closely, potential with execution"
        elif overall_score >= 4.0:
            return "⚠️ CAUTION - Significant risks present"
        else:
            return "❌ PASS - Does not meet investment criteria"
    
    def __repr__(self):
        return f"<{{self.__class__.__name__}} sector={{self.sector_name}}>",
