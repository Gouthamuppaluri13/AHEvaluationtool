"""
Fundraise Forecast Service
Provides unified interface for fundraising predictions.
"""
import logging
from typing import Dict, Any
from models.round_models import RoundLikelihoodModel, TimeToNextFinancingModel

logger = logging.getLogger(__name__)


class FundraiseForecastService:
    """Service for fundraising forecasting."""
    
    def __init__(self):
        logger.info("Initializing fundraise forecast service")
        self.round_model = RoundLikelihoodModel()
        self.time_model = TimeToNextFinancingModel()
        logger.info("Fundraise forecast service initialized")
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fundraising forecast.
        
        Args:
            inputs: Startup features including:
                - arr: Annual recurring revenue
                - burn: Monthly burn rate
                - cash: Cash on hand
                - team_size: Team size
                - num_investors: Number of investors
                - stage: Funding stage
                - investor_quality_score: Investor quality
                - product_maturity_score: Product maturity
                
        Returns:
            Dictionary with:
            - round_likelihood_6m: Probability of round in 6 months
            - round_likelihood_12m: Probability of round in 12 months
            - expected_time_to_next_round_months: Expected months to next round
        """
        try:
            # Get 12-month probability
            prob_12m = self.round_model.predict(inputs)
            
            # Derive 6-month probability (typically lower)
            prob_6m = prob_12m * 0.7
            
            # Get expected time
            expected_months = self.time_model.predict(inputs)
            
            forecast = {
                'round_likelihood_6m': float(prob_6m),
                'round_likelihood_12m': float(prob_12m),
                'expected_time_to_next_round_months': float(expected_months)
            }
            
            logger.info(f"Generated forecast: 6m={prob_6m:.2%}, 12m={prob_12m:.2%}, time={expected_months:.1f}mo")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            
            # Return conservative fallback
            return {
                'round_likelihood_6m': 0.3,
                'round_likelihood_12m': 0.45,
                'expected_time_to_next_round_months': 12.0
            }
