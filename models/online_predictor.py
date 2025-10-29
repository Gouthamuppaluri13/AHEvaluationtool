"""
Online ML Predictor
Loads joblib model bundle and provides prediction interface.
"""
import os
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class OnlinePredictor:
    """Handles online predictions using loaded ML model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the joblib model bundle."""
        try:
            import joblib
            
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions from inputs.
        
        Args:
            inputs: Dictionary with startup features
            
        Returns:
            Dictionary with predictions:
            - round_probability_12m: float
            - round_probability_6m: float
            - predicted_valuation_usd: Optional[float]
            - feature_importances: Dict[str, float] (top 20)
            - meta: Dict with model info
        """
        if not self.model_loaded or self.model is None:
            return self._heuristic_fallback(inputs)
        
        try:
            return self._model_predict(inputs)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return self._heuristic_fallback(inputs)
    
    def _model_predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the loaded model."""
        import pandas as pd
        
        # Prepare input data
        df = pd.DataFrame([inputs])
        
        # Get predictions from model
        # Assuming the model has a predict_proba method for classification
        if hasattr(self.model, 'predict_proba'):
            proba_12m = self.model.predict_proba(df)[0][1]  # Probability of positive class
        elif hasattr(self.model, 'predict'):
            proba_12m = float(self.model.predict(df)[0])
        else:
            raise ValueError("Model has no predict or predict_proba method")
        
        # Derive 6-month probability (adjusted down from 12m)
        proba_6m = proba_12m * 0.7
        
        # Try to get valuation prediction if model supports it
        valuation = None
        if hasattr(self.model, 'predict') and len(self.model.predict(df).shape) > 1:
            valuation = float(self.model.predict(df)[0][1])
        
        # Extract feature importances
        importances = self._extract_feature_importances()
        
        return {
            "round_probability_12m": float(proba_12m),
            "round_probability_6m": float(proba_6m),
            "predicted_valuation_usd": valuation,
            "feature_importances": importances,
            "meta": {
                "model_type": type(self.model).__name__,
                "source": "online_model"
            }
        }
    
    def _extract_feature_importances(self) -> Dict[str, float]:
        """Extract top 20 feature importances from model."""
        importances = {}
        
        try:
            # Try to get feature names and importances
            feature_names = None
            importance_values = None
            
            # Check for feature_importances_ (tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
            # Check for coef_ (linear models)
            elif hasattr(self.model, 'coef_'):
                importance_values = np.abs(self.model.coef_).flatten()
            
            # Try to get feature names from the model or pipeline
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            elif hasattr(self.model, 'named_steps'):
                # It's a pipeline, try to get from last step
                last_step = list(self.model.named_steps.values())[-1]
                if hasattr(last_step, 'feature_names_in_'):
                    feature_names = last_step.feature_names_in_
            
            if importance_values is not None and feature_names is not None:
                # Sort and get top 20
                indices = np.argsort(importance_values)[::-1][:20]
                importances = {
                    feature_names[i]: float(importance_values[i])
                    for i in indices
                }
        
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
        
        return importances
    
    def _heuristic_fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Robust heuristic fallback when model is unavailable."""
        logger.info("Using heuristic fallback for predictions")
        
        # Calculate heuristic probability based on key factors
        base_prob = 0.3
        
        # Factor in funding
        if inputs.get('total_funding_usd', 0) > 1000000:
            base_prob += 0.15
        
        # Factor in team size
        if inputs.get('team_size', 0) > 20:
            base_prob += 0.1
        
        # Factor in age
        age = inputs.get('age', 0)
        if 2 <= age <= 5:
            base_prob += 0.15
        
        # Factor in investors
        if inputs.get('num_investors', 0) > 3:
            base_prob += 0.1
        
        # Cap at reasonable value
        proba_12m = min(base_prob, 0.85)
        proba_6m = proba_12m * 0.65
        
        # Estimate valuation
        current_funding = inputs.get('total_funding_usd', 0)
        valuation = current_funding * 2.5 if current_funding > 0 else None
        
        return {
            "round_probability_12m": proba_12m,
            "round_probability_6m": proba_6m,
            "predicted_valuation_usd": valuation,
            "feature_importances": {},
            "meta": {
                "model_type": "heuristic_fallback",
                "source": "fallback"
            }
        }
