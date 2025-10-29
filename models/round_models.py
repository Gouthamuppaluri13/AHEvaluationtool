"""
Round Likelihood and Time-to-Next-Financing Models
Calibrated models for fundraising forecasting.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RoundLikelihoodModel:
    """Predicts likelihood of next funding round."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self._try_train()
    
    def _try_train(self):
        """Try to train model from available CSV data."""
        try:
            # Look for training data files
            data_files = [
                'next_gen_deal_data.csv',
                'default_training_data.csv'
            ]
            
            df = None
            for filename in data_files:
                if os.path.exists(filename):
                    logger.info(f"Found training data: {filename}")
                    df = pd.read_csv(filename)
                    break
            
            if df is None:
                logger.info("No training data found, using heuristics only")
                return
            
            # Train calibrated model
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features
            feature_cols = []
            for col in ['annual_revenue', 'monthly_burn', 'team_size', 
                       'investor_quality_score', 'product_maturity_score']:
                if col in df.columns:
                    feature_cols.append(col)
            
            if not feature_cols:
                logger.warning("No suitable features found in training data")
                return
            
            # Fill missing values
            X = df[feature_cols].fillna(df[feature_cols].median())
            
            # Create synthetic target if not present
            if 'next_round_raised' in df.columns:
                y = df['next_round_raised']
            elif 'exit_valuation' in df.columns:
                # Use exit valuation as proxy
                y = (df['exit_valuation'] > df['annual_revenue'] * 2).astype(int)
            else:
                logger.warning("No target variable found for training")
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train calibrated classifier
            base_model = LogisticRegression(random_state=42)
            self.model = CalibratedClassifierCV(base_model, cv=3)
            self.model.fit(X_scaled, y)
            
            self.scaler = scaler
            self.feature_cols = feature_cols
            self.is_trained = True
            
            logger.info("Round likelihood model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train round likelihood model: {e}")
            self.is_trained = False
    
    def predict(self, inputs: Dict[str, Any]) -> float:
        """
        Predict probability of next round in 12 months.
        
        Args:
            inputs: Startup features
            
        Returns:
            Probability between 0 and 1
        """
        if self.is_trained and self.model:
            try:
                # Prepare features
                features = []
                for col in self.feature_cols:
                    # Map input keys to feature columns
                    value = inputs.get(col, 0)
                    if col == 'annual_revenue':
                        value = inputs.get('arr', value)
                    elif col == 'monthly_burn':
                        value = inputs.get('burn', value)
                    features.append(value)
                
                # Scale and predict
                X = self.scaler.transform([features])
                proba = self.model.predict_proba(X)[0][1]
                
                return float(proba)
                
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                return self._heuristic_predict(inputs)
        
        return self._heuristic_predict(inputs)
    
    def _heuristic_predict(self, inputs: Dict[str, Any]) -> float:
        """Heuristic fallback for probability estimation."""
        score = 0.3  # Base probability
        
        # Growth indicators
        arr = inputs.get('arr', 0)
        if arr > 50000000:  # 5 Cr
            score += 0.15
        
        burn = inputs.get('burn', 0)
        if arr > burn * 6:  # Positive unit economics
            score += 0.15
        
        # Team and funding
        if inputs.get('team_size', 0) > 30:
            score += 0.1
        
        if inputs.get('num_investors', 0) > 5:
            score += 0.1
        
        # Stage
        stage = inputs.get('stage', '')
        if 'Seed' in stage:
            score += 0.1
        elif 'Series A' in stage:
            score += 0.05
        
        return min(score, 0.95)


class TimeToNextFinancingModel:
    """Predicts time until next financing event."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self._try_train()
    
    def _try_train(self):
        """Try to train survival/regression model."""
        try:
            # Look for event data
            data_files = [
                'next_gen_deal_data.csv',
                'default_training_data.csv'
            ]
            
            df = None
            for filename in data_files:
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    break
            
            if df is None:
                logger.info("No training data found for time-to-financing model")
                return
            
            # Check if we have event data for survival analysis
            has_event_data = 'time_to_next_round' in df.columns or 'months_to_next_round' in df.columns
            
            if has_event_data:
                # Use lifelines Cox model
                try:
                    from lifelines import CoxPHFitter
                    
                    # Prepare data for Cox model
                    feature_cols = []
                    for col in ['annual_revenue', 'monthly_burn', 'cash_reserves', 
                               'team_size', 'investor_quality_score']:
                        if col in df.columns:
                            feature_cols.append(col)
                    
                    if not feature_cols:
                        raise ValueError("No features for Cox model")
                    
                    # Get duration and event columns
                    if 'time_to_next_round' in df.columns:
                        duration_col = 'time_to_next_round'
                    else:
                        duration_col = 'months_to_next_round'
                    
                    event_col = 'event_observed' if 'event_observed' in df.columns else None
                    
                    # Prepare dataframe
                    cox_df = df[feature_cols + [duration_col]].copy()
                    cox_df = cox_df.fillna(cox_df.median())
                    
                    if event_col:
                        cox_df[event_col] = df[event_col].fillna(1)
                    else:
                        cox_df['event'] = 1
                        event_col = 'event'
                    
                    # Fit Cox model
                    cph = CoxPHFitter()
                    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
                    
                    self.model = cph
                    self.feature_cols = feature_cols
                    self.model_type = 'cox'
                    self.is_trained = True
                    
                    logger.info("Cox survival model trained successfully")
                    return
                    
                except Exception as e:
                    logger.warning(f"Cox model training failed: {e}, falling back to gradient boosting")
            
            # Fallback to gradient boosting regression
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            
            feature_cols = []
            for col in ['annual_revenue', 'monthly_burn', 'cash_reserves',
                       'team_size', 'investor_quality_score']:
                if col in df.columns:
                    feature_cols.append(col)
            
            if not feature_cols:
                return
            
            X = df[feature_cols].fillna(df[feature_cols].median())
            
            # Create synthetic target
            if 'months_to_next_round' in df.columns:
                y = df['months_to_next_round']
            else:
                # Estimate based on runway
                cash = df.get('cash_reserves', 0)
                burn = df.get('monthly_burn', 1)
                y = (cash / (burn + 1e-6)).clip(0, 36)
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            self.scaler = scaler
            self.feature_cols = feature_cols
            self.model_type = 'gbr'
            self.is_trained = True
            
            logger.info("Gradient boosting time model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train time-to-financing model: {e}")
            self.is_trained = False
    
    def predict(self, inputs: Dict[str, Any]) -> float:
        """
        Predict expected months until next financing.
        
        Args:
            inputs: Startup features
            
        Returns:
            Expected months (float)
        """
        if self.is_trained and self.model:
            try:
                features = []
                for col in self.feature_cols:
                    value = inputs.get(col, 0)
                    if col == 'annual_revenue':
                        value = inputs.get('arr', value)
                    elif col == 'monthly_burn':
                        value = inputs.get('burn', value)
                    elif col == 'cash_reserves':
                        value = inputs.get('cash', value)
                    features.append(value)
                
                if self.model_type == 'cox':
                    # Cox model prediction (median survival time)
                    import pandas as pd
                    X_df = pd.DataFrame([features], columns=self.feature_cols)
                    median_time = self.model.predict_median(X_df).iloc[0]
                    return float(median_time)
                else:
                    # Gradient boosting prediction
                    X = self.scaler.transform([features])
                    months = self.model.predict(X)[0]
                    return float(max(months, 1))
                
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                return self._heuristic_predict(inputs)
        
        return self._heuristic_predict(inputs)
    
    def _heuristic_predict(self, inputs: Dict[str, Any]) -> float:
        """Heuristic fallback for time estimation."""
        # Base estimate on runway
        cash = inputs.get('cash', 0)
        burn = inputs.get('burn', 1)
        
        runway_months = cash / (burn + 1e-6)
        
        # Companies typically raise 3-6 months before running out
        time_to_next = max(runway_months - 4, 3)
        
        # Adjust based on growth
        arr = inputs.get('arr', 0)
        if arr > burn * 12:  # Revenue covers burn
            time_to_next *= 1.5
        
        # Adjust based on stage
        stage = inputs.get('stage', '')
        if 'Seed' in stage:
            time_to_next = min(time_to_next, 12)
        elif 'Series A' in stage:
            time_to_next = min(time_to_next, 18)
        else:
            time_to_next = min(time_to_next, 24)
        
        return max(time_to_next, 3)
