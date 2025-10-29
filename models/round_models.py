from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

FEATURES = [
    "age", "arr", "burn", "cash", "ltv_cac_ratio",
    "gross_margin_pct", "monthly_churn_pct",
    "team_size", "num_investors", "investor_quality_score",
    "product_stage_score", "moat_score", "team_score"
]

class RoundLikelihoodModel:
    def __init__(self, model_path: str = "round_model.pkl"):
        self.model_path = model_path
        self.model: Optional[Pipeline] = None

    def _build(self) -> Pipeline:
        base = LogisticRegression(max_iter=200)
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3))
        ])

    def train_from_csv(self, path: str) -> bool:
        try:
            df = pd.read_csv(path)
            if "raised_next_round_12m" not in df.columns:
                return False
            X = df.reindex(columns=[c for c in FEATURES if c in df.columns]).fillna(0.0)
            y = df["raised_next_round_12m"].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe = self._build()
            pipe.fit(X_train, y_train)
            self.model = pipe
            try:
                import joblib
                joblib.dump(self.model, self.model_path)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def load(self) -> bool:
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            return True
        except Exception:
            return False

    def predict_proba(self, inputs: Dict[str, Any], horizon_months: int = 12) -> float:
        if self.model is None:
            arr = float(inputs.get("arr", 0.0))
            burn = float(inputs.get("burn", 1.0)) or 1.0
            runway = float(inputs.get("cash", 0.0)) / burn
            ltv_cac = float(inputs.get("ltv_cac_ratio", 1.0))
            investor_q = float(inputs.get("investor_quality_score", 5.0))
            base = 0.2 + 0.1*np.tanh((arr/1e7)) + 0.05*np.tanh(runway/6.0) + 0.05*np.tanh(ltv_cac/3.0) + 0.05*np.tanh((investor_q-5)/3.0)
            base = max(0.01, min(0.95, base))
            if horizon_months == 6:
                return min(0.9, base * 0.7)
            return base
        X = np.array([[inputs.get(c, 0.0) for c in FEATURES]])
        try:
            proba = float(self.model.predict_proba(X)[0][1])
            if horizon_months == 6:
                proba = 1 - (1 - proba) ** 0.5
            return max(0.0, min(1.0, proba))
        except Exception:
            return 0.5

class TimeToNextFinancingModel:
    def __init__(self):
        self.cox: Optional[CoxPHFitter] = None if LIFELINES_AVAILABLE else None
        self.reg: Optional[GradientBoostingRegressor] = None

    def train_from_csv(self, path: str) -> bool:
        try:
            df = pd.read_csv(path)
            if "months_to_next_round" not in df.columns:
                return False
            X = df.reindex(columns=[c for c in FEATURES if c in df.columns]).fillna(0.0)
            y = df["months_to_next_round"].astype(float)

            if LIFELINES_AVAILABLE and "event_observed" in df.columns:
                sdata = df[[*X.columns, "months_to_next_round", "event_observed"]].copy()
                sdata.rename(columns={"months_to_next_round": "T", "event_observed": "E"}, inplace=True)
                self.cox = CoxPHFitter()
                self.cox.fit(sdata, duration_col="T", event_col="E")
                return True
            else:
                self.reg = GradientBoostingRegressor(random_state=42)
                self.reg.fit(X, y)
                return True
        except Exception:
            return False

    def predict_months(self, inputs: Dict[str, Any]) -> float:
        X = np.array([[inputs.get(c, 0.0) for c in FEATURES]])
        try:
            if self.cox is not None:
                import pandas as pd
                row = pd.DataFrame(X, columns=FEATURES)
                med = self.cox.predict_median(row)
                return float(med.iloc[0]) if med is not None and len(med) else 12.0
            elif self.reg is not None:
                pred = float(self.reg.predict(X)[0])
                return max(1.0, min(48.0, pred))
        except Exception:
            pass
        burn = float(inputs.get("burn", 1.0)) or 1.0
        runway = float(inputs.get("cash", 0.0)) / burn
        investor_q = float(inputs.get("investor_quality_score", 5.0))
        return max(3.0, min(36.0, runway * (0.8 + 0.05*(investor_q-5))))
