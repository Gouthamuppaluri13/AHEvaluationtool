from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return float(d)

class OnlinePredictor:
    """
    Loads a scikit-learn Pipeline bundle (joblib) trained from Kaggle datasets.
    Bundle schema:
      {
        "clf": Pipeline,                  # required
        "reg": Optional[Pipeline],        # optional valuation regressor
        "feature_list": List[str],        # required
        "categoricals": List[str],        # optional
        "meta": {...}                     # optional
      }
    """
    def __init__(self, joblib_path: Optional[str]):
        self.ok = False
        self.clf = None
        self.reg = None
        self.feature_list = []
        self.categoricals = []
        self.meta = {}

        if joblib_path:
            try:
                import joblib
                bundle = joblib.load(joblib_path)
                self.clf = bundle.get("clf")
                self.reg = bundle.get("reg")
                self.feature_list = bundle.get("feature_list", [])
                self.categoricals = bundle.get("categoricals", [])
                self.meta = bundle.get("meta", {})
                self.ok = self.clf is not None and len(self.feature_list) > 0
            except Exception:
                self.ok = False

    def _build_row(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        traffic = inputs.get("monthly_web_traffic", []) or []
        avg_traffic = float(np.mean(traffic)) if traffic else 0.0
        growth = 0.0
        if isinstance(traffic, list) and len(traffic) > 1 and traffic[0] > 0:
            growth = (traffic[-1] - traffic[0]) / (traffic[0] + 1e-6)

        num_inv = _safe_float(inputs.get("num_investors", 0))
        team = _safe_float(inputs.get("team_size", 1))
        total_funding = _safe_float(inputs.get("total_funding_usd", 0))
        funding_per_investor = total_funding / (num_inv + 1e-6) if num_inv > 0 else 0.0
        funding_per_employee = total_funding / (team + 1e-6) if team > 0 else 0.0

        base = {
            "age": _safe_float(inputs.get("age", 0)),
            "arr": _safe_float(inputs.get("arr", 0)),
            "burn": _safe_float(inputs.get("burn", 0)),
            "cash": _safe_float(inputs.get("cash", 0)),
            "ltv_cac_ratio": _safe_float(inputs.get("ltv_cac_ratio", 1.0)),
            "gross_margin_pct": _safe_float(inputs.get("gross_margin_pct", 60.0)),
            "monthly_churn_pct": _safe_float(inputs.get("monthly_churn_pct", 2.0)),
            "team_size": team,
            "num_investors": num_inv,
            "investor_quality_score": _safe_float(inputs.get("investor_quality_score", 5.0)),
            "product_stage_score": _safe_float(inputs.get("product_stage_score", 5.0)),
            "moat_score": _safe_float(inputs.get("moat_score", 5.0)),
            "team_score": _safe_float(inputs.get("team_score", 5.0)),
            "avg_web_traffic": avg_traffic,
            "web_traffic_growth": growth,
            "stage": inputs.get("stage", ""),
            "sector": inputs.get("sector", ""),
            "focus_area": inputs.get("focus_area", ""),
        }
        row = {f: base.get(f, 0.0) for f in (self.feature_list or base.keys())}
        return pd.DataFrame([row])

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ok:
            # Heuristic fallback
            runway = _safe_float(inputs.get("cash", 0)) / max(1.0, _safe_float(inputs.get("burn", 1.0)))
            arr = _safe_float(inputs.get("arr", 0.0))
            investor_q = _safe_float(inputs.get("investor_quality_score", 5.0))
            base = 0.2 + 0.1*np.tanh(arr/1e7) + 0.1*np.tanh(runway/6.0) + 0.05*np.tanh((investor_q-5)/3.0)
            return {
                "round_probability_12m": float(max(0.01, min(0.95, base))),
                "round_probability_6m": float(min(0.9, (1 - (1 - base) ** 0.5))),
                "predicted_valuation_usd": None,
                "feature_importances": []
            }

        X = self._build_row(inputs)
        try:
            proba = float(self.clf.predict_proba(X)[0][1])
        except Exception:
            proba = 0.5
        p6 = 1 - (1 - proba) ** 0.5

        valuation = None
        if self.reg is not None:
            try:
                valuation = float(self.reg.predict(X)[0])
            except Exception:
                valuation = None

        importances = []
        try:
            est = getattr(self.clf, "named_steps", {}).get("model", None) or getattr(self.clf, "steps", [[None, None]])[-1][1]
            if hasattr(est, "coef_"):
                import numpy as np
                coefs = np.abs(est.coef_[0]) if est.coef_.ndim > 1 else np.abs(est.coef_)
                for name, val in zip(self.feature_list, coefs):
                    importances.append({"feature": name, "importance": float(val)})
            elif hasattr(est, "feature_importances_"):
                for name, val in zip(self.feature_list, est.feature_importances_):
                    importances.append({"feature": name, "importance": float(val)})
        except Exception:
            pass
        importances = sorted(importances, key=lambda x: x["importance"], reverse=True)[:20]

        return {
            "round_probability_12m": proba,
            "round_probability_6m": float(p6),
            "predicted_valuation_usd": valuation,
            "feature_importances": importances,
            "meta": self.meta
        }
