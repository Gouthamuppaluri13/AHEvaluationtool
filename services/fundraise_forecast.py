from typing import Dict, Any

from models.round_models import RoundLikelihoodModel, TimeToNextFinancingModel

class FundraiseForecastService:
    def __init__(self, csv_candidates = ("next_gen_deal_data.csv", "default_training_data.csv")):
        self.round_model = RoundLikelihoodModel()
        self.time_model = TimeToNextFinancingModel()

        loaded = self.round_model.load()
        trained_round = False
        trained_time = False

        if not loaded:
            for path in csv_candidates:
                try:
                    trained_round = self.round_model.train_from_csv(path) or trained_round
                except Exception:
                    pass

        for path in csv_candidates:
            try:
                trained_time = self.time_model.train_from_csv(path) or trained_time
            except Exception:
                pass

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        p6 = self.round_model.predict_proba(inputs, horizon_months=6)
        p12 = self.round_model.predict_proba(inputs, horizon_months=12)
        ttnf = self.time_model.predict_months(inputs)
        return {
            "round_likelihood_6m": p6,
            "round_likelihood_12m": p12,
            "expected_time_to_next_round_months": ttnf
        }
