from __future__ import annotations

from typing import Dict, List, Optional

from .base_model import BaseLotteryModel, ModelPrediction


class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseLotteryModel] = {}

        # Itt tudod szabályozni, melyik játékhoz mely modellek érhetők el
        self.game_model_map: Dict[str, List[str]] = {
            "otos": ["random_forest", "xgboost", "lstm"],
            "hatos": ["random_forest", "xgboost", "lstm"],
            "skandi_gepi": ["random_forest", "xgboost"],
            "skandi_kezi": ["random_forest", "xgboost", "lstm"],
        }

    def register_model(self, key: str, model: BaseLotteryModel) -> None:
        self.models[key] = model

    def register_models(self, model_dict: Dict[str, BaseLotteryModel]) -> None:
        for key, model in model_dict.items():
            self.register_model(key, model)

    def get_available_models(self, game_type: str) -> List[BaseLotteryModel]:
        keys = self.game_model_map.get(game_type, [])
        return [
            self.models[key]
            for key in keys
            if key in self.models
        ]

    def get_trained_models(self, game_type: str) -> List[BaseLotteryModel]:
        return [
            model for model in self.get_available_models(game_type)
            if model.is_trained
        ]

    def choose_primary_model(
        self,
        game_type: str,
        sample_count: int
    ) -> Optional[BaseLotteryModel]:
        """
        Első verzió:
        - kevés adat -> RF
        - közepes adat -> XGBoost / RF
        - sok adat -> LSTM előny
        """
        trained = self.get_trained_models(game_type)
        if not trained:
            return None

        if sample_count < 80:
            preferred_order = ["random_forest", "xgboost", "lstm"]
        elif sample_count < 200:
            preferred_order = ["xgboost", "random_forest", "lstm"]
        else:
            preferred_order = ["lstm", "xgboost", "random_forest"]

        for preferred_name in preferred_order:
            for model in trained:
                if model.name == preferred_name:
                    return model

        return max(trained, key=lambda m: m.last_performance_score)

    def predict_with_model(
        self,
        model: BaseLotteryModel,
        X,
        game_type: str
    ) -> ModelPrediction:
        probabilities = model.predict_proba(X)
        return model.build_prediction(game_type=game_type, probabilities=probabilities)

    def predict_best(
        self,
        X,
        game_type: str,
        sample_count: int
    ) -> Optional[ModelPrediction]:
        model = self.choose_primary_model(game_type, sample_count)
        if model is None:
            return None
        return self.predict_with_model(model, X, game_type)

    def predict_all(
        self,
        X,
        game_type: str
    ) -> List[ModelPrediction]:
        results: List[ModelPrediction] = []

        for model in self.get_trained_models(game_type):
            try:
                pred = self.predict_with_model(model, X, game_type)
                results.append(pred)
            except Exception as e:
                print(f"[ModelManager] Predikciós hiba ({model.name}): {e}")

        return results

    def get_model_summary(self, game_type: str) -> List[dict]:
        summary = []
        for model in self.get_available_models(game_type):
            summary.append({
                "key": model.name,
                "trained": model.is_trained,
                "performance_score": model.last_performance_score,
                "train_size": model.last_train_size,
            })
        return summary