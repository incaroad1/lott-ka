from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .base_model import BaseLotteryModel


class RandomForestLotteryModel(BaseLotteryModel):
    """
    Többkimenetű RandomForest-alapú modell.
    A cél minden számhoz egy 0..1 közeli score becslése.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name="random_forest")

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.metadata.update({
            "model_type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })

    def _to_numpy_2d(self, arr) -> np.ndarray:
        x = np.asarray(arr, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _validate_y(self, y) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError("A y célmátrixnak 2 dimenziósnak kell lennie: (minta_db, szamter_db)")
        return y_arr

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).flatten()
        scores = np.clip(scores, 0.0, 1.0)

        s = scores.sum()
        if s > 0:
            return scores / s

        # Ha minden 0 lenne, adjunk egyenletes eloszlást
        if len(scores) == 0:
            return scores
        return np.ones_like(scores) / len(scores)

    def fit(self, X, y, **kwargs) -> None:
        X_arr = self._to_numpy_2d(X)
        y_arr = self._validate_y(y)

        self.model.fit(X_arr, y_arr)

        self.is_trained = True
        self.last_train_size = len(X_arr)

        # Egyszerű belső train score közelítés
        train_pred = self.model.predict(X_arr)
        mse = mean_squared_error(y_arr, train_pred)
        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("A RandomForest modell még nincs betanítva.")

        X_arr = self._to_numpy_2d(X)
        pred = self.model.predict(X_arr)

        # csak az első mintára dolgozunk
        if pred.ndim == 2:
            scores = pred[0]
        else:
            scores = pred

        return self._normalize_scores(scores)

    def evaluate(self, X_val, y_val, **kwargs) -> float:
        if not self.is_trained:
            return 0.0

        X_arr = self._to_numpy_2d(X_val)
        y_arr = self._validate_y(y_val)

        pred = self.model.predict(X_arr)
        mse = mean_squared_error(y_arr, pred)

        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))
        return self.last_performance_score

    def feature_importances(self) -> Optional[np.ndarray]:
        if not self.is_trained:
            return None

        if hasattr(self.model, "feature_importances_"):
            return np.asarray(self.model.feature_importances_, dtype=float)

        return None

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "is_trained": self.is_trained,
            "last_performance_score": self.last_performance_score,
            "last_train_size": self.last_train_size,
            "metadata": self.metadata,
        }