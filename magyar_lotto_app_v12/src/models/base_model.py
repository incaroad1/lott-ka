from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelPrediction:
    model_name: str
    game_type: str
    probabilities: np.ndarray
    confidence: float
    performance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLotteryModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_trained: bool = False
        self.last_performance_score: float = 0.5
        self.last_train_size: int = 0
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X, y, **kwargs) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X, **kwargs) -> np.ndarray:
        pass

    def evaluate(self, X_val, y_val, **kwargs) -> float:
        """
        Opcionális felüldefiniálás.
        Ha nincs külön validációs logikád, maradhat az alap.
        """
        return self.last_performance_score

    def get_confidence(self, probabilities: np.ndarray) -> float:
        """
        Egyszerű első confidence:
        minél jobban kiemelkednek a top valószínűségek,
        annál nagyobb a confidence.
        """
        if probabilities is None:
            return 0.0

        p = np.asarray(probabilities, dtype=float).flatten()
        if len(p) == 0:
            return 0.0

        total = np.sum(p)
        if total > 0:
            p = p / total

        top_k = min(5, len(p))
        top_mean = float(np.mean(np.sort(p)[-top_k:]))
        overall_mean = float(np.mean(p))

        conf = top_mean - overall_mean
        return max(0.0, min(1.0, conf * 10.0))

    def build_prediction(
        self,
        game_type: str,
        probabilities: np.ndarray,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> ModelPrediction:
        metadata = dict(self.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)

        return ModelPrediction(
            model_name=self.name,
            game_type=game_type,
            probabilities=np.asarray(probabilities, dtype=float),
            confidence=self.get_confidence(probabilities),
            performance_score=float(self.last_performance_score),
            metadata=metadata
        )