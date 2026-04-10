from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_squared_error

from .base_model import BaseLotteryModel


class QuantumInspiredLotteryModel(BaseLotteryModel):
    """
    Quantum-inspired lottómodell.

    Nem valódi kvantumfuttatás, hanem a kvantumos repo ötletének
    magyar lottóra adaptált, stabilan futó változata:

    - frekvencia score
    - recency score
    - számpár-együttjárás
    - diverzitási büntetés
    - egyszerű iteratív optimalizálás

    A kimenet minden számra egy normalizált score.
    """

    def __init__(
        self,
        recency_window: int = 24,
        freq_weight: float = 0.34,
        recency_weight: float = 0.26,
        pair_weight: float = 0.24,
        diversity_weight: float = 0.16,
        optimization_steps: int = 120,
        random_state: int = 42,
    ):
        super().__init__(name="quantum")

        self.recency_window = int(recency_window)
        self.freq_weight = float(freq_weight)
        self.recency_weight = float(recency_weight)
        self.pair_weight = float(pair_weight)
        self.diversity_weight = float(diversity_weight)
        self.optimization_steps = int(optimization_steps)
        self.random_state = int(random_state)

        self.rng = np.random.default_rng(self.random_state)

        self.target_dim: int | None = None
        self.numbers_to_pick: int | None = None

        self.frequency_scores: np.ndarray | None = None
        self.recency_scores: np.ndarray | None = None
        self.pair_matrix: np.ndarray | None = None
        self.preselection_score_vector: np.ndarray | None = None
        self.base_score_vector: np.ndarray | None = None

        self.metadata.update({
            "model_type": "QuantumInspiredHeuristic",
            "recency_window": self.recency_window,
            "freq_weight": self.freq_weight,
            "recency_weight": self.recency_weight,
            "pair_weight": self.pair_weight,
            "diversity_weight": self.diversity_weight,
            "optimization_steps": self.optimization_steps,
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
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if len(scores) == 0:
            return scores

        min_val = float(np.min(scores))
        if min_val < 0:
            scores = scores - min_val

        scores = np.clip(scores, 0.0, None)
        total = float(np.sum(scores))
        if total > 0:
            return scores / total

        return np.ones_like(scores, dtype=float) / len(scores)

    def _build_frequency_scores(self, y: np.ndarray) -> np.ndarray:
        freq = np.mean(y, axis=0)
        return self._normalize_scores(freq)

    def _build_recency_scores(self, y: np.ndarray) -> np.ndarray:
        window = min(len(y), self.recency_window)
        recent = y[-window:]
        weights = np.linspace(0.35, 1.0, window, dtype=float)
        weighted = recent * weights[:, None]
        score = np.sum(weighted, axis=0)
        return self._normalize_scores(score)

    def _build_pair_matrix(self, y: np.ndarray) -> np.ndarray:
        target_dim = y.shape[1]
        pair = np.zeros((target_dim, target_dim), dtype=float)

        for row in y:
            active = np.where(row > 0.5)[0]
            for i in active:
                for j in active:
                    if i != j:
                        pair[i, j] += 1.0

        if np.max(pair) > 0:
            pair = pair / float(np.max(pair))

        np.fill_diagonal(pair, 0.0)
        return pair

    def _estimate_numbers_to_pick(self, y: np.ndarray) -> int:
        if len(y) == 0:
            return 5
        val = int(round(float(np.mean(np.sum(y, axis=1)))))
        return max(1, val)

    def _build_preselection_scores(self) -> np.ndarray:
        assert self.frequency_scores is not None
        assert self.recency_scores is not None

        base = (
            self.freq_weight * self.frequency_scores
            + self.recency_weight * self.recency_scores
        )
        return self._normalize_scores(base)

    def _candidate_energy(self, candidate_idx: np.ndarray, base_vector: np.ndarray) -> float:
        assert self.pair_matrix is not None

        base_gain = float(np.sum(base_vector[candidate_idx]))

        pair_gain = 0.0
        pair_penalty = 0.0
        for i in candidate_idx:
            for j in candidate_idx:
                if i >= j:
                    continue
                v = float(self.pair_matrix[i, j])
                if v > 0:
                    pair_gain += v
                    pair_penalty += v * v

        diversity_term = pair_gain - 0.65 * pair_penalty
        return base_gain + self.pair_weight * diversity_term

    def _optimize_candidate_set(self, base_vector: np.ndarray) -> np.ndarray:
        assert self.numbers_to_pick is not None

        target_dim = len(base_vector)
        start_idx = np.argsort(base_vector)[::-1][:self.numbers_to_pick]
        best = np.array(sorted(start_idx.tolist()), dtype=int)
        best_score = self._candidate_energy(best, base_vector)

        for _ in range(self.optimization_steps):
            candidate = best.copy()

            out_pos = int(self.rng.integers(0, len(candidate)))
            current_set = set(candidate.tolist())
            available = np.array([i for i in range(target_dim) if i not in current_set], dtype=int)
            if len(available) == 0:
                break

            in_idx = int(self.rng.choice(available))
            candidate[out_pos] = in_idx
            candidate = np.array(sorted(np.unique(candidate).tolist()), dtype=int)

            while len(candidate) < self.numbers_to_pick:
                extra = int(self.rng.integers(0, target_dim))
                if extra not in set(candidate.tolist()):
                    candidate = np.array(sorted(candidate.tolist() + [extra]), dtype=int)

            if len(candidate) > self.numbers_to_pick:
                keep = np.argsort(base_vector[candidate])[::-1][:self.numbers_to_pick]
                candidate = np.array(sorted(candidate[keep].tolist()), dtype=int)

            score = self._candidate_energy(candidate, base_vector)
            if score > best_score:
                best = candidate
                best_score = score

        return best

    def _build_final_score_vector(self) -> np.ndarray:
        assert self.target_dim is not None
        assert self.numbers_to_pick is not None
        assert self.pair_matrix is not None

        preselection = self._build_preselection_scores()
        self.preselection_score_vector = preselection.copy()

        candidate = self._optimize_candidate_set(preselection)

        pair_boost = np.zeros(self.target_dim, dtype=float)
        for idx in candidate:
            pair_boost += self.pair_matrix[idx]

        pair_boost = self._normalize_scores(pair_boost)

        final = (
            0.72 * preselection
            + 0.28 * pair_boost
        )

        selection_bonus = np.zeros(self.target_dim, dtype=float)
        selection_bonus[candidate] = 1.0
        selection_bonus = self._normalize_scores(selection_bonus)

        final = final + self.diversity_weight * selection_bonus
        return self._normalize_scores(final)

    def fit(self, X, y, **kwargs) -> None:
        _ = self._to_numpy_2d(X)
        y_arr = self._validate_y(y)

        self.target_dim = int(y_arr.shape[1])
        self.numbers_to_pick = self._estimate_numbers_to_pick(y_arr)

        self.frequency_scores = self._build_frequency_scores(y_arr)
        self.recency_scores = self._build_recency_scores(y_arr)
        self.pair_matrix = self._build_pair_matrix(y_arr)
        self.base_score_vector = self._build_final_score_vector()

        self.is_trained = True
        self.last_train_size = len(y_arr)

        pred = np.tile(self.base_score_vector, (len(y_arr), 1))
        mse = mean_squared_error(y_arr, pred)
        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        if not self.is_trained or self.base_score_vector is None:
            raise RuntimeError("A QuantumInspired modell még nincs betanítva.")

        X_arr = self._to_numpy_2d(X)
        if len(X_arr) <= 1:
            return self.base_score_vector.copy()

        return np.tile(self.base_score_vector, (len(X_arr), 1))

    def evaluate(self, X_val, y_val, **kwargs) -> float:
        if not self.is_trained or self.base_score_vector is None:
            return 0.0

        X_arr = self._to_numpy_2d(X_val)
        y_arr = self._validate_y(y_val)

        pred = np.tile(self.base_score_vector, (len(X_arr), 1))
        mse = mean_squared_error(y_arr, pred)
        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))
        return self.last_performance_score

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "is_trained": self.is_trained,
            "last_performance_score": self.last_performance_score,
            "last_train_size": self.last_train_size,
            "target_dim": self.target_dim,
            "numbers_to_pick": self.numbers_to_pick,
            "metadata": self.metadata,
        }