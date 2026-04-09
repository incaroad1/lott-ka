from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.builders.dataset_builder import DatasetBuilder
from src.importers.factory import get_importer
from src.models.lstm_model import LSTMLotteryModel
from src.models.random_forest_model import RandomForestLotteryModel
from src.models.xgboost_model import XGBoostLotteryModel
from src.services.lstm_pipeline import _build_latest_sequence, _build_sequences
from src.services.pipeline_utils import prediction_row_to_array, split_features_targets


@dataclass
class ModelManagerConfig:
    min_test_rows: int = 100
    sequence_length: int = 15
    min_history: int = 10
    auto_rf_threshold: int = 180
    auto_xgb_threshold: int = 420
    top_k_prediction: int = 12


class ModelManager:
    """
    Központi modellkezelő réteg.

    Cél:
    - egységes interfész RF / XGBoost / LSTM fölé
    - automatikus modellválasztás adatmennyiség alapján
    - egységes prediction / score-vector / meta visszaadás
    - compare / ensemble pipeline alapjának stabilizálása
    """

    SUPPORTED_MODES = {
        "auto",
        "random_forest",
        "xgboost",
        "lstm",
        "ensemble",
        "compare",
    }

    def __init__(self, config: ModelManagerConfig | None = None) -> None:
        self.config = config or ModelManagerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        csv_path: str | Path,
        jatek: str,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """
        Egységes belépési pont.

        mode:
        - auto
        - random_forest
        - xgboost
        - lstm
        - ensemble
        - compare
        """
        mode = (mode or "auto").strip().lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Nem támogatott mód: {mode}")

        prepared = self._prepare_dataset(csv_path=csv_path, jatek=jatek)

        if mode == "auto":
            selected_mode = self._select_mode(prepared["flat_train_rows"])
            single_result = self._run_single_model(prepared, selected_mode)
            return {
                "strategy": "auto",
                "selected_model": selected_mode,
                "imported_record_count": prepared["imported_record_count"],
                "error_count": prepared["error_count"],
                "prediction": single_result["prediction"],
                "training_result": single_result["training_result"],
                "meta": {
                    **single_result["meta"],
                    "auto_selected_model": selected_mode,
                },
            }

        if mode == "compare":
            compare_result = self.compare_models(csv_path=csv_path, jatek=jatek)
            return {
                "strategy": "compare",
                "selected_model": compare_result["best_model"],
                **compare_result,
            }

        if mode == "ensemble":
            ensemble_result = self.build_ensemble_prediction(prepared)
            return {
                "strategy": "ensemble",
                "selected_model": "ensemble_avg",
                "imported_record_count": prepared["imported_record_count"],
                "error_count": prepared["error_count"],
                "prediction": ensemble_result["prediction"],
                "training_result": ensemble_result["training_result"],
                "meta": ensemble_result["meta"],
                "models": ensemble_result["models"],
            }

        single_result = self._run_single_model(prepared, mode)
        return {
            "strategy": "single_model",
            "selected_model": mode,
            "imported_record_count": prepared["imported_record_count"],
            "error_count": prepared["error_count"],
            "prediction": single_result["prediction"],
            "training_result": single_result["training_result"],
            "meta": single_result["meta"],
        }

    def compare_models(
        self,
        csv_path: str | Path,
        jatek: str,
    ) -> dict[str, Any]:
        prepared = self._prepare_dataset(csv_path=csv_path, jatek=jatek)

        rf_result = self._run_single_model(prepared, "random_forest")
        xgb_result = self._run_single_model(prepared, "xgboost")
        lstm_result = self._run_single_model(prepared, "lstm")
        ensemble_result = self.build_ensemble_prediction(prepared)

        scoreboard = [
            {"modell": "random_forest", **rf_result["evaluation"]},
            {"modell": "xgboost", **xgb_result["evaluation"]},
            {"modell": "lstm", **lstm_result["evaluation"]},
            {"modell": "ensemble_avg", **ensemble_result["evaluation"]},
        ]
        scoreboard.sort(
            key=lambda row: (
                row["avg_hit_at_5"],
                row["avg_hit_at_10"],
                row["any_hit_rate_at_5"],
                row["exact_match_rate_at_5"],
            ),
            reverse=True,
        )

        return {
            "imported_record_count": prepared["imported_record_count"],
            "error_count": prepared["error_count"],
            "scoreboard": scoreboard,
            "best_model": scoreboard[0]["modell"] if scoreboard else None,
            "models": {
                "random_forest": {
                    "training_result": rf_result["training_result"],
                    "prediction": rf_result["prediction"],
                    "meta": rf_result["meta"],
                },
                "xgboost": {
                    "training_result": xgb_result["training_result"],
                    "prediction": xgb_result["prediction"],
                    "meta": xgb_result["meta"],
                },
                "lstm": {
                    "training_result": lstm_result["training_result"],
                    "prediction": lstm_result["prediction"],
                    "meta": lstm_result["meta"],
                },
                "ensemble_avg": {
                    "training_result": ensemble_result["training_result"],
                    "prediction": ensemble_result["prediction"],
                    "meta": ensemble_result["meta"],
                },
            },
        }

    def build_ensemble_prediction(self, prepared: dict[str, Any]) -> dict[str, Any]:
        rf_result = self._run_single_model(prepared, "random_forest")
        xgb_result = self._run_single_model(prepared, "xgboost")
        lstm_result = self._run_single_model(prepared, "lstm")

        rf_latest = np.asarray(rf_result["latest_vector"], dtype=float)
        xgb_latest = np.asarray(xgb_result["latest_vector"], dtype=float)
        lstm_latest = np.asarray(lstm_result["latest_vector"], dtype=float)

        ensemble_latest = (
            self._normalize_scores(rf_latest)
            + self._normalize_scores(xgb_latest)
            + self._normalize_scores(lstm_latest)
        ) / 3.0

        rf_test = np.asarray(rf_result["test_vectors"], dtype=float)
        xgb_test = np.asarray(xgb_result["test_vectors"], dtype=float)
        lstm_test = np.asarray(lstm_result["test_vectors"], dtype=float)

        min_test_len = min(len(rf_test), len(xgb_test), len(lstm_test))
        if min_test_len <= 0:
            raise ValueError("Nincs elég tesztvektor az ensemble értékeléshez.")

        rf_test = rf_test[-min_test_len:]
        xgb_test = xgb_test[-min_test_len:]
        lstm_test = lstm_test[-min_test_len:]

        y_test = np.asarray(lstm_result["y_test"], dtype=int)[-min_test_len:]

        ensemble_test = (
            np.apply_along_axis(self._normalize_scores, 1, rf_test)
            + np.apply_along_axis(self._normalize_scores, 1, xgb_test)
            + np.apply_along_axis(self._normalize_scores, 1, lstm_test)
        ) / 3.0

        top_k = min(self.config.top_k_prediction, prepared["target_count"])
        prediction = self._vector_to_prediction_result(
            jatek=prepared["jatek"],
            modell="ensemble_avg",
            proba=ensemble_latest,
            feature_count=prepared["feature_count"],
            target_count=prepared["target_count"],
            top_k=top_k,
        )

        evaluation = self._evaluate_prediction_matrix(
            proba=ensemble_test,
            y_test=y_test,
            top_k_primary=min(5, prepared["numbers_to_pick"]),
            top_k_secondary=min(10, prepared["target_count"]),
        )

        training_result = {
            "jatek": prepared["jatek"],
            "modell": "ensemble_avg",
            "train_rows": int(prepared["flat_train_rows"] - self.config.min_test_rows),
            "test_rows": int(len(y_test)),
            "feature_count": int(prepared["feature_count"]),
            "target_count": int(prepared["target_count"]),
            **evaluation,
        }

        return {
            "training_result": training_result,
            "prediction": prediction,
            "evaluation": evaluation,
            "latest_vector": ensemble_latest,
            "test_vectors": ensemble_test,
            "y_test": y_test,
            "models": {
                "random_forest": rf_result["prediction"],
                "xgboost": xgb_result["prediction"],
                "lstm": lstm_result["prediction"],
            },
            "meta": {
                "strategy": "equal_weight_avg",
                "components": ["random_forest", "xgboost", "lstm"],
                "sample_count": prepared["flat_train_rows"],
            },
        }

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _prepare_dataset(self, csv_path: str | Path, jatek: str) -> dict[str, Any]:
        importer = get_importer("skandi" if jatek.startswith("skandi") else jatek)
        imported = importer.import_file(csv_path)

        records = [r for r in imported["records"] if r["jatek"] == jatek]
        if not records:
            raise ValueError(f"Nincs feldolgozható rekord ehhez a játékhoz: {jatek}")

        builder = DatasetBuilder(min_history=self.config.min_history)
        training_rows = builder.build_training_rows(records)
        prediction_row = builder.build_prediction_row(records, jatek)

        X_flat, y_flat, feature_cols, target_cols = split_features_targets(training_rows)
        X_latest_flat = prediction_row_to_array(prediction_row, feature_cols)

        if len(X_flat) <= self.config.min_test_rows:
            raise ValueError("Túl kevés tanító sor a stabil teszteléshez.")

        X_seq, y_seq = _build_sequences(
            X_flat,
            y_flat,
            sequence_length=self.config.sequence_length,
        )
        X_latest_seq = _build_latest_sequence(
            X_flat,
            X_latest_flat,
            sequence_length=self.config.sequence_length,
        )

        if len(X_seq) <= self.config.min_test_rows:
            raise ValueError("Túl kevés LSTM szekvencia a stabil teszteléshez.")

        flat_split_idx = len(X_flat) - self.config.min_test_rows
        seq_split_idx = len(X_seq) - self.config.min_test_rows

        numbers_to_pick = int(np.max(np.sum(y_flat, axis=1))) if len(y_flat) else 5
        numbers_to_pick = max(1, numbers_to_pick)

        return {
            "jatek": jatek,
            "imported_record_count": len(records),
            "error_count": len(imported["errors"]),
            "records": records,
            "training_rows": training_rows,
            "prediction_row": prediction_row,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "feature_count": int(X_flat.shape[1]),
            "target_count": int(y_flat.shape[1]),
            "numbers_to_pick": numbers_to_pick,
            "flat_train_rows": int(len(X_flat)),
            "seq_train_rows": int(len(X_seq)),
            "X_train_flat": X_flat[:flat_split_idx],
            "X_test_flat": X_flat[flat_split_idx:],
            "y_train_flat": y_flat[:flat_split_idx],
            "y_test_flat": y_flat[flat_split_idx:],
            "X_latest_flat": X_latest_flat,
            "X_train_seq": X_seq[:seq_split_idx],
            "X_test_seq": X_seq[seq_split_idx:],
            "y_train_seq": y_seq[:seq_split_idx],
            "y_test_seq": y_seq[seq_split_idx:],
            "X_latest_seq": X_latest_seq,
        }

    # ------------------------------------------------------------------
    # Single model execution
    # ------------------------------------------------------------------

    def _run_single_model(self, prepared: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "random_forest":
            model = RandomForestLotteryModel()
            X_train = prepared["X_train_flat"]
            y_train = prepared["y_train_flat"]
            X_test = prepared["X_test_flat"]
            y_test = prepared["y_test_flat"]
            X_latest = prepared["X_latest_flat"]
            input_kind = "flat"

        elif mode == "xgboost":
            model = XGBoostLotteryModel()
            X_train = prepared["X_train_flat"]
            y_train = prepared["y_train_flat"]
            X_test = prepared["X_test_flat"]
            y_test = prepared["y_test_flat"]
            X_latest = prepared["X_latest_flat"]
            input_kind = "flat"

        elif mode == "lstm":
            model = LSTMLotteryModel()
            X_train = prepared["X_train_seq"]
            y_train = prepared["y_train_seq"]
            X_test = prepared["X_test_seq"]
            y_test = prepared["y_test_seq"]
            X_latest = prepared["X_latest_seq"]
            input_kind = "sequence"

        else:
            raise ValueError(f"Ismeretlen single model mód: {mode}")

        model.fit(X_train, y_train, jatek=prepared["jatek"])

        test_vectors = self._predict_matrix(model, X_test)
        latest_vector = self._predict_single(model, X_latest)

        top_k = min(self.config.top_k_prediction, prepared["target_count"])
        prediction = self._vector_to_prediction_result(
            jatek=prepared["jatek"],
            modell=mode,
            proba=latest_vector,
            feature_count=prepared["feature_count"],
            target_count=prepared["target_count"],
            top_k=top_k,
        )

        evaluation = self._evaluate_prediction_matrix(
            proba=test_vectors,
            y_test=y_test,
            top_k_primary=min(5, prepared["numbers_to_pick"]),
            top_k_secondary=min(10, prepared["target_count"]),
        )

        training_result = {
            "jatek": prepared["jatek"],
            "modell": mode,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(prepared["feature_count"]),
            "target_count": int(prepared["target_count"]),
            "input_kind": input_kind,
            **evaluation,
        }

        return {
            "training_result": training_result,
            "prediction": prediction,
            "evaluation": evaluation,
            "latest_vector": latest_vector,
            "test_vectors": test_vectors,
            "y_test": y_test,
            "meta": {
                "sample_count": len(X_train),
                "input_kind": input_kind,
                "sequence_length": self.config.sequence_length if mode == "lstm" else None,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_mode(self, flat_train_rows: int) -> str:
        if flat_train_rows < self.config.auto_rf_threshold:
            return "random_forest"
        if flat_train_rows < self.config.auto_xgb_threshold:
            return "xgboost"
        return "lstm"

    def _predict_single(self, model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba_vector"):
            pred = model.predict_proba_vector(X)
            pred = np.asarray(pred, dtype=float)
            if pred.ndim == 2:
                return pred[0]
            return pred.flatten()

        pred = model.predict_proba(X)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 2:
            return pred[0]
        return pred.flatten()

    def _predict_matrix(self, model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba_vector"):
            pred = model.predict_proba_vector(X)
            pred = np.asarray(pred, dtype=float)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            return pred

        rows = []
        for row in np.asarray(X):
            if row.ndim == 0:
                row = np.asarray([row], dtype=float)

            if np.asarray(X).ndim == 3:
                row_input = np.asarray(row, dtype=float).reshape(1, row.shape[0], row.shape[1])
            else:
                row_input = np.asarray(row, dtype=float).reshape(1, -1)

            pred = model.predict_proba(row_input)
            pred = np.asarray(pred, dtype=float)
            if pred.ndim == 2:
                rows.append(pred[0])
            else:
                rows.append(pred.flatten())

        return np.asarray(rows, dtype=float)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).flatten()
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        min_val = float(np.min(scores)) if len(scores) else 0.0
        if min_val < 0:
            scores = scores - min_val

        scores = np.clip(scores, 0.0, None)

        total = float(np.sum(scores))
        if total > 0:
            return scores / total

        if len(scores) == 0:
            return scores

        return np.ones_like(scores, dtype=float) / len(scores)

    def _evaluate_prediction_matrix(
        self,
        proba: np.ndarray,
        y_test: np.ndarray,
        top_k_primary: int = 5,
        top_k_secondary: int = 10,
    ) -> dict[str, float]:
        proba = np.asarray(proba, dtype=float)
        y_test = np.asarray(y_test, dtype=int)

        if len(proba) == 0 or len(y_test) == 0:
            return {
                "avg_hit_at_5": 0.0,
                "avg_hit_at_10": 0.0,
                "any_hit_rate_at_5": 0.0,
                "exact_match_rate_at_5": 0.0,
            }

        hit_counts_k1: list[int] = []
        hit_counts_k2: list[int] = []
        any_hit_k1: list[int] = []
        exact_match_k1: list[int] = []

        for i in range(min(len(proba), len(y_test))):
            row_proba = self._normalize_scores(proba[i])
            true_numbers = set(np.where(y_test[i] == 1)[0] + 1)

            top_k1_idx = np.argsort(row_proba)[::-1][:top_k_primary]
            top_k2_idx = np.argsort(row_proba)[::-1][:top_k_secondary]

            pred_k1 = set(int(x) + 1 for x in top_k1_idx)
            pred_k2 = set(int(x) + 1 for x in top_k2_idx)

            hits_k1 = len(pred_k1 & true_numbers)
            hits_k2 = len(pred_k2 & true_numbers)

            hit_counts_k1.append(hits_k1)
            hit_counts_k2.append(hits_k2)
            any_hit_k1.append(1 if hits_k1 > 0 else 0)
            exact_match_k1.append(1 if pred_k1 == true_numbers else 0)

        return {
            "avg_hit_at_5": round(float(np.mean(hit_counts_k1)), 6),
            "avg_hit_at_10": round(float(np.mean(hit_counts_k2)), 6),
            "any_hit_rate_at_5": round(float(np.mean(any_hit_k1)), 6),
            "exact_match_rate_at_5": round(float(np.mean(exact_match_k1)), 6),
        }

    def _vector_to_prediction_result(
        self,
        jatek: str,
        modell: str,
        proba: np.ndarray,
        feature_count: int,
        target_count: int,
        top_k: int = 5,
    ) -> dict[str, Any]:
        normalized = self._normalize_scores(proba)
        top_idx = np.argsort(normalized)[::-1][:top_k]
        top_numbers = sorted(int(i) + 1 for i in top_idx)
        scores = {int(i) + 1: round(float(normalized[i]), 6) for i in top_idx}

        return {
            "jatek": jatek,
            "modell": modell,
            "top_szamok": top_numbers,
            "scoreok": scores,
            "score_vector": [float(x) for x in normalized],
            "feature_count": int(feature_count),
            "target_count": int(target_count),
        }