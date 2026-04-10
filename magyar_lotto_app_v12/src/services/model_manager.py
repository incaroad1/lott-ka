from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.builders.dataset_builder import DatasetBuilder
from src.importers.factory import get_importer
from src.models.quantum_inspired_model import QuantumInspiredLotteryModel
from src.models.random_forest_model import RandomForestLotteryModel
from src.models.xgboost_model import XGBoostLotteryModel
from src.services.pipeline_utils import prediction_row_to_array, split_features_targets

try:
    from src.models.lstm_model import LSTMLotteryModel
    _LSTM_IMPORT_ERROR: Exception | None = None
except Exception as e:
    LSTMLotteryModel = None
    _LSTM_IMPORT_ERROR = e


@dataclass
class ModelManagerConfig:
    min_test_rows: int = 100
    sequence_length: int = 15
    min_history: int = 10
    auto_rf_threshold: int = 180
    auto_xgb_threshold: int = 420
    top_k_prediction: int = 12

    recency_window: int = 12
    recency_strength: float = 0.22
    last_draw_boost_strength: float = 0.12
    hot_strength: float = 0.08
    cold_strength: float = 0.04


class ModelManager:
    SUPPORTED_MODES = {
        "auto",
        "random_forest",
        "xgboost",
        "lstm",
        "quantum",
        "ensemble",
        "compare",
    }

    def __init__(self, config: ModelManagerConfig | None = None) -> None:
        self.config = config or ModelManagerConfig()

    def predict(
        self,
        csv_path: str | Path,
        jatek: str,
        mode: str = "auto",
    ) -> dict[str, Any]:
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
                    "lstm_available": self._is_lstm_available(),
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
                "selected_model": "ensemble_smart",
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

        model_results: dict[str, dict[str, Any]] = {}
        lstm_runtime_error: str | None = None
        lstm_used = False

        rf_result = self._run_single_model(prepared, "random_forest")
        model_results["random_forest"] = rf_result

        xgb_result = self._run_single_model(prepared, "xgboost")
        model_results["xgboost"] = xgb_result

        quantum_result = self._run_single_model(prepared, "quantum")
        model_results["quantum"] = quantum_result

        if self._is_lstm_available():
            try:
                lstm_result = self._run_single_model(prepared, "lstm")
                model_results["lstm"] = lstm_result
                lstm_used = True
            except Exception as e:
                lstm_runtime_error = str(e)
                model_results["lstm_unavailable"] = {"error": lstm_runtime_error}
        else:
            lstm_runtime_error = self._get_lstm_error_message()
            model_results["lstm_unavailable"] = {"error": lstm_runtime_error}

        ensemble_result = self.build_ensemble_prediction(prepared, precomputed=model_results)

        scoreboard = [
            {"modell": "random_forest", **rf_result["evaluation"]},
            {"modell": "xgboost", **xgb_result["evaluation"]},
            {"modell": "quantum", **quantum_result["evaluation"]},
        ]

        if "lstm" in model_results:
            scoreboard.append({"modell": "lstm", **model_results["lstm"]["evaluation"]})

        scoreboard.append({"modell": "ensemble_smart", **ensemble_result["evaluation"]})

        scoreboard.sort(
            key=lambda row: (
                row["avg_hit_at_5"],
                row["avg_hit_at_10"],
                row["any_hit_rate_at_5"],
                row["exact_match_rate_at_5"],
            ),
            reverse=True,
        )

        models_payload = {
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
            "quantum": {
                "training_result": quantum_result["training_result"],
                "prediction": quantum_result["prediction"],
                "meta": quantum_result["meta"],
            },
            "ensemble_avg": {
                "training_result": ensemble_result["training_result"],
                "prediction": ensemble_result["prediction"],
                "meta": ensemble_result["meta"],
            },
        }

        if "lstm" in model_results:
            models_payload["lstm"] = {
                "training_result": model_results["lstm"]["training_result"],
                "prediction": model_results["lstm"]["prediction"],
                "meta": {
                    **model_results["lstm"]["meta"],
                    "available": True,
                },
            }
        else:
            models_payload["lstm"] = {
                "training_result": None,
                "prediction": None,
                "meta": {
                    "available": False,
                    "error": lstm_runtime_error or self._get_lstm_error_message(),
                },
            }

        return {
            "imported_record_count": prepared["imported_record_count"],
            "error_count": prepared["error_count"],
            "scoreboard": scoreboard,
            "best_model": scoreboard[0]["modell"] if scoreboard else None,
            "models": models_payload,
            "meta": {
                "lstm_available": lstm_used,
                "lstm_error": None if lstm_used else (lstm_runtime_error or self._get_lstm_error_message()),
            },
        }

    def build_ensemble_prediction(
        self,
        prepared: dict[str, Any],
        precomputed: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        available_results: dict[str, dict[str, Any]] = {}

        if precomputed and "random_forest" in precomputed:
            available_results["random_forest"] = precomputed["random_forest"]
        else:
            available_results["random_forest"] = self._run_single_model(prepared, "random_forest")

        if precomputed and "xgboost" in precomputed:
            available_results["xgboost"] = precomputed["xgboost"]
        else:
            available_results["xgboost"] = self._run_single_model(prepared, "xgboost")

        if precomputed and "quantum" in precomputed:
            available_results["quantum"] = precomputed["quantum"]
        else:
            available_results["quantum"] = self._run_single_model(prepared, "quantum")

        if precomputed and "lstm" in precomputed:
            available_results["lstm"] = precomputed["lstm"]
        elif self._is_lstm_available():
            try:
                available_results["lstm"] = self._run_single_model(prepared, "lstm")
            except Exception:
                pass

        component_names = list(available_results.keys())
        if len(component_names) < 2:
            raise ValueError("Az ensemble futtatásához legalább 2 elérhető modell kell.")

        component_metrics = self._compute_model_weight_metrics(available_results, prepared)
        model_weights = self._normalize_weight_dict(
            {name: component_metrics[name]["raw_weight"] for name in component_names}
        )

        latest_vectors_raw = []
        test_vectors = []
        y_test_candidates = []

        for model_name in component_names:
            result = available_results[model_name]
            weight = model_weights[model_name]

            latest_vec_raw = self._normalize_scores(np.asarray(result["latest_vector_raw"], dtype=float))
            test_vec = np.apply_along_axis(
                self._normalize_scores,
                1,
                np.asarray(result["test_vectors"], dtype=float),
            )

            latest_vectors_raw.append(weight * latest_vec_raw)
            test_vectors.append((model_name, weight, test_vec))
            y_test_candidates.append(np.asarray(result["y_test"], dtype=int))

        ensemble_latest_raw = np.sum(np.asarray(latest_vectors_raw, dtype=float), axis=0)
        ensemble_latest_raw = self._normalize_scores(ensemble_latest_raw)

        ensemble_latest_adjusted, recency_meta = self._apply_recency_bias(
            ensemble_latest_raw,
            prepared["y_train_flat"],
        )

        min_test_len = min(len(v[2]) for v in test_vectors)
        if min_test_len <= 0:
            raise ValueError("Nincs elég tesztvektor az ensemble értékeléshez.")

        weighted_test_rows = []
        for _name, weight, vec in test_vectors:
            weighted_test_rows.append(weight * vec[-min_test_len:])

        ensemble_test = np.sum(np.asarray(weighted_test_rows, dtype=float), axis=0)
        y_test = y_test_candidates[0][-min_test_len:]

        top_k = min(self.config.top_k_prediction, prepared["target_count"])
        prediction = self._vector_to_prediction_result(
            jatek=prepared["jatek"],
            modell="ensemble_smart",
            proba=ensemble_latest_adjusted,
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
            "modell": "ensemble_smart",
            "train_rows": int(prepared["flat_train_rows"] - self.config.min_test_rows),
            "test_rows": int(len(y_test)),
            "feature_count": int(prepared["feature_count"]),
            "target_count": int(prepared["target_count"]),
            "component_count": len(component_names),
            **evaluation,
        }

        return {
            "training_result": training_result,
            "prediction": prediction,
            "evaluation": evaluation,
            "latest_vector": ensemble_latest_adjusted,
            "latest_vector_raw": ensemble_latest_raw,
            "test_vectors": ensemble_test,
            "y_test": y_test,
            "models": {name: available_results[name]["prediction"] for name in component_names},
            "meta": {
                "strategy": "smart_weighted_ensemble",
                "components": component_names,
                "sample_count": prepared["flat_train_rows"],
                "lstm_available": "lstm" in component_names,
                "weights": {k: round(float(v), 6) for k, v in model_weights.items()},
                "weight_metrics": {
                    name: {
                        "performance": round(float(component_metrics[name]["performance"]), 6),
                        "confidence": round(float(component_metrics[name]["confidence"]), 6),
                        "consistency_bonus": round(float(component_metrics[name]["consistency_bonus"]), 6),
                        "diversity_factor": round(float(component_metrics[name]["diversity_factor"]), 6),
                        "raw_weight": round(float(component_metrics[name]["raw_weight"]), 6),
                    }
                    for name in component_names
                },
                "recency": recency_meta,
            },
        }

    def _compute_model_weight_metrics(
        self,
        available_results: dict[str, dict[str, Any]],
        prepared: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        names = list(available_results.keys())
        latest_top_sets_5: dict[str, set[int]] = {}
        latest_top_sets_12: dict[str, set[int]] = {}
        latest_vectors: dict[str, np.ndarray] = {}
        metrics: dict[str, dict[str, float]] = {}

        for name in names:
            result = available_results[name]
            latest_vec = self._normalize_scores(np.asarray(result["latest_vector_raw"], dtype=float))
            eval_data = result["evaluation"]

            performance = (
                0.55 * float(eval_data["avg_hit_at_5"]) / max(1.0, float(prepared["numbers_to_pick"]))
                + 0.20 * float(eval_data["avg_hit_at_10"]) / max(1.0, float(min(10, prepared["target_count"])))
                + 0.20 * float(eval_data["any_hit_rate_at_5"])
                + 0.05 * float(eval_data["exact_match_rate_at_5"])
            )

            confidence = self._prediction_confidence(
                latest_vec,
                top_k=min(5, prepared["numbers_to_pick"]),
            )

            consistency_bonus = self._prediction_consistency(
                np.asarray(result["test_vectors"], dtype=float),
                np.asarray(result["y_test"], dtype=int),
                top_k=min(5, prepared["numbers_to_pick"]),
            )

            latest_top_sets_5[name] = self._top_number_set(
                latest_vec,
                top_k=min(5, prepared["numbers_to_pick"]),
            )
            latest_top_sets_12[name] = self._top_number_set(
                latest_vec,
                top_k=min(12, prepared["target_count"]),
            )
            latest_vectors[name] = latest_vec

            metrics[name] = {
                "performance": float(np.clip(performance, 0.0, 1.0)),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "consistency_bonus": float(np.clip(consistency_bonus, 0.0, 1.0)),
                "diversity_factor": 0.0,
                "raw_weight": 0.0,
            }

        for name in names:
            overlaps_5 = []
            overlaps_12 = []
            corr_penalties = []

            for other in names:
                if other == name:
                    continue

                set5_a = latest_top_sets_5[name]
                set5_b = latest_top_sets_5[other]
                denom5 = max(1, len(set5_a | set5_b))
                overlaps_5.append(len(set5_a & set5_b) / denom5)

                set12_a = latest_top_sets_12[name]
                set12_b = latest_top_sets_12[other]
                denom12 = max(1, len(set12_a | set12_b))
                overlaps_12.append(len(set12_a & set12_b) / denom12)

                vec_a = latest_vectors[name]
                vec_b = latest_vectors[other]
                if np.std(vec_a) < 1e-12 or np.std(vec_b) < 1e-12:
                    corr = 1.0
                else:
                    corr = float(np.corrcoef(vec_a, vec_b)[0, 1])
                    if np.isnan(corr):
                        corr = 1.0
                corr_penalties.append((corr + 1.0) / 2.0)

            avg_overlap_5 = float(np.mean(overlaps_5)) if overlaps_5 else 0.0
            avg_overlap_12 = float(np.mean(overlaps_12)) if overlaps_12 else 0.0
            avg_corr = float(np.mean(corr_penalties)) if corr_penalties else 0.0

            diversity_factor = (
                1.0
                - 0.45 * avg_overlap_5
                - 0.35 * avg_overlap_12
                - 0.20 * avg_corr
            )
            metrics[name]["diversity_factor"] = float(np.clip(diversity_factor, 0.0, 1.0))

        for name in names:
            m = metrics[name]
            raw_weight = (
                0.52 * m["performance"]
                + 0.23 * m["confidence"]
                + 0.15 * m["consistency_bonus"]
                + 0.10 * m["diversity_factor"]
            )
            metrics[name]["raw_weight"] = float(max(raw_weight, 1e-9) ** 2.4)

        return metrics

    def _prediction_confidence(self, vec: np.ndarray, top_k: int) -> float:
        norm = self._normalize_scores(vec)
        if len(norm) == 0:
            return 0.0

        top_idx = np.argsort(norm)[::-1][:top_k]
        top_mass = float(np.sum(norm[top_idx]))
        top_mean = float(np.mean(norm[top_idx]))
        rest_idx = np.argsort(norm)[::-1][top_k:]
        rest_mean = float(np.mean(norm[rest_idx])) if len(rest_idx) else 0.0
        margin = top_mean - rest_mean

        score = 0.65 * top_mass + 0.35 * min(1.0, margin * 20.0)
        return float(np.clip(score, 0.0, 1.0))

    def _prediction_consistency(self, matrix: np.ndarray, y_test: np.ndarray, top_k: int) -> float:
        mat = np.asarray(matrix, dtype=float)
        y_arr = np.asarray(y_test, dtype=int)

        n = min(len(mat), len(y_arr))
        if n < 5:
            return 0.35

        mat = mat[-25:] if n >= 25 else mat[:n]
        y_arr = y_arr[-len(mat):]

        hit_scores = []
        any_scores = []
        set_change_scores = []

        prev_pred_set: set[int] | None = None

        for i in range(len(mat)):
            vec = self._normalize_scores(mat[i])
            true_numbers = set(np.where(y_arr[i] == 1)[0] + 1)
            pred_set = self._top_number_set(vec, top_k=top_k)

            hits = len(pred_set & true_numbers)
            hit_scores.append(hits / max(1, top_k))
            any_scores.append(1.0 if hits > 0 else 0.0)

            if prev_pred_set is not None:
                denom = max(1, len(prev_pred_set | pred_set))
                jacc = len(prev_pred_set & pred_set) / denom
                set_change_scores.append(jacc)

            prev_pred_set = pred_set

        mean_hit = float(np.mean(hit_scores))
        std_hit = float(np.std(hit_scores))
        mean_any = float(np.mean(any_scores))
        std_any = float(np.std(any_scores))

        if set_change_scores:
            mean_set_stability = float(np.mean(set_change_scores))
            std_set_stability = float(np.std(set_change_scores))
        else:
            mean_set_stability = 0.5
            std_set_stability = 0.0

        quality_part = 0.55 * mean_hit + 0.25 * mean_any + 0.20 * mean_set_stability
        volatility_penalty = 0.50 * std_hit + 0.25 * std_any + 0.25 * std_set_stability

        score = quality_part * (1.0 - min(0.90, volatility_penalty))
        return float(np.clip(score, 0.0, 1.0))

    def _apply_recency_bias(
        self,
        score_vector: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        base = self._normalize_scores(score_vector)
        y_arr = np.asarray(y_train, dtype=int)

        if len(y_arr) == 0:
            return base, {
                "applied": False,
                "reason": "empty_y_train",
            }

        target_count = y_arr.shape[1]
        window = min(self.config.recency_window, len(y_arr))
        recent = y_arr[-window:]

        recent_weights = np.linspace(0.25, 1.0, window, dtype=float)
        recency_profile = np.sum(recent * recent_weights[:, None], axis=0)
        recency_profile = self._normalize_scores(recency_profile)

        last_draw = y_arr[-1].astype(float)
        if np.sum(last_draw) > 0:
            last_draw = last_draw / np.sum(last_draw)

        historical_freq = np.mean(y_arr, axis=0)
        hot_profile = self._normalize_scores(historical_freq)

        cold_profile = 1.0 - hot_profile
        cold_profile = self._normalize_scores(cold_profile)

        adjusted = (
            (1.0 - self.config.recency_strength) * base
            + self.config.recency_strength * recency_profile
            + self.config.last_draw_boost_strength * last_draw
            + self.config.hot_strength * hot_profile
            + self.config.cold_strength * cold_profile
        )
        adjusted = self._normalize_scores(adjusted)

        top_idx = np.argsort(adjusted)[::-1][: min(12, target_count)]

        return adjusted, {
            "applied": True,
            "window": window,
            "top_recency_numbers": [int(i) + 1 for i in np.argsort(recency_profile)[::-1][: min(8, target_count)]],
            "top_last_draw_numbers": [int(i) + 1 for i in np.where(y_arr[-1] == 1)[0]],
            "top_adjusted_numbers": [int(i) + 1 for i in top_idx],
            "recency_strength": self.config.recency_strength,
            "last_draw_boost_strength": self.config.last_draw_boost_strength,
            "hot_strength": self.config.hot_strength,
            "cold_strength": self.config.cold_strength,
        }

    def _top_number_set(self, vec: np.ndarray, top_k: int) -> set[int]:
        norm = self._normalize_scores(vec)
        idx = np.argsort(norm)[::-1][:top_k]
        return {int(i) + 1 for i in idx}

    def _normalize_weight_dict(self, weights: dict[str, float]) -> dict[str, float]:
        total = float(sum(max(0.0, v) for v in weights.values()))
        if total <= 0:
            uniform = 1.0 / max(1, len(weights))
            return {k: uniform for k in weights}
        return {k: float(max(0.0, v) / total) for k, v in weights.items()}

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

        X_seq, y_seq = self._build_sequences(
            X_flat,
            y_flat,
            sequence_length=self.config.sequence_length,
        )
        X_latest_seq = self._build_latest_sequence(
            X_flat,
            X_latest_flat,
            sequence_length=self.config.sequence_length,
        )

        flat_split_idx = len(X_flat) - self.config.min_test_rows
        seq_split_idx = len(X_seq) - self.config.min_test_rows if len(X_seq) > self.config.min_test_rows else None

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
            "X_train_seq": X_seq[:seq_split_idx] if seq_split_idx is not None else X_seq,
            "X_test_seq": X_seq[seq_split_idx:] if seq_split_idx is not None else np.empty((0, self.config.sequence_length, X_flat.shape[1])),
            "y_train_seq": y_seq[:seq_split_idx] if seq_split_idx is not None else y_seq,
            "y_test_seq": y_seq[seq_split_idx:] if seq_split_idx is not None else np.empty((0, y_flat.shape[1])),
            "X_latest_seq": X_latest_seq,
        }

    def _run_single_model(self, prepared: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "random_forest":
            model = RandomForestLotteryModel()
            X_train = prepared["X_train_flat"]
            y_train = prepared["y_train_flat"]
            X_test = prepared["X_test_flat"]
            y_test = prepared["y_test_flat"]
            X_latest = prepared["X_latest_flat"]
            recency_train = prepared["y_train_flat"]
            input_kind = "flat"

        elif mode == "xgboost":
            model = XGBoostLotteryModel()
            X_train = prepared["X_train_flat"]
            y_train = prepared["y_train_flat"]
            X_test = prepared["X_test_flat"]
            y_test = prepared["y_test_flat"]
            X_latest = prepared["X_latest_flat"]
            recency_train = prepared["y_train_flat"]
            input_kind = "flat"

        elif mode == "quantum":
            model = QuantumInspiredLotteryModel()
            X_train = prepared["X_train_flat"]
            y_train = prepared["y_train_flat"]
            X_test = prepared["X_test_flat"]
            y_test = prepared["y_test_flat"]
            X_latest = prepared["X_latest_flat"]
            recency_train = prepared["y_train_flat"]
            input_kind = "flat"

        elif mode == "lstm":
            if not self._is_lstm_available():
                raise ImportError(self._get_lstm_error_message())
            if len(prepared["X_test_seq"]) == 0:
                raise ValueError("Túl kevés LSTM szekvencia a stabil teszteléshez.")
            model = LSTMLotteryModel()
            X_train = prepared["X_train_seq"]
            y_train = prepared["y_train_seq"]
            X_test = prepared["X_test_seq"]
            y_test = prepared["y_test_seq"]
            X_latest = prepared["X_latest_seq"]
            recency_train = prepared["y_train_seq"]
            input_kind = "sequence"

        else:
            raise ValueError(f"Ismeretlen single model mód: {mode}")

        model.fit(X_train, y_train, jatek=prepared["jatek"])

        test_vectors = self._predict_matrix(model, X_test)
        latest_vector_raw = self._predict_single(model, X_latest)

        latest_vector_adjusted, recency_meta = self._apply_recency_bias(
            latest_vector_raw,
            recency_train,
        )

        top_k = min(self.config.top_k_prediction, prepared["target_count"])
        prediction = self._vector_to_prediction_result(
            jatek=prepared["jatek"],
            modell=mode,
            proba=latest_vector_adjusted,
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
            "latest_vector": latest_vector_adjusted,
            "latest_vector_raw": np.asarray(latest_vector_raw, dtype=float),
            "test_vectors": test_vectors,
            "y_test": y_test,
            "meta": {
                "sample_count": len(X_train),
                "input_kind": input_kind,
                "sequence_length": self.config.sequence_length if mode == "lstm" else None,
                "recency": recency_meta,
            },
        }

    def _select_mode(self, flat_train_rows: int) -> str:
        if flat_train_rows < self.config.auto_rf_threshold:
            return "random_forest"
        if flat_train_rows < self.config.auto_xgb_threshold:
            return "xgboost"
        if self._is_lstm_available():
            return "lstm"
        return "quantum"

    def _predict_single(self, model: Any, X: np.ndarray) -> np.ndarray:
        pred = model.predict_proba(X)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 2:
            return pred[0]
        return pred.flatten()

    def _predict_matrix(self, model: Any, X: np.ndarray) -> np.ndarray:
        pred = model.predict_proba(X)
        pred = np.asarray(pred, dtype=float)

        if pred.ndim == 1:
            return np.tile(pred, (len(X), 1))

        return pred

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

        hit_counts_k1 = []
        hit_counts_k2 = []
        any_hit_k1 = []
        exact_match_k1 = []

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

    def _build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(X) < sequence_length:
            return np.empty((0, sequence_length, X.shape[1])), np.empty((0, y.shape[1]))

        X_seq = []
        y_seq = []
        for end_idx in range(sequence_length - 1, len(X)):
            start_idx = end_idx - sequence_length + 1
            X_seq.append(X[start_idx:end_idx + 1])
            y_seq.append(y[end_idx])

        return np.array(X_seq, dtype=float), np.array(y_seq, dtype=int)

    def _build_latest_sequence(
        self,
        X: np.ndarray,
        prediction_row_array: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        if sequence_length <= 1:
            return prediction_row_array.reshape(1, 1, -1)

        tail_length = sequence_length - 1
        if len(X) < tail_length:
            raise ValueError("Túl kevés előzmény van az LSTM előrejelzéshez.")

        tail = X[-tail_length:]
        latest = np.vstack([tail, prediction_row_array])
        return latest.reshape(1, sequence_length, -1)

    def _is_lstm_available(self) -> bool:
        return LSTMLotteryModel is not None

    def _get_lstm_error_message(self) -> str:
        if _LSTM_IMPORT_ERROR is None:
            return "Az LSTM modell nem elérhető."
        return "A tensorflow csomag nincs telepítve. Telepítsd például így: pip install tensorflow"