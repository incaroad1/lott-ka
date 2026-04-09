from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.builders.dataset_builder import DatasetBuilder
from src.importers.factory import get_importer
from src.models.lstm_model import LSTMLotteryModel, LSTMTrainingResult
from src.models.random_forest_model import RandomForestLotteryModel, RandomForestTrainingResult
from src.models.xgboost_model import XGBoostLotteryModel, XGBoostTrainingResult
from src.services.lstm_pipeline import _build_latest_sequence, _build_sequences
from src.services.pipeline_utils import prediction_row_to_array, split_features_targets


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    if max_val - min_val < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_val) / (max_val - min_val)


def _evaluate_prediction_vector(proba: np.ndarray, y_test: np.ndarray, top_k_primary: int = 5, top_k_secondary: int = 10) -> dict[str, float]:
    hit_counts_k1 = []
    hit_counts_k2 = []
    any_hit_k1 = []
    exact_match_k1 = []

    for i in range(len(y_test)):
        row_proba = proba[i]
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
        'avg_hit_at_5': round(float(np.mean(hit_counts_k1)), 6),
        'avg_hit_at_10': round(float(np.mean(hit_counts_k2)), 6),
        'any_hit_rate_at_5': round(float(np.mean(any_hit_k1)), 6),
        'exact_match_rate_at_5': round(float(np.mean(exact_match_k1)), 6),
    }


def _vector_to_prediction_result(jatek: str, modell: str, proba: np.ndarray, feature_count: int, target_count: int, top_k: int = 5) -> dict[str, Any]:
    top_idx = np.argsort(proba)[::-1][:top_k]
    top_numbers = sorted([int(i) + 1 for i in top_idx])
    scores = {int(i) + 1: round(float(proba[i]), 6) for i in top_idx}
    return {
        'jatek': jatek,
        'modell': modell,
        'top_szamok': top_numbers,
        'scoreok': scores,
        'feature_count': int(feature_count),
        'target_count': int(target_count),
    }


def run_compare_pipeline(csv_path: str | Path, jatek: str, sequence_length: int = 15) -> dict[str, Any]:
    importer = get_importer('skandi' if jatek.startswith('skandi') else jatek)
    imported = importer.import_file(csv_path)
    records = [r for r in imported['records'] if r['jatek'] == jatek]

    builder = DatasetBuilder(min_history=10)
    training_rows = builder.build_training_rows(records)
    prediction_row = builder.build_prediction_row(records, jatek)

    X_flat, y, feature_cols, _target_cols = split_features_targets(training_rows)
    X_latest_flat = prediction_row_to_array(prediction_row, feature_cols)

    if len(X_flat) <= 100:
        raise ValueError('Túl kevés tanító sor a stabil összehasonlításhoz.')

    split_idx = len(X_flat) - 100
    X_train_flat, X_test_flat = X_flat[:split_idx], X_flat[split_idx:]
    y_train_flat, y_test_flat = y[:split_idx], y[split_idx:]

    rf_model = RandomForestLotteryModel()
    rf_model.fit(X_train_flat, y_train_flat, jatek=jatek)
    rf_eval = rf_model.evaluate(X_test_flat, y_test_flat, top_k_primary=5, top_k_secondary=10)
    rf_train = RandomForestTrainingResult(
        jatek=jatek,
        train_rows=len(X_train_flat),
        test_rows=len(X_test_flat),
        feature_count=X_train_flat.shape[1],
        target_count=y_train_flat.shape[1],
        sample_f1=0.0,
        avg_hit_at_5=rf_eval['avg_hit_at_5'],
        avg_hit_at_10=rf_eval['avg_hit_at_10'],
        any_hit_rate_at_5=rf_eval['any_hit_rate_at_5'],
        exact_match_rate_at_5=rf_eval['exact_match_rate_at_5'],
    )
    rf_prediction = rf_model.build_prediction_result(X_latest_flat, top_k=min(12, y_train_flat.shape[1]))
    rf_latest_vector = rf_model.predict_proba_vector(X_latest_flat)[0]
    rf_prediction['score_vector'] = [float(x) for x in rf_latest_vector]

    xgb_model = XGBoostLotteryModel()
    xgb_model.fit(X_train_flat, y_train_flat, jatek=jatek)
    xgb_eval = xgb_model.evaluate(X_test_flat, y_test_flat, top_k_primary=5, top_k_secondary=10)
    xgb_train = XGBoostTrainingResult(
        jatek=jatek,
        train_rows=len(X_train_flat),
        test_rows=len(X_test_flat),
        feature_count=X_train_flat.shape[1],
        target_count=y_train_flat.shape[1],
        avg_hit_at_5=xgb_eval['avg_hit_at_5'],
        avg_hit_at_10=xgb_eval['avg_hit_at_10'],
        any_hit_rate_at_5=xgb_eval['any_hit_rate_at_5'],
        exact_match_rate_at_5=xgb_eval['exact_match_rate_at_5'],
    )
    xgb_prediction = xgb_model.build_prediction_result(X_latest_flat, top_k=min(12, y_train_flat.shape[1]))
    xgb_latest_vector = xgb_model.predict_proba_vector(X_latest_flat)[0]
    xgb_prediction['score_vector'] = [float(x) for x in xgb_latest_vector]

    X_seq, y_seq = _build_sequences(X_flat, y, sequence_length=sequence_length)
    X_latest_seq = _build_latest_sequence(X_flat, X_latest_flat, sequence_length=sequence_length)
    if len(X_seq) <= 100:
        raise ValueError('Túl kevés LSTM szekvencia a stabil összehasonlításhoz.')

    split_idx_seq = len(X_seq) - 100
    X_train_seq, X_test_seq = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
    y_train_seq, y_test_seq = y_seq[:split_idx_seq], y_seq[split_idx_seq:]

    lstm_model = LSTMLotteryModel()
    lstm_model.fit(X_train_seq, y_train_seq, jatek=jatek)
    lstm_eval = lstm_model.evaluate(X_test_seq, y_test_seq, top_k_primary=5, top_k_secondary=10)
    lstm_train = LSTMTrainingResult(
        jatek=jatek,
        train_rows=len(X_train_seq),
        test_rows=len(X_test_seq),
        sequence_length=sequence_length,
        feature_count=X_train_seq.shape[2],
        target_count=y_train_seq.shape[1],
        avg_hit_at_5=lstm_eval['avg_hit_at_5'],
        avg_hit_at_10=lstm_eval['avg_hit_at_10'],
        any_hit_rate_at_5=lstm_eval['any_hit_rate_at_5'],
        exact_match_rate_at_5=lstm_eval['exact_match_rate_at_5'],
        epochs=lstm_model.epochs,
    )
    lstm_prediction = lstm_model.build_prediction_result(X_latest_seq, top_k=min(12, y_train_seq.shape[1]))
    lstm_latest_vector = lstm_model.predict_proba_vector(X_latest_seq)[0]
    lstm_prediction['score_vector'] = [float(x) for x in lstm_latest_vector]

    ensemble_latest = (
        _normalize_scores(rf_latest_vector) +
        _normalize_scores(xgb_latest_vector) +
        _normalize_scores(lstm_latest_vector)
    ) / 3.0
    ensemble_prediction = _vector_to_prediction_result(
        jatek=jatek,
        modell='ensemble_avg',
        proba=ensemble_latest,
        feature_count=X_train_flat.shape[1],
        target_count=y_train_flat.shape[1],
        top_k=min(12, y_train_flat.shape[1]),
    )
    ensemble_prediction['score_vector'] = [float(x) for x in ensemble_latest]

    rf_test_vector = rf_model.predict_proba_vector(X_test_flat)
    xgb_test_vector = xgb_model.predict_proba_vector(X_test_flat)
    # az LSTM csak az utolsó 100 célhúzásra ad tesztet; ez ugyanarra az utolsó 100 húzásra esik
    lstm_test_vector = lstm_model.predict_proba_vector(X_test_seq)

    ensemble_test = (
        np.apply_along_axis(_normalize_scores, 1, rf_test_vector) +
        np.apply_along_axis(_normalize_scores, 1, xgb_test_vector) +
        np.apply_along_axis(_normalize_scores, 1, lstm_test_vector)
    ) / 3.0
    ensemble_eval = _evaluate_prediction_vector(ensemble_test, y_test_seq, top_k_primary=5, top_k_secondary=10)

    scoreboard = [
        {'modell': 'lstm', **lstm_eval},
        {'modell': 'xgboost', **xgb_eval},
        {'modell': 'random_forest', **rf_eval},
        {'modell': 'ensemble_avg', **ensemble_eval},
    ]
    scoreboard.sort(key=lambda row: (row['avg_hit_at_5'], row['avg_hit_at_10'], row['any_hit_rate_at_5']), reverse=True)

    return {
        'imported_record_count': len(records),
        'error_count': len(imported['errors']),
        'scoreboard': scoreboard,
        'best_model': scoreboard[0]['modell'] if scoreboard else None,
        'models': {
            'random_forest': {
                'training_result': rf_train,
                'prediction': rf_prediction,
            },
            'xgboost': {
                'training_result': xgb_train,
                'prediction': xgb_prediction,
            },
            'lstm': {
                'training_result': lstm_train,
                'prediction': lstm_prediction,
            },
            'ensemble_avg': {
                'training_result': ensemble_eval,
                'prediction': ensemble_prediction,
            },
        },
    }
