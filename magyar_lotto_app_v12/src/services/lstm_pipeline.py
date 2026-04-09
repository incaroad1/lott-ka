
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.builders.dataset_builder import DatasetBuilder
from src.importers.factory import get_importer
from src.models.lstm_model import LSTMLotteryModel, LSTMTrainingResult
from src.services.pipeline_utils import prediction_row_to_array, split_features_targets


def _build_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) < sequence_length:
        raise ValueError(f'Túl kevés tanító sor az LSTM-hez. Kell legalább {sequence_length} sor.')

    X_seq = []
    y_seq = []
    for end_idx in range(sequence_length - 1, len(X)):
        start_idx = end_idx - sequence_length + 1
        X_seq.append(X[start_idx:end_idx + 1])
        y_seq.append(y[end_idx])

    return np.array(X_seq, dtype=float), np.array(y_seq, dtype=int)


def _build_latest_sequence(X: np.ndarray, prediction_row_array: np.ndarray, sequence_length: int) -> np.ndarray:
    if sequence_length <= 1:
        return prediction_row_array.reshape(1, 1, -1)

    tail_length = sequence_length - 1
    if len(X) < tail_length:
        raise ValueError('Túl kevés előzmény van az LSTM előrejelzéshez.')

    tail = X[-tail_length:]
    latest = np.vstack([tail, prediction_row_array])
    return latest.reshape(1, sequence_length, -1)


def run_lstm_pipeline(csv_path: str | Path, jatek: str, sequence_length: int = 15) -> dict[str, Any]:
    importer = get_importer('skandi' if jatek.startswith('skandi') else jatek)
    imported = importer.import_file(csv_path)

    records = [r for r in imported['records'] if r['jatek'] == jatek]
    builder = DatasetBuilder(min_history=10)

    training_rows = builder.build_training_rows(records)
    prediction_row = builder.build_prediction_row(records, jatek)

    X_flat, y, feature_cols, _target_cols = split_features_targets(training_rows)
    prediction_row_array = prediction_row_to_array(prediction_row, feature_cols)

    X_seq, y_seq = _build_sequences(X_flat, y, sequence_length=sequence_length)
    X_latest = _build_latest_sequence(X_flat, prediction_row_array, sequence_length=sequence_length)

    if len(X_seq) <= 100:
        raise ValueError('Túl kevés szekvencia a stabil teszteléshez.')

    split_idx = len(X_seq) - 100
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = LSTMLotteryModel()
    model.fit(X_train, y_train, jatek=jatek)

    evaluation = model.evaluate(X_test, y_test, top_k_primary=5, top_k_secondary=10)

    training_result = LSTMTrainingResult(
        jatek=jatek,
        train_rows=len(X_train),
        test_rows=len(X_test),
        sequence_length=sequence_length,
        feature_count=X_train.shape[2],
        target_count=y_train.shape[1],
        avg_hit_at_5=evaluation['avg_hit_at_5'],
        avg_hit_at_10=evaluation['avg_hit_at_10'],
        any_hit_rate_at_5=evaluation['any_hit_rate_at_5'],
        exact_match_rate_at_5=evaluation['exact_match_rate_at_5'],
        epochs=model.epochs,
    )

    prediction = model.build_prediction_result(X_latest, top_k=min(12, y_train.shape[1]))
    prediction['score_vector'] = [float(x) for x in model.predict_proba_vector(X_latest)[0]]

    return {
        'imported_record_count': len(records),
        'error_count': len(imported['errors']),
        'training_result': training_result,
        'prediction': prediction,
    }
