from __future__ import annotations

from typing import Any

import numpy as np


def split_features_targets(training_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not training_rows:
        raise ValueError("Nincs tanító sor.")

    target_cols = sorted(
        [k for k in training_rows[0].keys() if k.startswith("target_")],
        key=lambda x: int(x.split("_")[1]),
    )
    meta_cols = {"jatek", "ev", "het", "datum"}
    feature_cols = [k for k in training_rows[0].keys() if k not in meta_cols and k not in target_cols]

    X = np.array([[row[col] for col in feature_cols] for row in training_rows], dtype=float)
    y = np.array([[row[col] for col in target_cols] for row in training_rows], dtype=int)

    return X, y, feature_cols, target_cols


def prediction_row_to_array(prediction_row: dict[str, Any], feature_cols: list[str]) -> np.ndarray:
    return np.array([[prediction_row[col] for col in feature_cols]], dtype=float)
