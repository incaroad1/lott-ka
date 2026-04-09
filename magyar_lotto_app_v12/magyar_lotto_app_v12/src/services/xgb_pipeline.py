from __future__ import annotations

from pathlib import Path
from typing import Any

from src.builders.dataset_builder import DatasetBuilder
from src.importers.factory import get_importer
from src.models.xgboost_model import (
    XGBoostLotteryModel,
    XGBoostTrainingResult,
)
from src.services.pipeline_utils import prediction_row_to_array, split_features_targets


def run_xgboost_pipeline(csv_path: str | Path, jatek: str) -> dict[str, Any]:
    importer = get_importer("skandi" if jatek.startswith("skandi") else jatek)
    imported = importer.import_file(csv_path)

    records = [r for r in imported["records"] if r["jatek"] == jatek]
    builder = DatasetBuilder(min_history=10)

    training_rows = builder.build_training_rows(records)
    prediction_row = builder.build_prediction_row(records, jatek)

    X, y, feature_cols, _target_cols = split_features_targets(training_rows)
    X_latest = prediction_row_to_array(prediction_row, feature_cols)

    if len(X) <= 100:
        raise ValueError("Túl kevés tanító sor a stabil teszteléshez.")

    split_idx = len(X) - 100
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBoostLotteryModel()
    model.fit(X_train, y_train, jatek=jatek)

    evaluation = model.evaluate(X_test, y_test, top_k_primary=5, top_k_secondary=10)

    training_result = XGBoostTrainingResult(
        jatek=jatek,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_count=X_train.shape[1],
        target_count=y_train.shape[1],
        avg_hit_at_5=evaluation["avg_hit_at_5"],
        avg_hit_at_10=evaluation["avg_hit_at_10"],
        any_hit_rate_at_5=evaluation["any_hit_rate_at_5"],
        exact_match_rate_at_5=evaluation["exact_match_rate_at_5"],
    )

    prediction = model.build_prediction_result(X_latest, top_k=min(12, y_train.shape[1]))
    prediction["score_vector"] = [float(x) for x in model.predict_proba_vector(X_latest)[0]]

    return {
        "imported_record_count": len(records),
        "error_count": len(imported["errors"]),
        "training_result": training_result,
        "prediction": prediction,
    }
