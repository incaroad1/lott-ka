from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.compare_pipeline import run_compare_pipeline
from src.services.lstm_pipeline import run_lstm_pipeline
from src.services.rf_pipeline import run_random_forest_pipeline
from src.services.ticket_generator import combine_predictions
from src.services.xgb_pipeline import run_xgboost_pipeline


def run_skandi_combined_pipeline(csv_path: str | Path, mode: str = "compare") -> dict[str, Any]:
    if mode == "compare":
        gepi = run_compare_pipeline(csv_path=csv_path, jatek="skandi_gepi")
        kezi = run_compare_pipeline(csv_path=csv_path, jatek="skandi_kezi")
        combined_prediction = combine_predictions(
            [gepi["models"]["ensemble_avg"]["prediction"], kezi["models"]["ensemble_avg"]["prediction"]],
            jatek="skandi_kombinalt",
            model_name="skandi_kombinalt_ensemble",
        )
        return {
            "imported_record_count": max(gepi.get("imported_record_count", 0), kezi.get("imported_record_count", 0)),
            "error_count": max(gepi.get("error_count", 0), kezi.get("error_count", 0)),
            "best_model": "skandi_kombinalt_ensemble",
            "scoreboard": [
                {"modell": "skandi_gepi_best", "best_model": gepi.get("best_model")},
                {"modell": "skandi_kezi_best", "best_model": kezi.get("best_model")},
            ],
            "models": {
                "skandi_gepi": gepi,
                "skandi_kezi": kezi,
                "skandi_kombinalt": {"prediction": combined_prediction},
            },
            "prediction": combined_prediction,
        }

    runners = {
        "random_forest": run_random_forest_pipeline,
        "xgboost": run_xgboost_pipeline,
        "lstm": run_lstm_pipeline,
    }
    runner = runners[mode]
    gepi = runner(csv_path=csv_path, jatek="skandi_gepi")
    kezi = runner(csv_path=csv_path, jatek="skandi_kezi")
    combined_prediction = combine_predictions(
        [gepi["prediction"], kezi["prediction"]],
        jatek="skandi_kombinalt",
        model_name=f"skandi_kombinalt_{mode}",
    )
    return {
        "imported_record_count": max(gepi.get("imported_record_count", 0), kezi.get("imported_record_count", 0)),
        "error_count": max(gepi.get("error_count", 0), kezi.get("error_count", 0)),
        "training_result": {"gepi": gepi.get("training_result"), "kezi": kezi.get("training_result")},
        "prediction": combined_prediction,
        "models": {"skandi_gepi": gepi, "skandi_kezi": kezi},
    }
