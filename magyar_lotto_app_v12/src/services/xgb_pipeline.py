from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.model_manager import ModelManager, ModelManagerConfig


def run_xgboost_pipeline(
    csv_path: str | Path,
    jatek: str,
) -> dict[str, Any]:
    manager = ModelManager(
        ModelManagerConfig(
            sequence_length=15,
            min_test_rows=100,
            min_history=10,
            auto_rf_threshold=180,
            auto_xgb_threshold=420,
            top_k_prediction=12,
        )
    )
    return manager.predict(csv_path=csv_path, jatek=jatek, mode="xgboost")