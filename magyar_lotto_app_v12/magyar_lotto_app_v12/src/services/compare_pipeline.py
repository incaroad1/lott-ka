from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.model_manager import ModelManager, ModelManagerConfig


def run_compare_pipeline(
    csv_path: str | Path,
    jatek: str,
    sequence_length: int = 15,
) -> dict[str, Any]:
    """
    V12 compare mód új, ModelManager-alapú belépési pont.

    Megtartja a GUI-kompatibilis eredménystruktúrát:
    - imported_record_count
    - error_count
    - scoreboard
    - best_model
    - models
      - random_forest
      - xgboost
      - lstm
      - ensemble_avg
    """

    manager = ModelManager(
        ModelManagerConfig(
            sequence_length=sequence_length,
            min_test_rows=100,
            min_history=10,
            auto_rf_threshold=180,
            auto_xgb_threshold=420,
            top_k_prediction=12,
        )
    )
    return manager.compare_models(csv_path=csv_path, jatek=jatek)