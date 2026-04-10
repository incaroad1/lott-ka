from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.model_manager import ModelManager, ModelManagerConfig
from src.services.ticket_generator import generate_tickets_from_prediction


def run_compare_pipeline(
    csv_path: str | Path,
    jatek: str,
    ticket_count: int = 3,
    ticket_profile: str = "kiegyensulyozott",
) -> dict[str, Any]:
    manager = ModelManager(ModelManagerConfig())
    result = manager.compare_models(csv_path=csv_path, jatek=jatek)

    best_model = result.get("best_model")
    models = result.get("models", {})

    prediction = None
    if best_model == "ensemble_smart":
        prediction = models.get("ensemble_avg", {}).get("prediction")
    elif best_model:
        prediction = models.get(best_model, {}).get("prediction")

    if prediction:
        result["ticket_bundle"] = generate_tickets_from_prediction(
            prediction=prediction,
            jatek=jatek,
            ticket_count=ticket_count,
            strategy="diverzifikalt",
            profile=ticket_profile,
        )
    else:
        result["ticket_bundle"] = None

    result.setdefault("meta", {})
    result["meta"]["ticket_profile"] = ticket_profile

    return result