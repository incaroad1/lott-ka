
from __future__ import annotations

from typing import Any

from src.core.config import LOTTO_CONFIG
from src.features.feature_engine import FeatureEngine


class DatasetBuilder:
    def __init__(self, feature_engine: FeatureEngine | None = None, min_history: int = 10):
        self.feature_engine = feature_engine or FeatureEngine()
        self.min_history = min_history

    def build_training_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []

        sorted_records = sorted(
    records,
    key=lambda r: (
        int(r.get("ev", 0)),
        int(r.get("het", 0)),
        r.get("datum") or "",
    )
)
        training_rows: list[dict[str, Any]] = []
        history_by_game: dict[str, list[dict[str, Any]]] = {}

        for record in sorted_records:
            jatek = record['jatek']
            rule = LOTTO_CONFIG[jatek]
            history = history_by_game.setdefault(jatek, [])

            if len(history) >= self.min_history:
                feature_row = self.feature_engine.build_feature_row(
                    history=history,
                    jatek=jatek,
                    min_val=rule.number_min,
                    max_val=rule.number_max,
                    current_ev=record.get('ev'),
                    current_het=record.get('het'),
                    current_datum=record.get('datum'),
                )
                feature_row.update(
                    self.feature_engine.encode_target(record['szamok'], rule.number_min, rule.number_max)
                )
                training_rows.append(feature_row)

            history.append(record)

        return training_rows

    def build_prediction_row(self, history_records: list[dict[str, Any]], jatek: str) -> dict[str, Any]:
        if not history_records:
            raise ValueError('Előrejelzéshez nincs történeti adat.')

        rule = LOTTO_CONFIG[jatek]
        sorted_records = sorted(
            [r for r in history_records if r['jatek'] == jatek],
            key=lambda r: (
                int(r.get('ev', 0)),
                int(r.get('het', 0)),
                r.get('datum') or '',
            ),
        )
        if len(sorted_records) < self.min_history:
            raise ValueError(f'Legalább {self.min_history} húzás kell az előrejelzéshez.')

        return self.feature_engine.build_feature_row(
            history=sorted_records,
            jatek=jatek,
            min_val=rule.number_min,
            max_val=rule.number_max,
        )
