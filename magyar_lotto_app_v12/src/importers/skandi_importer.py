from pathlib import Path
from typing import Any

from src.core.config import LOTTO_CONFIG
from src.core.validator import parse_int, normalize_date, validate_draw
from src.importers.base_importer import BaseImporter


class SkandiImporter(BaseImporter):
    def import_file(self, path: str | Path) -> dict[str, Any]:
        rule_gepi = LOTTO_CONFIG["skandi_gepi"]
        rule_kezi = LOTTO_CONFIG["skandi_kezi"]
        rows = self.read_rows(path)

        records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            try:
                ev = parse_int(row[0])
                het = parse_int(row[1])
                datum = normalize_date(row[2])

                gepi = validate_draw(row[11:18], rule_gepi.numbers_per_draw, rule_gepi.number_min, rule_gepi.number_max)
                kezi = validate_draw(row[18:25], rule_kezi.numbers_per_draw, rule_kezi.number_min, rule_kezi.number_max)

                records.append({
                    "jatek": "skandi_gepi",
                    "ev": ev,
                    "het": het,
                    "datum": datum,
                    "szamok": gepi,
                    "meta": {"huzas_tipus": "gepi"},
                })

                records.append({
                    "jatek": "skandi_kezi",
                    "ev": ev,
                    "het": het,
                    "datum": datum,
                    "szamok": kezi,
                    "meta": {"huzas_tipus": "kezi"},
                })
            except Exception as e:
                errors.append({"row_index": idx, "row": row, "error": str(e)})

        return {"records": records, "errors": errors}
