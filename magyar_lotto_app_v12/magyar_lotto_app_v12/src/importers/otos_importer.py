from pathlib import Path
from typing import Any

from src.core.config import LOTTO_CONFIG
from src.core.validator import parse_int, normalize_date, validate_draw
from src.importers.base_importer import BaseImporter


class OtosImporter(BaseImporter):
    def import_file(self, path: str | Path) -> dict[str, Any]:
        rule = LOTTO_CONFIG["otos"]
        rows = self.read_rows(path)

        records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            try:
                record = {
                    "jatek": "otos",
                    "ev": parse_int(row[0]),
                    "het": parse_int(row[1]),
                    "datum": normalize_date(row[2]),
                    "szamok": validate_draw(
                        row[-5:],
                        rule.numbers_per_draw,
                        rule.number_min,
                        rule.number_max,
                    ),
                    "meta": {},
                }
                records.append(record)
            except Exception as e:
                errors.append({"row_index": idx, "row": row, "error": str(e)})

        return {"records": records, "errors": errors}