from pathlib import Path
from typing import Any

from src.core.config import LOTTO_CONFIG
from src.core.validator import parse_int, normalize_date, validate_draw
from src.importers.base_importer import BaseImporter


class HatosImporter(BaseImporter):
    def import_file(self, path: str | Path) -> dict[str, Any]:
        rule = LOTTO_CONFIG["hatos"]
        rows = self.read_rows(path)

        records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            try:
                cleaned_row = list(row)
                while cleaned_row and cleaned_row[-1] == "":
                    cleaned_row.pop()

                record = {
                    "jatek": "hatos",
                    "ev": parse_int(cleaned_row[0]),
                    "het": parse_int(cleaned_row[1]),
                    "datum": normalize_date(cleaned_row[3]),
                    "szamok": validate_draw(
                        cleaned_row[-6:],
                        rule.numbers_per_draw,
                        rule.number_min,
                        rule.number_max,
                    ),
                    "meta": {"nap": cleaned_row[2].strip()},
                }
                records.append(record)
            except Exception as e:
                errors.append({"row_index": idx, "row": row, "error": str(e)})

        return {"records": records, "errors": errors}
