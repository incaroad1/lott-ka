import csv
from pathlib import Path
from typing import Any


class BaseImporter:
    delimiter = ";"
    encoding = "utf-8"

    def read_rows(self, path: str | Path) -> list[list[str]]:
        rows: list[list[str]] = []
        with open(path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                cleaned = [str(cell).strip() for cell in row]
                if any(cleaned):
                    rows.append(cleaned)
        return rows

    def import_file(self, path: str | Path) -> dict[str, Any]:
        raise NotImplementedError
