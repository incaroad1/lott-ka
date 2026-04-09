from pathlib import Path
from pprint import pprint

from src.importers.factory import get_importer
from src.builders.dataset_builder import DatasetBuilder


def main():
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "otos.csv"

    importer = get_importer("otos")
    imported = importer.import_file(csv_path)

    print(f"CSV fájl: {csv_path}")
    print(f"Beolvasott rekordok: {len(imported['records'])}")
    print(f"Hibás sorok: {len(imported['errors'])}")

    if not imported["records"]:
        print("Nincs feldolgozható rekord.")
        if imported["errors"]:
            print("Első hiba:", imported["errors"][0])
        return

    builder = DatasetBuilder()
    dataset = builder.build_training_rows(imported["records"])

    print(f"Tanító sorok: {len(dataset)}")
    if dataset:
        print("Első tanító sor részlete:")
        preview = dict(list(dataset[0].items())[:20])
        pprint(preview)


if __name__ == "__main__":
    main()
