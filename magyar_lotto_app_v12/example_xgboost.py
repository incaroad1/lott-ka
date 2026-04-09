from pathlib import Path
from pprint import pprint

from src.services.xgb_pipeline import run_xgboost_pipeline


def main():
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "otos.csv"

    result = run_xgboost_pipeline(csv_path=csv_path, jatek="otos")

    print(f"CSV fájl: {csv_path}")
    print(f"Beolvasott rekordok: {result['imported_record_count']}")
    print(f"Hibás sorok: {result['error_count']}")
    print("Tanítás eredménye:")
    pprint(result["training_result"])
    print("Következő húzás becslése:")
    pprint(result["prediction"])


if __name__ == "__main__":
    main()
