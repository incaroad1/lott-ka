from pathlib import Path
from pprint import pprint

from src.services.compare_pipeline import run_compare_pipeline


def main():
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / 'data' / 'skandi.csv'

    result = run_compare_pipeline(csv_path=csv_path, jatek='skandi_gepi')

    print(f"CSV fájl: {csv_path}")
    print(f"Beolvasott rekordok: {result['imported_record_count']}")
    print(f"Hibás sorok: {result['error_count']}")
    print(f"Legjobb modell: {result['best_model']}")
    print("Összesítő rangsor:")
    pprint(result['ranking'])
    print("\nEnsemble tipp:")
    pprint(result['ensemble_prediction'])
    print("\nModellek top tippjei:")
    for model_name, prediction in result['model_predictions'].items():
        print(f"[{model_name}]")
        pprint(prediction)


if __name__ == '__main__':
    main()
