from pathlib import Path
from pprint import pprint

from src.services.compare_pipeline import run_compare_pipeline


def main():
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / 'data' / 'hatos.csv'

    result = run_compare_pipeline(csv_path=csv_path, jatek='hatos', sequence_length=15)

    print(f'CSV fájl: {csv_path}')
    print(f"Beolvasott rekordok: {result['imported_record_count']}")
    print(f"Hibás sorok: {result['error_count']}")
    print(f"Legjobb modell: {result['best_model']}")
    print('Összesítő rangsor:')
    pprint(result['scoreboard'])
    print('\nEnsemble tipp:')
    pprint(result['models']['ensemble_avg']['prediction'])
    print('\nModellek top tippjei:')
    for model_name in ('random_forest', 'xgboost', 'lstm'):
        print(f'[{model_name}]')
        pprint(result['models'][model_name]['prediction'])


if __name__ == '__main__':
    main()
