Ez a csomag tartalmazza:
- magyar lottó CSV importerek
- validáció
- feature engine
- dataset builder
- RandomForest modell + pipeline
- XGBoost modell + pipeline

Futtatás:
python example_usage.py
python example_random_forest.py
python example_xgboost.py

Megjegyzés:
Az XGBoost példa futtatásához szükség lehet az xgboost csomag telepítésére.


V6: PyTorch alapú LSTM modell + LSTM pipeline + example_lstm.py
Az LSTM részhez PyTorch kell: pip install torch


V7: compare_pipeline + example_compare_models.py + ensemble átlagoló réteg.
Futtatás: python example_compare_models.py


V8: Hatos compare example hozzáadva: python example_compare_hatos.py

V9: Skandi gépi és kézi compare example hozzáadva.


GUI indítás:
python run_gui.py

A GUI módok:
- compare: mindhárom modell + ensemble
- random_forest
- xgboost
- lstm


V11: GUI csomag szinkronizálva a javított Hatos-importerrel (régi 6+1 sorok kezelése).


V12 újdonságok:
- X darab szelvény generálás
- top és diverzifikált stratégia
- Skandi kombinált mód (gépi + kézi közös pontszám)
