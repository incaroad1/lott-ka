from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import mean_squared_error

from .base_model import BaseLotteryModel


class LSTMLotteryModel(BaseLotteryModel):
    """
    LSTM-alapú modell.
    Bemenet:
        X: (minta_db, feature_db) vagy (minta_db, time_steps, feature_db)
    Kimenet:
        y: (minta_db, szamter_db) bináris mátrix
    """

    def __init__(
        self,
        time_steps: int = 10,
        lstm_units: int = 64,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 0,
    ):
        super().__init__(name="lstm")

        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            from tensorflow.keras.optimizers import Adam

            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            self.Input = Input
            self.Adam = Adam
        except ImportError as e:
            raise ImportError(
                "A tensorflow csomag nincs telepítve. Telepítsd például így: pip install tensorflow"
            ) from e

        self.time_steps = int(time_steps)
        self.lstm_units = int(lstm_units)
        self.dense_units = int(dense_units)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)

        self.model = None
        self.output_dim: Optional[int] = None
        self.feature_dim: Optional[int] = None

        self.metadata.update({
            "model_type": "KerasLSTM",
            "time_steps": self.time_steps,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        })

    def _ensure_numpy(self, arr) -> np.ndarray:
        return np.asarray(arr, dtype=float)

    def _validate_y(self, y) -> np.ndarray:
        y_arr = self._ensure_numpy(y)
        if y_arr.ndim != 2:
            raise ValueError("A y célmátrixnak 2 dimenziósnak kell lennie: (minta_db, szamter_db)")
        return y_arr

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """
        Elfogad:
        - 2D: (minta_db, feature_db)
        - 3D: (minta_db, time_steps, feature_db)
        """
        X_arr = self._ensure_numpy(X)

        if X_arr.ndim == 3:
            return X_arr

        if X_arr.ndim != 2:
            raise ValueError("Az X bemenetnek 2D vagy 3D tömbnek kell lennie.")

        n_samples, n_features = X_arr.shape

        if self.time_steps <= 1:
            return X_arr.reshape(n_samples, 1, n_features)

        # Egyszerű fallback:
        # a 2D feature-vektort time_steps darab azonos lépésre ismételjük,
        # ha még nincs valódi szekvenciás buildered.
        X_seq = np.repeat(X_arr[:, np.newaxis, :], self.time_steps, axis=1)
        return X_seq

    def _build_model(self, input_shape, output_dim: int):
        model = self.Sequential([
            self.Input(shape=input_shape),
            self.LSTM(self.lstm_units, return_sequences=False),
            self.Dropout(self.dropout_rate),
            self.Dense(self.dense_units, activation="relu"),
            self.Dropout(self.dropout_rate),
            self.Dense(output_dim, activation="sigmoid"),
        ])

        model.compile(
            optimizer=self.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[],
        )
        return model

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).flatten()
        scores = np.clip(scores, 0.0, 1.0)

        s = scores.sum()
        if s > 0:
            return scores / s

        if len(scores) == 0:
            return scores
        return np.ones_like(scores) / len(scores)

    def fit(self, X, y, validation_data=None, **kwargs) -> None:
        X_seq = self._reshape_input(X)
        y_arr = self._validate_y(y)

        if len(X_seq) != len(y_arr):
            raise ValueError("Az X és y mintaszáma nem egyezik.")

        self.feature_dim = X_seq.shape[-1]
        self.output_dim = y_arr.shape[1]

        self.model = self._build_model(
            input_shape=(X_seq.shape[1], X_seq.shape[2]),
            output_dim=self.output_dim
        )

        fit_kwargs = {
            "epochs": kwargs.get("epochs", self.epochs),
            "batch_size": kwargs.get("batch_size", self.batch_size),
            "verbose": kwargs.get("verbose", self.verbose),
        }

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq = self._reshape_input(X_val)
            y_val_arr = self._validate_y(y_val)
            fit_kwargs["validation_data"] = (X_val_seq, y_val_arr)

        self.model.fit(X_seq, y_arr, **fit_kwargs)

        self.is_trained = True
        self.last_train_size = len(X_seq)

        train_pred = self.model.predict(X_seq, verbose=0)
        mse = mean_squared_error(y_arr, train_pred)
        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Az LSTM modell még nincs betanítva.")

        X_seq = self._reshape_input(X)
        pred = self.model.predict(X_seq, verbose=0)

        if pred.ndim == 2:
            scores = pred[0]
        else:
            scores = pred.flatten()

        return self._normalize_scores(scores)

    def evaluate(self, X_val, y_val, **kwargs) -> float:
        if not self.is_trained or self.model is None:
            return 0.0

        X_seq = self._reshape_input(X_val)
        y_arr = self._validate_y(y_val)

        pred = self.model.predict(X_seq, verbose=0)
        mse = mean_squared_error(y_arr, pred)

        perf = 1.0 / (1.0 + float(mse))
        self.last_performance_score = float(np.clip(perf, 0.0, 1.0))
        return self.last_performance_score

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.name,
            "is_trained": self.is_trained,
            "last_performance_score": self.last_performance_score,
            "last_train_size": self.last_train_size,
            "output_dim": self.output_dim,
            "feature_dim": self.feature_dim,
            "metadata": self.metadata,
        }