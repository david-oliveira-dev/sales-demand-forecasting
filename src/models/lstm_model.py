"""Modelo LSTM (Keras/TensorFlow) para previsão de vendas.

Rede recorrente univariada: aprende a mapear uma janela dos últimos `window`
dias (normalizados) para o dia seguinte. A previsão de múltiplos passos é
**recursiva** — a predição alimenta a janela do passo seguinte.

Roda em CPU (máquina sem GPU); a rede é pequena de propósito. O import do
TensorFlow é isolado nos métodos para não pesar quem só usa os outros modelos.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import make_sequences

# Silencia logs verbosos do TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NAME = "lstm"


class LSTMModel:
    name = NAME

    def __init__(self, window: int = 28, units: int = 32, epochs: int = 25,
                 batch_size: int = 32, seed: int = 42) -> None:
        self.window = window
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._min = None
        self._max = None
        self._last_window: np.ndarray | None = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self._min) / (self._max - self._min)

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        return x * (self._max - self._min) + self._min

    def fit(self, df: pd.DataFrame) -> "LSTMModel":
        import tensorflow as tf

        tf.random.set_seed(self.seed)
        series = df["sales"].astype(float).to_numpy()
        self._min, self._max = float(series.min()), float(series.max())
        scaled = self._scale(series)

        X, y = make_sequences(scaled, self.window)
        model = tf.keras.Sequential([
            tf.keras.layers.Input((self.window, 1)),
            tf.keras.layers.LSTM(self.units),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        self._model = model
        self._last_window = scaled[-self.window:].copy()
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        window = self._last_window.copy()
        preds = []
        for _ in range(len(future_df)):
            x = window.reshape(1, self.window, 1)
            yhat_scaled = float(self._model.predict(x, verbose=0)[0, 0])
            preds.append(yhat_scaled)
            window = np.append(window[1:], yhat_scaled)
        return self._unscale(np.array(preds)).clip(min=0)

    def save(self, path: Path) -> None:
        """Salva o modelo Keras (.keras) e os parâmetros de escala ao lado."""
        import joblib

        path = Path(path)
        keras_path = path.with_suffix(".keras")
        self._model.save(keras_path)
        joblib.dump(
            {"min": self._min, "max": self._max, "window": self.window,
             "last_window": self._last_window, "keras_path": str(keras_path)},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        import joblib
        import tensorflow as tf

        data = joblib.load(path)
        obj = cls(window=data["window"])
        obj._min, obj._max = data["min"], data["max"]
        obj._last_window = data["last_window"]
        obj._model = tf.keras.models.load_model(data["keras_path"])
        return obj
