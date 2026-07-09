"""Modelo Prophet para previsão de vendas.

Prophet decompõe a série em tendência + sazonalidades (semanal/anual) + efeito de
regressores. Usamos `promo` e `holiday` como regressores externos, que são
conhecidos no futuro (o negócio planeja promoções e o calendário dá os feriados).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import EXOG

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

NAME = "prophet"


class ProphetModel:
    """Wrapper com interface uniforme fit/predict/save/load."""

    name = NAME

    def __init__(self) -> None:
        self._model = None

    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        from prophet import Prophet

        train = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", *EXOG]]
        model = Prophet(
            weekly_seasonality=True, yearly_seasonality=True,
            daily_seasonality=False, seasonality_mode="multiplicative",
        )
        for reg in EXOG:
            model.add_regressor(reg)
        model.fit(train)
        self._model = model
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        future = future_df.rename(columns={"date": "ds"})[["ds", *EXOG]]
        forecast = self._model.predict(future)
        return forecast["yhat"].to_numpy().clip(min=0)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: Path) -> "ProphetModel":
        import joblib

        obj = cls()
        obj._model = joblib.load(path)
        return obj
