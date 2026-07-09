"""Modelos ARIMA e SARIMA (statsmodels) para previsão de vendas.

- **ARIMA**: captura autocorrelação + tendência (via diferenciação), sem
  componente sazonal explícito.
- **SARIMA**: adiciona sazonalidade **semanal** (m=7) e usa `promo`/`holiday`
  como variáveis exógenas (SARIMAX) — o efeito de promoção/feriado é modelado
  explicitamente.

As ordens são fixas (escolhidas pela natureza da série) para evitar a dependência
extra de auto-arima e manter o treino rápido e reproduzível.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import EXOG


class _SarimaxBase:
    name = "sarimax"
    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)
    use_exog = False

    def __init__(self) -> None:
        self._res = None

    def fit(self, df: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        endog = df["sales"].astype(float).to_numpy()
        exog = df[EXOG].astype(float).to_numpy() if self.use_exog else None
        self._res = SARIMAX(
            endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        exog = future_df[EXOG].astype(float).to_numpy() if self.use_exog else None
        forecast = self._res.forecast(steps=len(future_df), exog=exog)
        return np.asarray(forecast, dtype=float).clip(min=0)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump({"res": self._res, "cls": type(self).__name__}, path)

    @classmethod
    def load(cls, path: Path):
        import joblib

        data = joblib.load(path)
        obj = cls()
        obj._res = data["res"]
        return obj


class ArimaModel(_SarimaxBase):
    """ARIMA(2,1,2) — sem sazonalidade nem exógenas."""

    name = "arima"
    order = (2, 1, 2)
    seasonal_order = (0, 0, 0, 0)
    use_exog = False


class SarimaModel(_SarimaxBase):
    """SARIMA(1,1,1)(1,1,1,7) com promo/holiday como exógenas (sazonalidade semanal)."""

    name = "sarima"
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    use_exog = True
