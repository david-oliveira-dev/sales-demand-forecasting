"""Modelo XGBoost para previsão de vendas (abordagem tabular + recursiva).

Trata a previsão como regressão: features de calendário, Fourier, lags e médias
móveis do alvo, mais os exógenos promo/holiday. Como os lags dependem do passado,
a previsão de múltiplos passos é **recursiva** — cada dia previsto realimenta o
histórico para calcular os lags do dia seguinte (sem vazamento de futuro).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import (
    add_calendar_features, add_fourier_terms, add_lag_features, build_supervised,
)

NAME = "xgboost"


class XGBModel:
    name = NAME

    def __init__(self) -> None:
        self._model = None
        self._history: pd.DataFrame | None = None
        self._feature_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> "XGBModel":
        from xgboost import XGBRegressor

        X, y = build_supervised(df)
        self._feature_cols = list(X.columns)
        self._model = XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        )
        self._model.fit(X, y)
        self._history = df[["date", "sales", "promo", "holiday"]].copy()
        return self

    def _features_for_next(self, history: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
        """Constrói a linha de features para o próximo dia a prever."""
        new = pd.DataFrame([{
            "date": row["date"], "sales": np.nan,
            "promo": row["promo"], "holiday": row["holiday"],
        }])
        tmp = pd.concat([history, new], ignore_index=True)
        feats = add_lag_features(add_fourier_terms(add_calendar_features(tmp)))
        return feats[self._feature_cols].iloc[[-1]]

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        history = self._history.copy()
        preds = []
        for _, row in future_df.iterrows():
            x = self._features_for_next(history, row)
            yhat = float(self._model.predict(x)[0])
            yhat = max(yhat, 0.0)
            preds.append(yhat)
            history = pd.concat([history, pd.DataFrame([{
                "date": row["date"], "sales": yhat,
                "promo": row["promo"], "holiday": row["holiday"],
            }])], ignore_index=True)
        return np.array(preds)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(
            {"model": self._model, "history": self._history, "cols": self._feature_cols},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "XGBModel":
        import joblib

        data = joblib.load(path)
        obj = cls()
        obj._model, obj._history, obj._feature_cols = (
            data["model"], data["history"], data["cols"],
        )
        return obj
