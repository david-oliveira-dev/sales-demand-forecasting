"""Feature engineering para os modelos de previsão de vendas.

Dois consumidores:
- **XGBoost** (tabular): features de calendário, termos de Fourier (sazonalidade
  anual), lags e médias móveis do alvo, além dos regressores exógenos conhecidos
  (promo, holiday).
- **LSTM** (sequencial): janelas deslizantes da série normalizada — geradas por
  `make_sequences`.

Todas as features respeitam a causalidade temporal (só usam passado), evitando
vazamento de futuro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "sales"
EXOG = ["promo", "holiday"]
LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 14, 28)
FOURIER_ORDER = 3


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de calendário a partir da coluna `date`."""
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["dayofweek"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return df


def add_fourier_terms(df: pd.DataFrame, period: float = 365.25, order: int = FOURIER_ORDER) -> pd.DataFrame:
    """Adiciona pares seno/cosseno para capturar a sazonalidade anual suave."""
    df = df.copy()
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    for k in range(1, order + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * doy / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * doy / period)
    return df


def add_lag_features(
    df: pd.DataFrame, target: str = TARGET,
    lags: tuple[int, ...] = LAGS, windows: tuple[int, ...] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """Adiciona lags e médias móveis do alvo (shift(1) evita usar o próprio dia)."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    for w in windows:
        df[f"rollmean_{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"rollstd_{w}"] = df[target].shift(1).rolling(w).std()
    return df


def build_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Monta a matriz de features (X) e o alvo (y) para modelos tabulares.

    Remove as primeiras linhas sem histórico de lag. `df` deve ter as colunas
    date, sales, promo, holiday.
    """
    feats = add_lag_features(add_fourier_terms(add_calendar_features(df)))
    feats = feats.dropna().reset_index(drop=True)
    feature_cols = [c for c in feats.columns if c not in ("date", TARGET)]
    return feats[feature_cols], feats[TARGET]


def make_sequences(
    values: np.ndarray, window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera janelas deslizantes (X) e o próximo valor (y) para o LSTM.

    `values` é a série já normalizada (1D). Retorna X de shape
    (n, window, 1) e y de shape (n,).
    """
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    X = np.array(X).reshape(-1, window, 1)
    return X, np.array(y)
