"""Métricas de avaliação para previsão de séries temporais.

Reportamos várias métricas porque cada uma conta uma parte da história:
- **MAE**: erro médio absoluto, na unidade das vendas (interpretável).
- **RMSE**: penaliza mais os erros grandes.
- **MAPE**: erro percentual — comparável entre séries de escalas diferentes.
- **sMAPE**: variante simétrica do MAPE, estável quando os valores são pequenos.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Devolve todas as métricas de uma vez, arredondadas."""
    return {
        "mae": round(mae(y_true, y_pred), 3),
        "rmse": round(rmse(y_true, y_pred), 3),
        "mape": round(mape(y_true, y_pred), 3),
        "smape": round(smape(y_true, y_pred), 3),
    }
