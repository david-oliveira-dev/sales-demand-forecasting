"""Testes de métricas, modelos e backtest (Etapa 5).

Usa séries pequenas e horizonte curto para rodar rápido no CI. O LSTM é testado
à parte (test_lstm.py) por ser mais lento.
"""
import numpy as np

from src.data.generate_synthetic import generate_sales
from src.models.arima_model import ArimaModel, SarimaModel
from src.models.backtest import run_backtest, temporal_split
from src.models.metrics import all_metrics, mae, rmse
from src.models.prophet_model import ProphetModel
from src.models.xgb_model import XGBModel


def test_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = all_metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["mape"] == 0.0


def test_rmse_penalizes_more_than_mae():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    p = np.array([0.0, 0.0, 0.0, 8.0])  # um erro grande
    assert rmse(y, p) > mae(y, p)


def test_temporal_split_has_no_leakage():
    df = generate_sales(days=120, seed=1)
    train, test = temporal_split(df, 14)
    assert len(test) == 14 and len(train) == 106
    assert train["date"].max() < test["date"].min()


def test_arima_forecast_shape_and_nonnegative():
    train, test = temporal_split(generate_sales(days=220, seed=1), 14)
    preds = ArimaModel().fit(train).predict(test)
    assert len(preds) == 14 and (preds >= 0).all()


def test_sarima_uses_exog_and_forecasts():
    train, test = temporal_split(generate_sales(days=220, seed=2), 14)
    preds = SarimaModel().fit(train).predict(test)
    assert len(preds) == 14 and np.isfinite(preds).all()


def test_xgb_recursive_forecast():
    train, test = temporal_split(generate_sales(days=220, seed=3), 14)
    preds = XGBModel().fit(train).predict(test)
    assert len(preds) == 14 and (preds > 0).all()


def test_prophet_forecast_reasonable():
    train, test = temporal_split(generate_sales(days=400, seed=4), 21)
    preds = ProphetModel().fit(train).predict(test)
    # Erro percentual sanidade: bem abaixo de 100%.
    assert all_metrics(test["sales"].to_numpy(), preds)["mape"] < 50


def test_backtest_selects_and_saves():
    summary = run_backtest(n_days=260, seed=1, horizon=14, use_lstm=False)
    assert summary["best_model"] in {"prophet", "arima", "sarima", "xgboost"}
    assert summary["best_rmse"] <= summary["all"][summary["best_model"]]["rmse"] + 1e-6
    # Todos os modelos foram avaliados com as 4 métricas.
    for mets in summary["all"].values():
        assert set(mets) == {"mae", "rmse", "mape", "smape"}
