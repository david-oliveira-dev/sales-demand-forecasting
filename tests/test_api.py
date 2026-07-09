"""Testes da API FastAPI (Etapa 6).

Garante que exista um modelo salvo (roda um backtest curto sem LSTM se preciso) e
valida os contratos de /health e /forecast via TestClient.
"""
import pytest
from fastapi.testclient import TestClient

import app.main as api
from src.models.backtest import run_backtest
from src.models.registry import BEST_MODEL_PATH, BEST_META_PATH


@pytest.fixture(scope="module")
def client():
    if not (BEST_MODEL_PATH.exists() and BEST_META_PATH.exists()):
        run_backtest(n_days=220, seed=1, horizon=14, use_lstm=False)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] in {"prophet", "arima", "sarima", "xgboost", "lstm"}


def test_forecast_returns_horizon_points(client):
    resp = client.post("/forecast", json={"horizon": 21, "promo_dates": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == 21
    assert len(body["forecast"]) == 21
    first = body["forecast"][0]
    assert set(first) == {"date", "promo", "holiday", "forecast"}
    assert first["forecast"] >= 0


def test_forecast_dates_are_future_and_sequential(client):
    body = client.post("/forecast", json={"horizon": 5}).json()
    dates = [p["date"] for p in body["forecast"]]
    assert dates == sorted(dates)
    assert len(set(dates)) == 5


def test_forecast_validation_error(client):
    resp = client.post("/forecast", json={"horizon": 0})  # abaixo do mínimo (ge=1)
    assert resp.status_code == 422
