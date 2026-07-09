"""Backtest e comparação dos modelos de previsão de vendas.

Estratégia de avaliação: **holdout temporal** — treina em todo o histórico menos
os últimos `horizon` dias e prevê exatamente esse período (que nunca foi visto no
treino). Isso respeita a causalidade da série (o teste é sempre futuro em relação
ao treino), ao contrário de um split aleatório.

Cada modelo é avaliado com MAE/RMSE/MAPE/sMAPE, registrado no MLflow, e o melhor
(por RMSE) é re-treinado em todo o histórico e salvo para servir na API.

Uso:
    python -m src.models.backtest --horizon 90
    python -m src.models.backtest --horizon 90 --no-lstm   # pula o LSTM (rápido)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.data.generate_synthetic import generate_sales
from src.models.arima_model import ArimaModel, SarimaModel
from src.models.metrics import all_metrics
from src.models.prophet_model import ProphetModel
from src.models.xgb_model import XGBModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
METRICS_PATH = REPORTS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "model.joblib"
BEST_META_PATH = MODELS_DIR / "best_model.json"

EXPERIMENT = "sales-forecasting"


def temporal_split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide a série: tudo menos os últimos `horizon` dias vs. esses dias."""
    train = df.iloc[:-horizon].reset_index(drop=True)
    test = df.iloc[-horizon:].reset_index(drop=True)
    return train, test


def build_models(use_lstm: bool) -> list:
    models = [ProphetModel(), ArimaModel(), SarimaModel(), XGBModel()]
    if use_lstm:
        from src.models.lstm_model import LSTMModel

        models.append(LSTMModel())
    return models


def run_backtest(n_days: int = 1277, seed: int = 42, horizon: int = 90,
                 use_lstm: bool = True) -> dict:
    """Executa o backtest de todos os modelos e salva o melhor."""
    df = generate_sales(days=n_days, seed=seed)
    train, test = temporal_split(df, horizon)
    y_true = test["sales"].to_numpy()

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results: dict[str, dict[str, float]] = {}
    best_name, best_rmse = None, float("inf")

    for model in build_models(use_lstm):
        with mlflow.start_run(run_name=model.name):
            model.fit(train)
            y_pred = model.predict(test)
            metrics = all_metrics(y_true, y_pred)
            mlflow.log_param("model", model.name)
            mlflow.log_param("horizon", horizon)
            mlflow.log_metrics(metrics)
            results[model.name] = metrics
            print(f"{model.name:>10}: " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
            if metrics["rmse"] < best_rmse:
                best_name, best_rmse = model.name, metrics["rmse"]

    # Re-treina o melhor modelo em TODO o histórico e salva para a API.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = {m.name: m for m in build_models(use_lstm)}[best_name]
    best.fit(df)
    best.save(BEST_MODEL_PATH)
    BEST_META_PATH.write_text(json.dumps({
        "best_model": best_name,
        "horizon": horizon,
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
    }))

    summary = {"best_model": best_name, "best_rmse": best_rmse,
               "horizon": horizon, "all": results}
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nMelhor modelo: {best_name} (RMSE={best_rmse}) -> {BEST_MODEL_PATH}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest de modelos de previsão de vendas.")
    parser.add_argument("--days", type=int, default=1277)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--no-lstm", action="store_true", help="pula o LSTM (mais rápido)")
    args = parser.parse_args()
    run_backtest(n_days=args.days, seed=args.seed, horizon=args.horizon,
                 use_lstm=not args.no_lstm)


if __name__ == "__main__":
    main()
