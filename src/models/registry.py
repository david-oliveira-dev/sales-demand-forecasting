"""Registro dos modelos disponíveis e utilitários de serving.

Centraliza o mapeamento nome -> classe e a lógica de montar o quadro de
regressores futuros (datas + promo + holiday), usada tanto pela API quanto pelo
dashboard. Importar este módulo NÃO carrega TensorFlow/Prophet/etc — cada classe
importa sua dependência pesada só quando usada.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.arima_model import ArimaModel, SarimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgb_model import XGBModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "model.joblib"
BEST_META_PATH = MODELS_DIR / "best_model.json"

MODEL_CLASSES = {
    "prophet": ProphetModel,
    "arima": ArimaModel,
    "sarima": SarimaModel,
    "xgboost": XGBModel,
}


def _model_class(name: str):
    if name == "lstm":
        from src.models.lstm_model import LSTMModel

        return LSTMModel
    return MODEL_CLASSES[name]


def load_best(meta_path: Path = BEST_META_PATH, model_path: Path = BEST_MODEL_PATH):
    """Carrega o melhor modelo salvo + metadados. Levanta se não existir."""
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Rode antes: python -m src.models.backtest"
        )
    meta = json.loads(meta_path.read_text())
    model = _model_class(meta["best_model"]).load(model_path)
    return model, meta


def build_future_frame(
    last_date: str, horizon: int, promo_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Monta o quadro de regressores futuros: date, promo, holiday.

    `holiday` vem do calendário nacional (Brasil); `promo` é 1 nas datas
    informadas em `promo_dates` (padrão: nenhuma promoção futura).
    """
    import holidays

    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")
    br = holidays.Brazil(years=list(range(dates.year.min(), dates.year.max() + 1)))
    promo_set = {pd.to_datetime(d).date() for d in (promo_dates or [])}
    return pd.DataFrame({
        "date": dates,
        "promo": [1 if d.date() in promo_set else 0 for d in dates],
        "holiday": [1 if d.date() in br else 0 for d in dates],
    })


def forecast(horizon: int, promo_dates: list[str] | None = None) -> pd.DataFrame:
    """Carrega o melhor modelo e devolve a previsão dos próximos `horizon` dias."""
    model, meta = load_best()
    future = build_future_frame(meta["last_date"], horizon, promo_dates)
    future["forecast"] = model.predict(future).round(0)
    return future[["date", "promo", "holiday", "forecast"]]
