"""API FastAPI para servir as previsões de vendas.

Endpoints:
    GET  /health    -> status e se há modelo carregado
    POST /forecast  -> previsão dos próximos N dias (com o melhor modelo salvo)

O modelo e os metadados são carregados uma vez no startup (lifespan).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.registry import build_future_frame, load_best

logger = logging.getLogger("sales-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_state: dict = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, meta = load_best()
        _state["model"], _state["meta"] = model, meta
        logger.info("Modelo carregado: %s (última data %s)", meta["best_model"], meta["last_date"])
    except FileNotFoundError:
        logger.warning("Nenhum modelo salvo. Rode `python -m src.models.backtest`.")
    yield


app = FastAPI(
    title="Sales Demand Forecasting API",
    description="Previsão de demanda de vendas com o melhor modelo de série temporal.",
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    horizon: int = Field(30, ge=1, le=365, description="nº de dias a prever", examples=[30])
    promo_dates: list[str] = Field(
        default_factory=list,
        description="datas (YYYY-MM-DD) com promoção planejada",
        examples=[["2024-07-15", "2024-07-16"]],
    )


class ForecastPoint(BaseModel):
    date: str
    promo: int
    holiday: int
    forecast: float


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    forecast: list[ForecastPoint]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None,
            "model": _state["meta"]["best_model"] if _state["meta"] else None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    model, meta = _state["model"], _state["meta"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Rode o backtest.")

    future = build_future_frame(meta["last_date"], req.horizon, req.promo_dates)
    future["forecast"] = model.predict(future).round(0)
    points = [
        ForecastPoint(date=row.date.strftime("%Y-%m-%d"), promo=int(row.promo),
                      holiday=int(row.holiday), forecast=float(row.forecast))
        for row in future.itertuples()
    ]
    return ForecastResponse(model=meta["best_model"], horizon=req.horizon, forecast=points)
