"""Gerador de série temporal sintética de vendas de varejo.

Em vez de ruído aleatório, a série é montada a partir de **componentes
interpretáveis** — tendência, sazonalidade semanal e anual, efeito de feriados e
de promoções — mais um ruído multiplicativo. Assim os modelos de série temporal
(Prophet, ARIMA/SARIMA, XGBoost, LSTM) têm sinal real para aprender e os
resultados fazem sentido de negócio.

Uso:
    python -m src.data.generate_synthetic --start 2021-01-01 --days 1277 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Fatores multiplicativos por dia da semana (seg=0 ... dom=6): varejo vende mais
# no fim de semana.
WEEKDAY_FACTORS = np.array([0.95, 0.93, 0.96, 1.00, 1.15, 1.35, 1.20])

BASE_SALES = 1000.0        # nível médio de vendas/dia
TREND_PER_YEAR = 0.08      # crescimento anual (~8%)
PROMO_PROB = 0.10          # ~10% dos dias têm promoção
PROMO_UPLIFT = 0.35        # promoção aumenta as vendas em ~35%
HOLIDAY_UPLIFT = 0.25      # feriado aumenta ~25% (compras antecipadas)


def _brazil_holidays(years: range):
    """Conjunto de feriados nacionais do Brasil para os anos dados."""
    import holidays

    return holidays.Brazil(years=list(years))


def generate_sales(start: str = "2021-01-01", days: int = 1277, seed: int = 42) -> pd.DataFrame:
    """Gera uma série diária de vendas com componentes realistas.

    Retorna DataFrame com colunas: date, sales, promo, holiday.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    t = np.arange(days)

    # --- Tendência (crescimento suave) ---
    trend = 1.0 + TREND_PER_YEAR * (t / 365.25)

    # --- Sazonalidade anual (pico no fim de ano, vale em jan/fev) ---
    doy = dates.dayofyear.to_numpy()
    yearly = (
        1.0
        + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365.25)   # ciclo anual
        + 0.12 * np.cos(2 * np.pi * (doy - 20) / 365.25)   # 2º harmônico
    )
    # Reforço explícito de novembro/dezembro (Black Friday + Natal).
    yearly += np.where(np.isin(dates.month, [11, 12]), 0.15, 0.0)

    # --- Sazonalidade semanal ---
    weekly = WEEKDAY_FACTORS[dates.dayofweek.to_numpy()]

    # --- Feriados ---
    br_holidays = _brazil_holidays(range(dates.year.min(), dates.year.max() + 1))
    holiday_flag = np.array([1 if d.date() in br_holidays else 0 for d in dates])

    # --- Promoções (eventos aleatórios de marketing) ---
    promo_flag = rng.binomial(1, PROMO_PROB, size=days)

    # --- Ruído multiplicativo ---
    noise = rng.normal(1.0, 0.05, size=days)

    sales = (
        BASE_SALES
        * trend
        * yearly
        * weekly
        * (1.0 + PROMO_UPLIFT * promo_flag)
        * (1.0 + HOLIDAY_UPLIFT * holiday_flag)
        * noise
    )
    sales = np.round(sales).clip(min=0).astype(int)

    return pd.DataFrame({
        "date": dates,
        "sales": sales,
        "promo": promo_flag.astype(int),
        "holiday": holiday_flag.astype(int),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera série sintética de vendas.")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--days", type=int, default=1277, help="nº de dias (~3,5 anos)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RAW_DIR / "sales.csv")
    args = parser.parse_args()

    df = generate_sales(start=args.start, days=args.days, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Gerados {len(df)} dias de vendas -> {args.out}")
    print(f"Período: {df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Vendas média/dia: {df['sales'].mean():.0f} | promo em {df['promo'].mean():.1%} dos dias")


if __name__ == "__main__":
    main()
