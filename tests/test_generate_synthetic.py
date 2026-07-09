"""Testes do gerador de série temporal de vendas (Etapa 1)."""
import pandas as pd

from src.data.generate_synthetic import generate_sales

EXPECTED_COLUMNS = {"date", "sales", "promo", "holiday"}


def test_shape_and_columns():
    df = generate_sales(days=400, seed=1)
    assert len(df) == 400
    assert set(df.columns) == EXPECTED_COLUMNS
    assert df["sales"].min() >= 0


def test_dates_are_continuous_daily():
    df = generate_sales(days=300, seed=2)
    diffs = df["date"].diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(days=1)


def test_reproducible_with_seed():
    a = generate_sales(days=200, seed=7)
    b = generate_sales(days=200, seed=7)
    assert a.equals(b)


def test_weekly_seasonality_present():
    # Sanidade: fim de semana (sáb/dom) deve vender mais que a média de dias úteis.
    df = generate_sales(days=730, seed=3)
    dow = df["date"].dt.dayofweek
    weekend = df.loc[dow >= 5, "sales"].mean()
    weekday = df.loc[dow < 5, "sales"].mean()
    assert weekend > weekday


def test_promo_boosts_sales():
    # Em média, dias de promoção vendem mais que dias sem promoção.
    df = generate_sales(days=1277, seed=42)
    assert df.loc[df["promo"] == 1, "sales"].mean() > df.loc[df["promo"] == 0, "sales"].mean()
