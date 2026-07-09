"""Testes de feature engineering (Etapa 4)."""
import numpy as np

from src.data.generate_synthetic import generate_sales
from src.features.build_features import (
    add_calendar_features, add_fourier_terms, build_supervised, make_sequences,
)


def test_calendar_and_fourier_features():
    df = add_fourier_terms(add_calendar_features(generate_sales(days=60, seed=1)))
    for col in ("dayofweek", "month", "is_weekend", "sin_1", "cos_1"):
        assert col in df.columns
    assert df["dayofweek"].between(0, 6).all()


def test_build_supervised_no_nan_and_no_leakage():
    X, y = build_supervised(generate_sales(days=200, seed=2))
    # Lags criam NaN nas primeiras linhas; devem ser descartadas.
    assert not X.isna().any().any()
    assert len(X) == len(y)
    # O alvo não pode estar entre as features.
    assert "sales" not in X.columns


def test_make_sequences_shapes():
    values = np.arange(100, dtype=float)
    X, y = make_sequences(values, window=10)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)
    # A janela seguinte prevê o próximo valor.
    assert y[0] == values[10]
