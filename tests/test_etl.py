"""Testes do ETL (Etapa 2)."""
import pytest
from sqlalchemy import create_engine

from src.data.etl import clean, load_series, run_etl
from src.data.generate_synthetic import generate_sales


def test_clean_types_and_sorts():
    df = clean(generate_sales(days=200, seed=5))
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["date"].is_monotonic_increasing
    assert df["promo"].dtype == "int8"


def test_clean_rejects_gaps():
    df = generate_sales(days=100, seed=1)
    df = df.drop(index=50).reset_index(drop=True)  # cria uma lacuna de data
    with pytest.raises(ValueError, match="lacunas|diária"):
        clean(df)


def test_clean_rejects_negative_sales():
    df = generate_sales(days=100, seed=1)
    df.loc[0, "sales"] = -10
    with pytest.raises(ValueError, match="negativas"):
        clean(df)


def test_run_etl_and_load(tmp_path):
    raw = tmp_path / "sales.csv"
    parquet = tmp_path / "sales.parquet"
    generate_sales(days=300, seed=2).to_csv(raw, index=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    df = run_etl(raw_csv=raw, parquet_out=parquet, engine=engine)
    assert parquet.exists()

    series = load_series(engine=engine)
    assert len(series) == len(df) == 300
    assert series.index.name == "date"


def test_run_etl_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_etl(raw_csv=tmp_path / "nope.csv", parquet_out=tmp_path / "x.parquet",
                engine=create_engine("sqlite://"))
