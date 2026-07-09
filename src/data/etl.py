"""ETL: lê a série bruta, valida/tipa e carrega no banco e em parquet.

    data/raw/sales.csv  ->  validação  ->  PostgreSQL (tabela `sales`)
                                           + data/processed/sales.parquet

A conexão vem de ``DATABASE_URL``; sem ela, usa **SQLite** local (fallback para
testes/CI).

Uso:
    python -m src.data.etl
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "sales.csv"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "sales.parquet"
SQLITE_FALLBACK = ROOT / "data" / "processed" / "sales.db"
TABLE = "sales"


def get_engine(database_url: str | None = None) -> Engine:
    """Cria um Engine do SQLAlchemy a partir de ``DATABASE_URL`` (fallback SQLite)."""
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        SQLITE_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{SQLITE_FALLBACK}"
    return create_engine(url)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Valida e tipa a série de vendas, falhando cedo em dados inconsistentes."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")
    for col in ("promo", "holiday"):
        df[col] = df[col].astype("int8")

    df = df.sort_values("date").reset_index(drop=True)

    # Invariantes de série temporal.
    if df["date"].duplicated().any():
        raise ValueError("datas duplicadas na série.")
    gaps = df["date"].diff().dropna().dt.days
    if not (gaps == 1).all():
        raise ValueError("série não é diária contínua (há lacunas de datas).")
    if df["sales"].isna().any():
        raise ValueError("valores de vendas nulos.")
    if (df["sales"] < 0).any():
        raise ValueError("vendas negativas encontradas.")

    return df


def run_etl(
    raw_csv: Path = RAW_CSV,
    parquet_out: Path = PROCESSED_PARQUET,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Executa o ETL completo e retorna o DataFrame limpo."""
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} não existe. Rode antes: python -m src.data.generate_synthetic"
        )

    df = clean(pd.read_csv(raw_csv))

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_out, index=False)

    engine = engine or get_engine()
    df.to_sql(TABLE, engine, if_exists="replace", index=False)

    return df


def load_series(engine: Engine | None = None) -> pd.DataFrame:
    """Lê a série do banco, com `date` como índice (conveniente p/ statsmodels)."""
    engine = engine or get_engine()
    df = pd.read_sql(f"SELECT * FROM {TABLE}", engine, parse_dates=["date"])
    return df.sort_values("date").set_index("date")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL da série de vendas.")
    parser.add_argument("--raw", type=Path, default=RAW_CSV)
    parser.add_argument("--parquet", type=Path, default=PROCESSED_PARQUET)
    args = parser.parse_args()

    engine = get_engine()
    df = run_etl(raw_csv=args.raw, parquet_out=args.parquet, engine=engine)
    print(f"ETL concluído: {len(df)} dias")
    print(f"  -> parquet: {args.parquet}")
    print(f"  -> banco:   {engine.url}")


if __name__ == "__main__":
    main()
