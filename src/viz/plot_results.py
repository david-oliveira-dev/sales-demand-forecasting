"""Gera as figuras de resultado usadas no README.

Reproduz o mesmo split temporal do backtest (mesma semente, mesmo horizonte),
treina o modelo campeão e salva:

- `reports/figures/forecast_vs_real.png` — previsão contra o realizado nos 90
  dias de teste, que é o resultado que importa num projeto de séries temporais;
- `reports/figures/comparacao_modelos.png` — erro de cada família de modelos,
  lido de `reports/metrics.json` (não retreina: os números vêm do backtest).

Uso:
    python -m src.viz.plot_results
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem display: roda em servidor e no CI
import matplotlib.pyplot as plt  # noqa: E402

from src.data.generate_synthetic import generate_sales  # noqa: E402
from src.models.backtest import temporal_split  # noqa: E402
from src.models.prophet_model import ProphetModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("plot_results")

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"

HORIZONTE = 90
DIAS = 1277
SEED = 42


def plot_forecast_vs_real() -> Path:
    """Previsão do campeão contra o realizado, no holdout temporal."""
    df = generate_sales(days=DIAS, seed=SEED)
    train, test = temporal_split(df, HORIZONTE)

    modelo = ProphetModel().fit(train)
    previsto = modelo.predict(test)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    # Contexto: os últimos 180 dias de treino, para a previsão não ficar solta.
    contexto = train.tail(180)
    ax.plot(contexto["date"], contexto["sales"], color="#8c8c8c", lw=1,
            label="histórico (treino)")
    ax.plot(test["date"], test["sales"], color="#4c72b0", lw=1.6, label="realizado")
    ax.plot(test["date"], previsto, color="#c44e52", lw=1.6, ls="--",
            label="previsto (Prophet)")
    ax.axvline(test["date"].iloc[0], color="k", lw=1, ls=":")
    ax.text(test["date"].iloc[0], ax.get_ylim()[1], "  início do teste",
            va="top", fontsize=9)

    ax.set_title("Previsão vs. realizado — holdout temporal de 90 dias "
                 "(treino sempre anterior ao teste)")
    ax.set_xlabel("data"); ax.set_ylabel("vendas diárias")
    ax.legend(loc="upper left")
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "forecast_vs_real.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    logger.info("Salvo %s", destino.name)
    return destino


def plot_comparacao_modelos() -> Path:
    """Erro por família de modelo, a partir das métricas já apuradas."""
    metricas = json.loads(METRICS.read_text())["all"]
    nomes = sorted(metricas, key=lambda n: metricas[n]["rmse"])
    campeao = min(metricas, key=lambda n: metricas[n]["rmse"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for eixo, chave, rotulo in [(ax[0], "rmse", "RMSE"), (ax[1], "mape", "MAPE (%)")]:
        valores = [metricas[n][chave] for n in nomes]
        cores = ["#c44e52" if n == campeao else "#4c72b0" for n in nomes]
        eixo.bar(nomes, valores, color=cores)
        eixo.set_title(f"{rotulo} por modelo (menor é melhor)")
        eixo.set_ylabel(rotulo)
        eixo.tick_params(axis="x", rotation=20)
        for i, v in enumerate(valores):
            eixo.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Backtest: o clássico venceu o deep learning nesta série", fontsize=12)
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "comparacao_modelos.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    logger.info("Salvo %s", destino.name)
    return destino


def main() -> None:
    plot_comparacao_modelos()
    plot_forecast_vs_real()


if __name__ == "__main__":
    main()
