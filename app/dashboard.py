"""Dashboard Streamlit para a plataforma de previsão de vendas.

Seções:
    1. KPIs da série histórica.
    2. Série + decomposição visual (histórico e sazonalidade semanal).
    3. Comparação das métricas dos modelos (do backtest).
    4. Previsão futura: escolhe o horizonte, chama a API e plota.

Rodar:
    streamlit run app/dashboard.py
API em ``API_URL`` (default http://localhost:8000).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Permite rodar via `streamlit run app/dashboard.py`: garante a raiz do projeto
# no sys.path (senão o pacote `src` não é encontrado, pois o cwd fica em app/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_synthetic import generate_sales

API_URL = os.getenv("API_URL", "http://localhost:8000")
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")


@st.cache_data
def load_history(days: int = 1277) -> pd.DataFrame:
    return generate_sales(days=days, seed=42)


df = load_history()

st.title("📈 Previsão de Demanda de Vendas")
st.caption("Plataforma de forecasting · varejo/logística · série sintética · modelo servido via API")

# --- 1. KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dias de histórico", f"{len(df):,}")
c2.metric("Venda média/dia", f"{df['sales'].mean():,.0f}")
c3.metric("Pico de vendas", f"{df['sales'].max():,.0f}")
c4.metric("% dias c/ promoção", f"{df['promo'].mean():.1%}")

st.divider()

# --- 2. Série histórica ---
st.subheader("Série histórica de vendas")
fig = px.line(df, x="date", y="sales", labels={"sales": "vendas", "date": "data"})
fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Sazonalidade semanal")
    dow = df.assign(dow=df["date"].dt.day_name()).groupby("dow")["sales"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.bar_chart(dow.reindex(order))
with col_b:
    st.subheader("Sazonalidade anual (por mês)")
    st.bar_chart(df.groupby(df["date"].dt.month)["sales"].mean())

st.divider()

# --- 3. Comparação de modelos (backtest) ---
st.subheader("Comparação dos modelos (backtest temporal)")
if METRICS_PATH.exists():
    summary = json.loads(METRICS_PATH.read_text())
    metrics_df = pd.DataFrame(summary["all"]).T
    st.dataframe(metrics_df.style.highlight_min(axis=0, color="#1b5e20"))
    st.caption(f"Melhor modelo: **{summary['best_model']}** (menor RMSE no holdout de "
               f"{summary['horizon']} dias).")
else:
    st.info("Rode `python -m src.models.backtest` para gerar as métricas.")

st.divider()

# --- 4. Previsão futura (via API) ---
st.subheader("🔮 Previsão futura")
horizon = st.slider("Horizonte (dias)", 7, 180, 30)
if st.button("Prever"):
    try:
        resp = requests.post(f"{API_URL}/forecast",
                             json={"horizon": horizon, "promo_dates": []}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        fc = pd.DataFrame(data["forecast"])
        fc["date"] = pd.to_datetime(fc["date"])

        fig2 = go.Figure()
        hist_tail = df.tail(180)
        fig2.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["sales"],
                                  name="histórico", line=dict(color="#607d8b")))
        fig2.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"],
                                  name=f"previsão ({data['model']})", line=dict(color="#e53935")))
        fig2.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.metric("Venda média prevista/dia", f"{fc['forecast'].mean():,.0f}")
    except requests.RequestException as exc:
        st.warning(f"Não foi possível chamar a API em {API_URL}. Ela está no ar? ({exc})")
