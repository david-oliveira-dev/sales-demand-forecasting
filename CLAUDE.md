# CLAUDE.md — Sales Demand Forecasting Platform

Contexto e convenções para qualquer sessão do Claude Code trabalhando neste repo.

## O que é
Projeto de portfólio (Cientista de Dados Pleno): plataforma ponta a ponta de
**previsão de demanda de vendas** (série temporal). O roteiro completo está em
**`BUILD_BRIEF.md`** — siga-o etapa a etapa.

## Como trabalhar aqui
- Construção **incremental**: uma etapa do BUILD_BRIEF por vez, com commit ao fim.
- Explique decisões técnicas nos commits e no README (é portfólio).
- **Commits sem a linha `Co-Authored-By`.**
- Não comitar dados grandes, modelos pesados nem segredos. Configs por env var.

## Stack
Python 3.12, Pandas/NumPy, statsmodels (ARIMA/SARIMA), Prophet, XGBoost,
TensorFlow-CPU (LSTM), scikit-learn, MLflow, FastAPI, Streamlit, Plotly,
SQLAlchemy + PostgreSQL (fallback SQLite), Docker, pytest.

## Ambiente
- Use `venv`; o `pip` global da máquina do dono é bloqueado (PEP 668).
- Máquina **sem GPU** e com HD mecânico: LSTM roda em CPU, em séries pequenas.
- Testes e CI devem funcionar **sem** PostgreSQL (fallback SQLite via `DATABASE_URL`)
  e **sem** TensorFlow/Prophet quando forem opcionais (imports isolados).

## Regras de série temporal
- **Nunca vazar futuro**: o conjunto de teste é sempre posterior ao de treino
  (backtest temporal, não split aleatório).
- Sempre reportar **múltiplas métricas** (MAE, RMSE, MAPE, sMAPE), não uma só.
