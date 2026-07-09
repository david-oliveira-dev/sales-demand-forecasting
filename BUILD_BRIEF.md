# Build Brief — Sales Demand Forecasting Platform

> Roteiro de construção do projeto. Cada etapa é implementada na ordem, com
> commit ao final. Qualidade de **portfólio para Cientista de Dados Pleno**.

## Objetivo
Plataforma ponta a ponta de **previsão de demanda/vendas** (série temporal
diária de um varejo), pronta para produção: dados → ETL → features → modelos
(clássicos + ML + deep learning) → API → dashboard → monitoramento.

Público-alvo: **varejo e logística** (planejamento de estoque e compras).

## Decisões já tomadas (não reabrir)
- **Dados:** sintéticos, gerados por código. Série diária de vendas com
  tendência, sazonalidade semanal e anual, efeito de feriados e promoções, e
  ruído — para que os modelos tenham sinal real de série temporal para aprender.
- **Modelos:** comparar **Prophet, ARIMA, SARIMA, XGBoost e LSTM** num
  backtest com holdout temporal. Métricas: **MAE, RMSE, MAPE, sMAPE**.
- **Dashboard:** Streamlit (histórico + previsão + comparação de modelos).
- **Banco:** PostgreSQL (SQLAlchemy; fallback SQLite p/ testes/CI).
- **Tracking:** MLflow (backend SQLite — file store foi aposentado no MLflow 3.x).
- **Deep learning:** LSTM em Keras/TensorFlow-CPU (máquina sem GPU); import
  isolado para não quebrar testes/CI se o extra faltar.

## Etapas (construção incremental — commit ao fim de cada uma)

### Etapa 1 — Geração de dados sintéticos
- `src/data/generate_synthetic.py`: série diária de vendas (~3,5 anos) com
  tendência + sazonalidade semanal/anual + feriados + promoções + ruído.
  Colunas: `date, sales, promo, holiday`. Salva `data/raw/sales.csv`.
- Teste: shape, continuidade das datas, sazonalidade semanal detectável,
  reprodutibilidade (seed).

### Etapa 2 — ETL + carga no banco
- `src/data/etl.py`: lê `data/raw`, valida (datas contínuas, sem nulos, sales≥0),
  grava em PostgreSQL (tabela `sales`) e `data/processed/sales.parquet`.
- Conexão via `DATABASE_URL` (fallback SQLite).

### Etapa 3 — EDA
- `notebooks/01-eda.ipynb`: decomposição sazonal (trend/seasonal/resid), ACF/PACF,
  vendas por dia da semana e mês, efeito de promoções, insights de negócio.

### Etapa 4 — Feature Engineering
- `src/features/build_features.py`: features de calendário (dow, mês, semana do
  ano), lags (1,7,14,28), médias móveis, termos de Fourier para sazonalidade
  anual. Usadas pelo XGBoost e como janelas do LSTM.

### Etapa 5 — Treino, backtest e comparação
- `src/models/`: um módulo por família (`prophet_model.py`, `arima_model.py`,
  `xgb_model.py`, `lstm_model.py`) + `backtest.py` que avalia todos no mesmo
  holdout temporal (MAE/RMSE/MAPE/sMAPE), registra no MLflow e salva o melhor.

### Etapa 6 — API FastAPI
- `app/main.py`: `/health` e `/forecast?horizon=N` (devolve previsão dos
  próximos N dias com o melhor modelo). Carrega artefatos salvos.

### Etapa 7 — Dashboard Streamlit
- `app/dashboard.py`: histórico, previsão futura, comparação de métricas dos
  modelos, e um seletor de horizonte que chama a API.

### Etapa 8 — Docker + Compose
- `Dockerfile` (API) e `docker-compose.yml` (API + Postgres + dashboard).

### Etapa 9 — Testes, CI, logs, docs
- Cobertura nas peças críticas; CI verde no GitHub Actions; logging estruturado.
- `README.md` profissional + `reports/ARCHITECTURE.md` (arquitetura, fluxograma,
  resultados, trade-offs, melhorias futuras).

## Padrões
- Código modular e tipado; funções pequenas; docstrings.
- Nada de segredos no repo; configs por env var.
- Métricas de série temporal (nunca só uma). Backtest temporal, sem vazamento
  de futuro (o teste é sempre posterior ao treino).
- Commits pequenos, um por etapa. **Sem linha de Co-Authored-By.**
