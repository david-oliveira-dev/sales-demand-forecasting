# Arquitetura & Relatório Técnico — Sales Demand Forecasting

## 1. Visão geral
Pipeline reprodutível de **previsão de demanda**: dados → ETL → features →
backtest de 5 modelos → melhor modelo servido em API + dashboard. Cada etapa é um
módulo isolado e testável.

## 2. Fluxograma
```
          ┌────────────────────┐
          │ generate_synthetic │  série diária: tendência + sazonalidade
          │  (componentes)     │  semanal/anual + feriados + promoções + ruído
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐   valida datas contínuas, sales>=0, sem nulos
          │        ETL         │──▶ PostgreSQL (tabela sales) + parquet
          └─────────┬──────────┘   (fallback SQLite via DATABASE_URL)
                    ▼
          ┌────────────────────┐   calendário, lags (1/7/14/28), médias móveis,
          │ Feature Engineering│   termos de Fourier, exógenas (promo/holiday)
          └─────────┬──────────┘
                    ▼
          ┌──────────────────────────────────────────────┐
          │        Backtest temporal (holdout)           │
          │  treino = [0 .. T-H)   teste = [T-H .. T)     │
          │  ┌─────────┬───────┬────────┬────────┬──────┐ │
          │  │ Prophet │ ARIMA │ SARIMA │XGBoost │ LSTM │ │
          │  └─────────┴───────┴────────┴────────┴──────┘ │
          │  métricas: MAE · RMSE · MAPE · sMAPE  → MLflow │
          └─────────┬────────────────────────────────────┘
                    ▼  seleciona menor RMSE, re-treina em todo o histórico
          ┌────────────────────┐
          │ model.joblib + meta│
          └─────────┬──────────┘
             ┌───────┴────────┐
             ▼                ▼
     ┌──────────────┐  ┌────────────────────┐
     │  API FastAPI │  │ Dashboard Streamlit│
     │  /forecast   │  │ histórico+previsão │
     └──────────────┘  └────────────────────┘
```

## 3. Dados
Série **sintética** diária (~3,5 anos) gerada por código, montada a partir de
componentes interpretáveis: nível base, tendência de crescimento (~8%/ano),
sazonalidade anual (pico nov/dez), sazonalidade semanal (fim de semana mais
forte), efeito de feriados nacionais (Brasil) e de promoções, e ruído
multiplicativo. Colunas: `date, sales, promo, holiday`.

## 4. Feature Engineering
- **Calendário**: dia da semana, mês, trimestre, dia do ano, semana do ano,
  flag de fim de semana.
- **Fourier** (ordem 3): sazonalidade anual suave para o XGBoost.
- **Lags** (1, 7, 14, 28) e **médias/desvios móveis** (7, 14, 28) — sempre com
  `shift` para não usar o próprio dia (sem vazamento).
- **Exógenas**: `promo` e `holiday`, conhecidas no futuro.
- **LSTM**: janelas deslizantes da série normalizada (min-max).

## 5. Modelos e método de avaliação
Cinco famílias comparadas no **mesmo holdout temporal** (últimos H dias):

| Família | Papel | Exógenas | Multi-passo |
|---|---|---|---|
| ARIMA(2,1,2) | baseline clássico | não | direto |
| SARIMA(1,1,1)(1,1,1,7) | sazonalidade semanal | sim (SARIMAX) | direto |
| Prophet | sazonalidades múltiplas + feriados | sim (regressores) | direto |
| XGBoost | features/exógenas, não-linear | sim | recursivo |
| LSTM | padrões não-lineares sequenciais | não (univariado) | recursivo |

**Sem vazamento de futuro**: o teste é sempre posterior ao treino; previsões
multi-passo (XGBoost/LSTM) são recursivas, realimentando as próprias predições.

Métricas reportadas para todos: **MAE, RMSE, MAPE, sMAPE**. Seleção do melhor por
**RMSE** (penaliza os grandes erros, que são os mais caros em estoque). Todos os
runs são registrados no **MLflow** (backend SQLite).

### Resultados (holdout de 90 dias)

| Modelo | MAE | RMSE | MAPE (%) | sMAPE (%) |
|---|---|---|---|---|
| **Prophet** ✅ | **61.0** | **80.1** | **3.9** | **3.9** |
| SARIMA | 64.1 | 87.6 | 4.0 | 4.0 |
| XGBoost | 107.8 | 137.5 | 6.6 | 7.0 |
| ARIMA | 227.4 | 287.2 | 13.6 | 14.6 |
| LSTM | 433.5 | 508.3 | 27.7 | 29.7 |

Leitura: Prophet e SARIMA lideram (MAPE ~4%) por capturarem sazonalidade semanal
+ anual e os regressores exógenos. ARIMA (sem sazonalidade) e LSTM (univariado,
recursivo em 90 passos) ficam atrás — resultado esperado e coerente com o método.

## 6. Serviço
- **API FastAPI**: `/health` e `/forecast` (horizonte N + datas de promoção
  planejadas → previsão diária). Modelo carregado no `lifespan`; os regressores
  futuros (feriado pelo calendário, promo pelo pedido) são montados no serving.
- **Dashboard Streamlit + Plotly**: histórico, sazonalidades, tabela comparativa
  dos modelos e previsão futura interativa que consome a API.

## 7. Qualidade & Operação
- **Testes** (pytest): gerador, ETL (validações), features (sem vazamento),
  métricas, cada modelo, backtest e API. LSTM testado com config enxuta.
- **CI** (GitHub Actions): instala dependências + roda a suíte a cada push/PR.
- **Docker Compose**: Postgres + API + dashboard, com healthchecks.
- **Logging** estruturado na API; configs por env var (`DATABASE_URL`,
  `MLFLOW_TRACKING_URI`, `API_URL`).

## 8. Limitações e próximos passos
- Série única (agregada); um passo natural é modelar por loja/SKU.
- LSTM univariado; incluir exógenas na rede é melhoria direta.
- Sem intervalos de predição ainda — calibração de incerteza é prioridade para
  decisão de estoque.
- Monitoramento de drift e re-treino agendado ficam como Etapa 10.
