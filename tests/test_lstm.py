"""Teste mínimo do LSTM (Etapa 5) — config enxuta para rodar rápido.

Verifica o caminho fit -> predict -> save -> load. TensorFlow é pesado; por isso
usamos poucos épocas e janela pequena.
"""
from src.data.generate_synthetic import generate_sales
from src.models.backtest import temporal_split
from src.models.lstm_model import LSTMModel


def test_lstm_fit_predict_save_load(tmp_path):
    train, test = temporal_split(generate_sales(days=200, seed=1), 10)
    model = LSTMModel(window=14, units=8, epochs=1)
    model.fit(train)

    preds = model.predict(test)
    assert len(preds) == 10
    assert (preds >= 0).all()

    path = tmp_path / "lstm.joblib"
    model.save(path)
    reloaded = LSTMModel.load(path)
    preds2 = reloaded.predict(test)
    assert len(preds2) == 10
