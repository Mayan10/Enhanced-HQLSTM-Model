import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hqlstm.models import (
    ClassicalLSTMModel,
    GRUModel,
    HybridHeadedModel,
    HybridQuantumLSTM,
    PVForecastingModel,
)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 6
N_FEATURES = 5


def _random_batch():
    return torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, N_FEATURES)


def test_hybrid_quantum_lstm_output_shape():
    model = HybridQuantumLSTM(N_FEATURES, hidden_size=8, n_qubits=2)
    out = model(_random_batch())
    assert out.shape == (BATCH_SIZE, 8)


def test_pv_forecasting_model_output_shape_and_range():
    model = PVForecastingModel(input_features=N_FEATURES, hidden_size=8, n_qubits=2)
    out = model(_random_batch())
    assert out.shape == (BATCH_SIZE, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_hybrid_headed_model_output_shape():
    model = HybridHeadedModel(input_features=N_FEATURES, hidden_size=8, n_qubits=2)
    out = model(_random_batch())
    assert out.shape == (BATCH_SIZE, 1)


def test_classical_lstm_output_shape():
    model = ClassicalLSTMModel(input_features=N_FEATURES, hidden_size=8)
    out = model(_random_batch())
    assert out.shape == (BATCH_SIZE, 1)


def test_gru_output_shape():
    model = GRUModel(input_features=N_FEATURES, hidden_size=8)
    out = model(_random_batch())
    assert out.shape == (BATCH_SIZE, 1)
