"""Neural architectures: the hybrid quantum-classical LSTM and the classical baselines used
in the comparative study (LSTM, GRU)."""

import torch
import torch.nn as nn

from .quantum import QuantumLayer


class HybridQuantumLSTM(nn.Module):
    """An LSTM backbone whose final hidden state is projected into a quantum circuit and
    merged back via a residual connection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int = 4,
        n_quantum_layers: int = 1,
        encoding_type: str = "angle",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = min(n_qubits, 8)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.quantum_layer = QuantumLayer(self.n_qubits, n_quantum_layers, encoding_type)
        self.to_quantum = nn.Linear(hidden_size, self.n_qubits)
        self.from_quantum = nn.Linear(self.n_qubits, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        quantum_input = torch.tanh(self.to_quantum(last_hidden))
        quantum_output = self.quantum_layer(quantum_input)
        enhanced_hidden = self.from_quantum(quantum_output)

        return self.layer_norm(last_hidden + enhanced_hidden)


class PVForecastingModel(nn.Module):
    """Full quantum-enhanced forecasting model: input normalization, HybridQuantumLSTM core,
    and a regression head bounded to [0, 1] by the target scaling in PVDataProcessor."""

    def __init__(
        self,
        input_features: int = 5,
        hidden_size: int = 64,
        n_qubits: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_features)
        self.hqlstm = HybridQuantumLSTM(input_features, hidden_size, n_qubits)

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_norm(x)
        enhanced_features = self.hqlstm(x)
        return self.output_layers(enhanced_features)


class HybridHeadedModel(nn.Module):
    """HybridQuantumLSTM with a lighter regression head, used as the "HybridQuantumLSTM"
    entry in the comparative study."""

    def __init__(
        self,
        input_features: int,
        hidden_size: int = 64,
        n_qubits: int = 4,
        n_quantum_layers: int = 1,
        encoding_type: str = "angle",
    ):
        super().__init__()
        self.hybrid = HybridQuantumLSTM(
            input_features, hidden_size, n_qubits, n_quantum_layers, encoding_type
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.hybrid(x)
        return self.output_head(features)


class ClassicalLSTMModel(nn.Module):
    """Plain LSTM baseline with the same output head shape as the quantum models."""

    def __init__(self, input_features: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = self.layer_norm(lstm_out[:, -1, :])
        return self.output_layers(last_hidden)


class GRUModel(nn.Module):
    """GRU baseline with the same output head shape as the quantum models."""

    def __init__(self, input_features: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = self.layer_norm(gru_out[:, -1, :])
        return self.output_layers(last_hidden)
