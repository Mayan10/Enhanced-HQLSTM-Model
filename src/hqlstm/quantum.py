"""Quantum feature encoding and the parameterized quantum circuit layer."""

import warnings
from typing import List

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class QuantumFeatureMap:
    """Encodes a classical feature vector into rotation angles on a qubit register."""

    def __init__(self, n_qubits: int, encoding_type: str = "angle"):
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def angle_encoding(self, x: torch.Tensor, wires: List[int]):
        """Map each feature to [-pi, pi] and apply it as an RY rotation."""
        x = x.cpu()
        for i, wire in enumerate(wires):
            if i < len(x):
                angle = x[i].item() * np.pi
                qml.RY(angle, wires=wire)

    def fourier_encoding(self, x: torch.Tensor, wires: List[int]):
        """Encode each feature at two fixed frequencies via RZ rotations."""
        x = x.cpu()
        frequencies = [1.0, 2.0]
        for i, wire in enumerate(wires):
            if i < len(x):
                for freq in frequencies:
                    qml.RZ(freq * x[i].item(), wires=wire)

    def __call__(self, x: torch.Tensor) -> None:
        wires = list(range(self.n_qubits))
        x = torch.clamp(x, -1.0, 1.0)
        if self.encoding_type == "fourier":
            self.fourier_encoding(x, wires)
        else:
            self.angle_encoding(x, wires)


class QuantumLayer(nn.Module):
    """A parameterized quantum circuit: data encoding, trainable rotations, entanglement, measurement."""

    def __init__(self, n_qubits: int, n_layers: int = 1, encoding_type: str = "angle"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.feature_map = QuantumFeatureMap(n_qubits, encoding_type)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")

    def quantum_circuit(self, inputs, theta):
        inputs = inputs.cpu()
        theta = theta.cpu()

        self.feature_map(inputs)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(
                    theta[layer, i, 0].item(),
                    theta[layer, i, 1].item(),
                    theta[layer, i, 2].item(),
                    wires=i,
                )
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the circuit per sample, falling back to a classical projection on failure."""
        input_device = x.device
        try:
            outputs = []
            for i in range(x.shape[0]):
                sample = x[i]
                try:
                    outputs.append(self.qnode(sample, self.theta))
                except Exception as exc:
                    warnings.warn(f"Quantum computation failed for sample {i}: {exc}")
                    outputs.append(torch.tanh(sample[: self.n_qubits]))

            if isinstance(outputs[0], list):
                outputs = torch.tensor(outputs, dtype=torch.float32)
            else:
                outputs = torch.stack(outputs)

            return outputs.to(input_device)
        except Exception as exc:
            warnings.warn(f"Forward pass failed: {exc}")
            return torch.tanh(x[:, : self.n_qubits])
