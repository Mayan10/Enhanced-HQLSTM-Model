#!/usr/bin/env python3
"""Minimal end-to-end example: train PVForecastingModel on synthetic PV data.

This does not require the real dataset and runs in well under a minute on CPU,
making it a quick way to confirm the package is installed and working correctly.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hqlstm import PVDataProcessor, PVForecastingModel  # noqa: E402
from hqlstm.device import DEVICE  # noqa: E402
from hqlstm.training import ModelTrainer  # noqa: E402


def synthetic_pv_data(n_samples: int = 500, sequence_length: int = 24, n_features: int = 5, seed: int = 42):
    """Generate a toy dataset with a daily solar irradiance pattern, for demonstration only."""
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        hours = np.arange(sequence_length)
        irradiance = np.maximum(0, np.sin(np.pi * hours / 24) + rng.normal(0, 0.1, sequence_length))
        temperature = 25 + 10 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 2, sequence_length)
        wind_speed = 5 + 3 * rng.random(sequence_length)
        humidity = 60 + 20 * rng.random(sequence_length)
        cloud_cover = rng.random(sequence_length)

        features = np.column_stack([irradiance, temperature, wind_speed, humidity, cloud_cover])
        X.append(features)

        power = np.maximum(0, irradiance * (1 - 0.004 * (temperature - 25)) * (1 - cloud_cover * 0.8))
        y.append(np.mean(power))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    X, y = synthetic_pv_data()

    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    processor = PVDataProcessor()
    X_train_t, y_train_t = processor.fit_transform(X_train, y_train)
    X_val_t, y_val_t = processor.transform(X_val, y_val)
    X_test_t, y_test_t = processor.transform(X_test, y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

    model = PVForecastingModel(input_features=5, hidden_size=32, n_qubits=4)
    trainer = ModelTrainer(model, processor, device=DEVICE, checkpoint_dir="/tmp/hqlstm_example")
    trainer.train(train_loader, val_loader, epochs=10, lr=0.001, patience=5)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
