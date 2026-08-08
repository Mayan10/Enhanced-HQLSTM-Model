"""Data loading, cleaning, sequence construction, and scaling for the PV forecasting task."""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET_COLUMN = "dc_power__422"
TIMESTAMP_COLUMN = "measured_on"


class PVDataProcessor:
    """Fits a StandardScaler on input features and a MinMaxScaler on the target, and
    converts between raw and scaled tensors."""

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_scaled = self.feature_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.fitted = True
        return (
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")

        X_scaled = self.feature_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        if y is not None:
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
            return X_tensor, torch.tensor(y_scaled, dtype=torch.float32)

        return X_tensor

    def inverse_transform_target(self, y_scaled) -> np.ndarray:
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def _find_csv_files(raw_dir: str, years: List[str]) -> List[str]:
    files = []
    for year in years:
        year_path = os.path.join(raw_dir, year)
        if not os.path.exists(year_path):
            continue
        for root, _dirs, filenames in os.walk(year_path):
            for name in filenames:
                if name.endswith(".csv"):
                    files.append(os.path.join(root, name))
    return sorted(files)


def load_raw_data(raw_dir: str = "data/raw", years: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and concatenate all per-day CSV files under `raw_dir/<year>/`, sorted by timestamp."""
    years = years or ["2022", "2023"]
    files = _find_csv_files(raw_dir, years)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir} for years {years}")

    data = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    if TIMESTAMP_COLUMN in data.columns:
        data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values and coerce feature/target columns to numeric."""
    data = data.dropna()
    feature_columns = [c for c in data.columns if c not in (TIMESTAMP_COLUMN, TARGET_COLUMN)]

    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

    return data.dropna(subset=feature_columns + [TARGET_COLUMN])


def build_sequences(
    data: pd.DataFrame, sequence_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences of length `sequence_length` predicting the next-step target."""
    feature_columns = [c for c in data.columns if c not in (TIMESTAMP_COLUMN, TARGET_COLUMN)]

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i : i + sequence_length][feature_columns].values.astype(np.float32))
        y.append(np.float32(data.iloc[i + sequence_length][TARGET_COLUMN]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_test_split(
    X: np.ndarray, y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
):
    """Shuffle and split sequences into train/val/test sets."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(train_frac * n_samples)
    val_size = int(val_frac * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def load_pv_dataset(
    raw_dir: str = "data/raw",
    sequence_length: int = 24,
    seed: int = 42,
    max_samples: Optional[int] = None,
):
    """End-to-end loading: raw CSVs to cleaned, sequenced, and split train/val/test arrays.

    If `max_samples` is set, a random subset of that many sequences is used instead of the
    full dataset. Useful for a quick end-to-end run before committing to full-scale training.
    """
    data = clean_data(load_raw_data(raw_dir))
    X, y = build_sequences(data, sequence_length)

    if max_samples is not None and max_samples < len(X):
        rng = np.random.default_rng(seed)
        subset = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[subset], y[subset]

    return train_val_test_split(X, y, seed=seed)
