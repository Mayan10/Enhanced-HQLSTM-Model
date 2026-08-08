"""Evaluation metrics and JSON serialization helpers."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """MSE, RMSE, MAE, MBE, VAF, R2, and MAPE between true and predicted values."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    if ss_tot == 0:
        vaf = 1.0 if ss_res == 0 else 0.0
    else:
        vaf = max(0.0, 1 - (ss_res / ss_tot))
    r2 = vaf

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = (
            np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
            * 100
        )
    else:
        mape = float("inf")

    mbe = np.mean(y_pred - y_true)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MBE": mbe,
        "VAF": vaf,
        "R2": r2,
        "MAPE": mape,
    }


def to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj
