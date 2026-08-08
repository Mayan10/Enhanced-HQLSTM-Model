"""Non-neural baselines used in the comparative study: XGBoost, CatBoost, and ARIMA."""

from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA

from .metrics import calculate_metrics


def flatten_sequences(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Flatten (samples, sequence_length, features) arrays to (samples, sequence_length * features)
    for the tree-based models, which do not use the temporal structure directly."""
    return tuple(a.reshape((a.shape[0], -1)) for a in arrays)


def run_xgboost(X_train, y_train, X_test, y_test, seed: int = 42) -> Tuple[Dict, np.ndarray]:
    X_train_flat, X_test_flat = flatten_sequences(X_train, X_test)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=seed
    )
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    return calculate_metrics(y_test, y_pred), y_pred


def run_catboost(X_train, y_train, X_test, y_test, seed: int = 42) -> Tuple[Dict, np.ndarray]:
    X_train_flat, X_test_flat = flatten_sequences(X_train, X_test)
    model = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05, loss_function="RMSE",
        verbose=False, random_seed=seed,
    )
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    return calculate_metrics(y_test, y_pred), y_pred


def run_arima(y_train, y_val, y_test, order=(2, 1, 2)) -> Tuple[Dict, np.ndarray]:
    """Fit a univariate ARIMA model on train+val targets and forecast the test horizon."""
    y_trainval = np.concatenate([y_train, y_val])
    try:
        fit = ARIMA(y_trainval, order=order).fit()
        y_pred = fit.forecast(steps=len(y_test))
        return calculate_metrics(y_test, y_pred), y_pred
    except Exception as exc:
        print(f"ARIMA failed: {exc}")
        y_pred = np.zeros_like(y_test)
        metrics = {k: float("nan") for k in ["MSE", "RMSE", "MAE", "MBE", "VAF", "R2", "MAPE"]}
        return metrics, y_pred
