import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hqlstm.metrics import calculate_metrics, to_serializable


def test_calculate_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = calculate_metrics(y, y)
    assert metrics["MSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["VAF"] == 1.0
    assert metrics["R2"] == 1.0


def test_calculate_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.25
    assert metrics["MBE"] == 0.25


def test_to_serializable_handles_numpy_types():
    payload = {
        "array": np.array([1, 2, 3]),
        "scalar": np.float32(1.5),
        "nested": [np.int64(2), {"x": np.float64(3.0)}],
    }
    result = to_serializable(payload)
    assert result == {"array": [1, 2, 3], "scalar": 1.5, "nested": [2, {"x": 3.0}]}
