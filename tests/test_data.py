import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hqlstm.data import PVDataProcessor, build_sequences


def test_processor_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6, 3)).astype(np.float32)
    y = rng.uniform(0, 100, size=20).astype(np.float32)

    processor = PVDataProcessor()
    X_scaled, y_scaled = processor.fit_transform(X, y)

    assert X_scaled.shape == X.shape
    assert float(y_scaled.min()) >= 0.0
    assert float(y_scaled.max()) <= 1.0

    y_recovered = processor.inverse_transform_target(y_scaled)
    np.testing.assert_allclose(y_recovered, y, atol=1e-4)


def test_processor_transform_requires_fit():
    processor = PVDataProcessor()
    X = np.zeros((2, 3, 2), dtype=np.float32)
    try:
        processor.transform(X)
    except ValueError:
        return
    raise AssertionError("expected ValueError before fit_transform")


def test_build_sequences_sliding_window():
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "measured_on": pd.date_range("2022-01-01", periods=n, freq="min"),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "dc_power__422": np.arange(n, dtype=np.float32),
    })

    X, y = build_sequences(df, sequence_length=5)

    assert X.shape == (15, 5, 2)
    assert y.shape == (15,)
    # target for the first window is the reading right after the window
    assert y[0] == 5.0
    np.testing.assert_allclose(X[0], df[["f1", "f2"]].to_numpy()[:5])
