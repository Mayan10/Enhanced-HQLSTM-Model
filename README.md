# Enhanced HQLSTM: Hybrid Quantum-Classical LSTM for Solar PV Forecasting

A benchmark for short-horizon solar photovoltaic (PV) power forecasting that compares a
hybrid quantum-classical LSTM against classical deep learning and gradient-boosted tree
baselines on real inverter telemetry.

This repository accompanies the paper:

> Mayan Sharma, "Enhanced Hybrid Quantum-LSTM for Solar PV Forecasting," IET BICET 2025.
> DOI: [10.1049/icp.2026.1046](https://digital-library.theiet.org/doi/10.1049/icp.2026.1046)

## Overview

The core model, `PVForecastingModel`, wraps an LSTM with a small parameterized quantum
circuit: the LSTM's final hidden state is projected onto a handful of qubits, processed by
a trainable circuit (data encoding, learnable rotations, nearest-neighbor entanglement, and
Pauli-Z measurement), and merged back into the hidden state with a residual connection. The
quantum circuit is simulated with [PennyLane](https://pennylane.ai) and trained end-to-end
with the rest of the network through PennyLane's PyTorch interface.

The benchmark trains this model, a lighter-head variant (`HybridHeadedModel`), and classical
LSTM/GRU baselines, then compares all of them against XGBoost, CatBoost, and ARIMA on the
same held-out test set using RMSE, MAE, MBE, VAF, R2, and MAPE.

## Repository layout

```
src/hqlstm/          Package: models, quantum circuit, data pipeline, training, plotting
scripts/
  run_benchmark.py   Full pipeline: load data, train or load checkpoints, evaluate, plot
  plot_history.py    Regenerate plots from saved history/*.json without retraining
examples/
  basic_usage.py     Self-contained example on synthetic data (no dataset required)
models/              Text dumps of the architectures explored at different capacities
history/             Saved training histories (loss and metrics per epoch) as JSON
data/                Expected raw data layout (data itself is not tracked, see data/README.md)
tests/               Unit tests for the model, data, and metrics modules
docs/assets/         Result plots referenced in this README
```

`checkpoints/` and `plots/` are created when you run the pipeline and are not tracked in
version control.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or install the package directly:

```bash
pip install -e .
```

Python 3.10+ is required.

## Usage

Run the full comparative study. Place raw data under `data/raw/<year>/` first (see
`data/README.md`):

```bash
python scripts/run_benchmark.py
```

By default this loads any existing checkpoint for each neural model instead of retraining
it. Pass `--retrain` to force training from scratch, and see `--help` for epochs, learning
rate, patience, and sequence length options:

```bash
python scripts/run_benchmark.py --retrain --epochs 75 --lr 0.001
```

Results are written to `plots/` (training curves, prediction scatter plots, residuals,
error boxplots, per-metric comparisons) and `history/` (per-epoch metrics as JSON).
Model checkpoints are written to `checkpoints/<ModelName>_best.pth`.

To regenerate the comparison plots from existing `history/*.json` files without touching
the models:

```bash
python scripts/plot_history.py
```

To confirm the package works end to end without the real dataset:

```bash
python examples/basic_usage.py
```

## Models compared

- `PVForecastingModel` — the full quantum-enhanced model (referred to as Quantum-Enhanced)
- `HybridHeadedModel` — HybridQuantumLSTM with a lighter regression head (HybridQuantumLSTM)
- `ClassicalLSTMModel` — LSTM baseline with a matching output head
- `GRUModel` — GRU baseline with a matching output head
- XGBoost, CatBoost — gradient-boosted trees on flattened input windows
- ARIMA — univariate baseline fit on the target series alone

All four neural models share the same input normalization, output head shape, optimizer
(Adam with weight decay), gradient clipping, learning-rate scheduling, and early stopping,
so differences in the results are attributable to the sequence model itself.

## Data

The pipeline expects per-minute PV inverter telemetry as daily CSV files under
`data/raw/<year>/<month>/`, with `dc_power__422` as the forecast target. See
`data/README.md` for the full column reference. Sequences of 24 consecutive readings are
used to predict the next reading; features are standardized and the target is scaled to
[0, 1] by `PVDataProcessor`, fit on the training split only.

## Results

Validation-set metrics from the final epoch of the full training run (see `history/*.json`
for the complete per-epoch record; full test-set results are reported in the paper):

| Model | Epochs | VAF | RMSE | MAE |
|---|---|---|---|---|
| Classical LSTM | 49 | 0.9863 | 41.35 | 14.30 |
| HybridQuantumLSTM | 24 | 0.9859 | 41.94 | 13.71 |
| Quantum-Enhanced | 22 | 0.9818 | 47.71 | 20.93 |
| GRU | 72 | 0.9863 | 41.36 | 14.87 |

The plots below are from a separate, reduced-scope run (a 6,000-sequence subsample, 6
epochs) used to validate the training and evaluation pipeline end to end, not the
full-scale run behind the numbers above or the paper. The per-sample quantum circuit
simulation makes a full epoch on the complete ~500K-sequence dataset take on the order of
an hour per quantum model on a single machine, so this smaller run is what's practical to
include here; regenerate at full scale with `scripts/run_benchmark.py --retrain`.

![HybridQuantumLSTM predicted vs true](docs/assets/HybridQuantumLSTM_pred_vs_true.png)

![Comparative metrics across models](docs/assets/all_models_metrics_bar.png)

![Quantum-Enhanced training curves](docs/assets/QuantumEnhanced_training_curves.png)

![Error distribution across models](docs/assets/all_models_error_boxplot.png)

## Known issue: OpenMP crash on macOS

Importing `torch` alongside `xgboost` or `catboost` can segfault on macOS due to a conflict
between the OpenMP runtimes each library bundles. If `scripts/run_benchmark.py` dies
without a traceback, set these before running:

```bash
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Tests

```bash
pip install pytest
pytest tests/
```

## Notes on the metrics

VAF and R2 are consistently high (0.98+) across all models on this dataset because PV
output is dominated by a strong, predictable diurnal cycle. MAPE is reported for
completeness but is not a reliable metric here: at night `dc_power__422` is at or near
zero, so small absolute errors during those hours produce extremely large percentage
errors. RMSE and MAE are more representative of practical forecasting error.

## Citation

```bibtex
@inproceedings{sharma2025hqlstm,
  author    = {Sharma, Mayan},
  title     = {Enhanced Hybrid Quantum-LSTM for Solar PV Forecasting},
  booktitle = {IET BICET 2025},
  year      = {2025},
  doi       = {10.1049/icp.2026.1046},
  url       = {https://digital-library.theiet.org/doi/10.1049/icp.2026.1046}
}
```

## License

MIT. See `LICENSE`.
