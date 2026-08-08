# Enhanced Hybrid Quantum-LSTM for Solar PV Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Mayan10/Enhanced-HQLSTM-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/Mayan10/Enhanced-HQLSTM-Model/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1049%2Ficp.2026.1046-b31b1b.svg)](https://digital-library.theiet.org/doi/10.1049/icp.2026.1046)

Official implementation of:

> **Enhanced Hybrid Quantum-LSTM for Solar PV Forecasting**
> Mayan Sharma. IET BICET 2025.
> [Paper](https://digital-library.theiet.org/doi/10.1049/icp.2026.1046) · [Citation](#citation)

A hybrid quantum-classical LSTM for short-horizon solar photovoltaic (PV) power
forecasting. A parameterized quantum circuit, simulated with
[PennyLane](https://pennylane.ai) and trained end-to-end through its PyTorch interface,
augments an LSTM's hidden state via a residual connection. Benchmarked against classical
LSTM/GRU baselines and XGBoost, CatBoost, and ARIMA on real PV inverter telemetry.

<p align="center">
  <img src="docs/assets/HybridQuantumLSTM_pred_vs_true.png" width="600" alt="Predicted vs true PV power output">
</p>

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

## Method

`PVForecastingModel` wraps an LSTM with a small parameterized quantum circuit:

1. The LSTM's final hidden state is linearly projected onto `n_qubits` qubits and bounded
   with `tanh`.
2. A data-encoding layer (angle encoding by default, Fourier encoding as an alternative)
   applies a per-qubit rotation scaled from the projected features.
3. Trainable single-qubit rotations followed by a chain of nearest-neighbor CNOTs entangle
   the register.
4. Pauli-Z expectation values are measured on each qubit and projected back to the hidden
   size.
5. The result is added to the original hidden state and layer-normalized.

Gradients flow through the quantum circuit via PennyLane's `torch` interface and through
the LSTM via standard autograd, so the whole model trains end-to-end with a single Adam
optimizer.

Four neural architectures share this training setup (Adam with weight decay, gradient
clipping, `ReduceLROnPlateau`, early stopping), isolating the sequence model as the only
variable between them:

| Model | Sequence layer | Quantum circuit |
|---|---|---|
| `PVForecastingModel` | LSTM | 4 qubits, deeper regression head |
| `HybridHeadedModel` | LSTM | 4 qubits, lighter regression head |
| `ClassicalLSTMModel` | LSTM | none |
| `GRUModel` | GRU | none |

These are compared against XGBoost and CatBoost (fit on flattened input windows) and
ARIMA (univariate, target series only) as non-neural baselines.

## Installation

```bash
git clone https://github.com/Mayan10/Enhanced-HQLSTM-Model.git
cd Enhanced-HQLSTM-Model
pip install -e .
```

or with `requirements.txt` in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

Place raw data under `data/raw/<year>/` (see `data/README.md` for the expected layout and
schema), then run the full comparative study:

```bash
python scripts/run_benchmark.py
```

By default this loads any existing checkpoint for each neural model instead of retraining
it. Pass `--retrain` to train from scratch, and see `--help` for epochs, learning rate,
patience, and sequence length:

```bash
python scripts/run_benchmark.py --retrain --epochs 75 --lr 0.001
```

Results are written to `plots/` (training curves, prediction scatter plots, residuals,
error boxplots, per-metric comparisons) and `history/` (per-epoch metrics as JSON).
Checkpoints are written to `checkpoints/<ModelName>_best.pth`, one file per model.

To regenerate the comparison plots from existing `history/*.json` files without touching
the models:

```bash
python scripts/plot_history.py
```

To confirm the package works end to end without the real dataset:

```bash
python examples/basic_usage.py
```

## Results

Validation-set metrics from the final epoch of the full training run (see `history/*.json`
for the complete per-epoch record; full test-set results are reported in the paper):

| Model | Epochs | VAF | RMSE | MAE |
|---|---|---|---|---|
| Classical LSTM | 49 | 0.9863 | 41.35 | 14.30 |
| HybridQuantumLSTM | 24 | 0.9859 | 41.94 | 13.71 |
| Quantum-Enhanced | 22 | 0.9818 | 47.71 | 20.93 |
| GRU | 72 | 0.9863 | 41.36 | 14.87 |

VAF and R2 are consistently high because PV output is dominated by a strong, predictable
diurnal cycle. MAPE is reported by the pipeline for completeness but is not a reliable
metric here: at night, `dc_power__422` is at or near zero, so small absolute errors during
those hours produce extremely large percentage errors. RMSE and MAE are more representative
of practical forecasting error.

The plots below are from a separate, reduced-scope run (a 6,000-sequence subsample, 6
epochs) used to validate the training and evaluation pipeline end to end, not the
full-scale run behind the table above or the paper. A full epoch on the complete
~500K-sequence dataset takes on the order of an hour per quantum model on a single machine
because the circuit is simulated per sample in a Python loop, so this is what's practical
to include here. Regenerate at full scale with `scripts/run_benchmark.py --retrain`.

<p align="center">
  <img src="docs/assets/all_models_metrics_bar.png" width="800" alt="Comparative metrics across models">
</p>
<p align="center">
  <img src="docs/assets/QuantumEnhanced_training_curves.png" width="600" alt="Quantum-Enhanced training curves">
</p>
<p align="center">
  <img src="docs/assets/all_models_error_boxplot.png" width="500" alt="Error distribution across models">
</p>

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

## Data

The pipeline expects per-minute PV inverter telemetry as daily CSV files under
`data/raw/<year>/<month>/`, with `dc_power__422` as the forecast target. See
`data/README.md` for the full column reference. Sequences of 24 consecutive readings are
used to predict the next reading; features are standardized and the target is scaled to
[0, 1] by `PVDataProcessor`, fit on the training split only.

## Reproducibility

```bash
pip install pytest
pytest tests/
```

CI runs the same suite on every push and pull request against `main`.

### Known issue: OpenMP crash on macOS

Importing `torch` alongside `xgboost` or `catboost` can segfault on macOS due to a conflict
between the OpenMP runtimes each library bundles. If `scripts/run_benchmark.py` dies
without a traceback, set these before running:

```bash
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

## License

MIT. See `LICENSE`.
