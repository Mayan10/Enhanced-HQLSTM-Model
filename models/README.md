# Architecture reference

These files are `print(model)` dumps of `PVForecastingModel` at three capacities explored
during development, kept for reference:

| File | Hidden size | Qubits |
|---|---|---|
| `base_hqlstm_architecture.txt` | 32 | 4 |
| `enhanced_hqlstm_architecture.txt` | 64 | 4 |
| `multi-scale_hqlstm_architecture.txt` | 128 | 6 |

`PVForecastingModel(input_features, hidden_size, n_qubits, dropout)` in
`src/hqlstm/models.py` reproduces this same structure for any of these configurations;
these text files are not regenerated automatically and may drift from the current default
hyperparameters over time.

Trained weights (`*.pth`) are not tracked in this repository. Training writes checkpoints
to `checkpoints/<ModelName>_best.pth`, one file per model.
