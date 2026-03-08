# Solar PV Power Forecasting: Quantum, Hybrid, Deep Learning & ML Benchmark

A comprehensive benchmark for solar photovoltaic (PV) power forecasting, comparing quantum machine learning (QML), hybrid quantum-classical, and classical models on real-world PV data.

Designed for reproducibility and direct use in academic publications — IET conference paper template included.

---

## Models Compared

| Model | Type |
|-------|------|
| Classical LSTM | Deep Learning |
| GRU | Deep Learning |
| Hybrid Quantum LSTM | Quantum-Classical |
| Quantum-Enhanced Model | Quantum ML |
| XGBoost | ML Baseline |
| CatBoost | ML Baseline |
| ARIMA | Statistical Baseline |

---

## Features

- Unified data pipeline for multi-year PV data (2022, 2023) with automatic cleaning and splitting
- Evaluation across MAE, RMSE, MBE, VAF, R², and MAPE on a held-out test set
- Publication-ready figures: training curves, error distributions, scatter plots, boxplots, bar charts
- IET conference LaTeX template for direct manuscript preparation
- All results, logs, and plots saved automatically

---

## Data Setup

Place raw PV CSVs in:
```
data/raw/2022/   ← organized by month
data/raw/2023/   ← organized by month
```

The pipeline handles loading, cleaning, and stratified train/val/test splitting automatically.

---

## Getting Started

**1. Set up the environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Run training and evaluation**
```bash
python src/training/Training_and_Evaluation_Script_for_Enhanced_QML_Models.py
```

By default, previously trained quantum and deep learning models are loaded from `history/` to save time. To force a full retrain, set `skip_trained_models=False` in the script.

---

## Outputs

| Location | Contents |
|----------|----------|
| `plots/` | All publication-ready figures |
| `history/` | Model training logs and saved weights |

Key figures include `training_curves.png`, `feature_importance.png`, and `model_architecture.png`.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{mayan_sharma_2024_pvqml,
  author = {Mayan Sharma},
  title  = {Solar PV Power Forecasting: Quantum, Hybrid, Deep Learning, and ML Benchmark},
  year   = {2024},
  url    = {https://github.com/Mayan10/Enhanced-HQLSTM-Model}
}
```

---

## Author

**Mayan Sharma**
GitHub: [@Mayan10](https://github.com/Mayan10)

For questions or suggestions, open an issue on the repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
