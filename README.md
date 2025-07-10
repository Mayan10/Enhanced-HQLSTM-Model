# Solar PV Power Forecasting: Quantum, Hybrid, Deep Learning, and ML Benchmark

## Project Overview
This project provides a comprehensive, publication-ready benchmark for solar photovoltaic (PV) power forecasting. It compares advanced quantum machine learning (QML) models, hybrid quantum-classical models, classical deep learning (LSTM, GRU), and state-of-the-art machine learning models (XGBoost, CatBoost, ARIMA) on real-world PV data.

The codebase is designed for reproducibility, extensibility, and direct use in academic publications (IET conference template included).

---

## Features
- **Quantum, Hybrid, and Classical Deep Learning Models**: Enhanced Quantum Feature Maps, Hybrid Quantum LSTM, Classical LSTM, GRU
- **State-of-the-Art ML Baselines**: XGBoost, CatBoost, ARIMA
- **Unified Data Pipeline**: Loads, cleans, and processes multi-year PV data (2022, 2023)
- **Robust Evaluation**: MAE, RMSE, MBE, VAF, R², MAPE on a true test set
- **Publication-Ready Plots**: Training curves, error distributions, scatter plots, boxplots, bar charts
- **Easy Reproducibility**: All dependencies in `requirements.txt`, results and logs saved
- **IET Conference Paper Template**: For direct manuscript preparation

---

## Data
- Place raw PV data CSVs in `data/raw/2022/` and `data/raw/2023/` (organized by month)
- The pipeline automatically loads, cleans, and splits data into train/val/test (stratified by time)
- All features are numerically validated and cleaned for robust model training

---

## Models Compared
- **Classical LSTM**
- **Hybrid Quantum LSTM**
- **Quantum-Enhanced Model**
- **GRU (Gated Recurrent Unit)**
- **XGBoost**
- **CatBoost**
- **ARIMA (univariate baseline)**

---

## How to Run
1. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare data**: Place all raw CSVs in the appropriate `data/raw/` folders.
3. **Run the main script**:
   ```bash
   python src/training/Training_and_Evaluation_Script_for_Enhanced_QML_Models.py
   ```
   - By default, previously trained quantum/deep models are loaded from `history/` to save time. To retrain, set `skip_trained_models=False` in the script.
4. **Results**:
   - All plots are saved in `plots/`
   - Training logs/history in `history/`
   - Key figures: training curves, error boxplots, bar charts, scatter plots, etc.

---

## Results & Plots
- **All results and publication-ready figures** are saved in the `plots/` directory.
- **Model training histories** are in `history/`.
- Example figures: `training_curves.png`, `feature_importance.png`, `model_architecture.png`
- Use these directly in your IET conference paper or other publications.

---

## For Publication
- The project includes `IET Conference Paper Template.docx` for manuscript preparation.
- All code, results, and figures are organized for easy integration into your paper.
- Cite this project or acknowledge the codebase as appropriate.

---

## Citation
If you use this benchmark or codebase in your research, please cite:

```
@software{yourname_2024_pvqml,
  author = {Your Name},
  title = {Solar PV Power Forecasting: Quantum, Hybrid, Deep Learning, and ML Benchmark},
  year = {2024},
  url = {https://github.com/yourrepo/pvqml}
}
```

---

## Contact
For questions, suggestions, or contributions, please open an issue or contact the maintainer.
