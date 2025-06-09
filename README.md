# Enhanced Quantum Machine Learning for PV Power Forecasting

This project implements an enhanced quantum machine learning approach for photovoltaic (PV) power forecasting using hybrid quantum-classical neural networks. The implementation combines quantum circuits with classical LSTM networks to improve prediction accuracy and capture complex temporal patterns in solar power generation.

## Features

- Hybrid Quantum-LSTM (HQLSTM) architecture
- Multiple quantum feature encoding strategies (Fourier, IQP, Amplitude)
- Enhanced data preprocessing and feature engineering
- Attention mechanisms for better temporal modeling
- Comprehensive evaluation metrics and visualizations
- Support for both synthetic and real-world PV data
- Optimized device handling for CPU and MPS (Apple Silicon)

## Project Structure

```
├── src/
│   ├── models/
│   │   └── Enhanced_Quantum_Feature_Maps_for_PV_Power_Forecasting.py
│   ├── training/
│   │   └── Training_and_Evaluation_Script_for_Enhanced_QML_Models.py
│   └── utils/
│       └── data_processor.py
├── data/
│   └── System Data Jan 1 2023.csv
├── notebooks/
│   └── examples/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PennyLane 0.30+
- NumPy 1.20+
- Pandas 1.3+
- scikit-learn 1.0+
- Matplotlib 3.4+
- tqdm 4.62+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mayan10/Enhanced-HQLSTM-Model.git
cd Enhanced-HQLSTM-Model
```

2. Create and activate a virtual environment:
```