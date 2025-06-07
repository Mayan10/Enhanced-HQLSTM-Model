# Enhanced Quantum Machine Learning for PV Power Forecasting

This project implements an enhanced quantum machine learning approach for photovoltaic (PV) power forecasting using hybrid quantum-classical neural networks. The implementation combines quantum circuits with classical LSTM networks to improve prediction accuracy and capture complex temporal patterns in solar power generation.

## Features

- Hybrid Quantum-LSTM (HQLSTM) architecture
- Multiple quantum feature encoding strategies (Fourier, IQP, Amplitude)
- Enhanced data preprocessing and feature engineering
- Attention mechanisms for better temporal modeling
- Comprehensive evaluation metrics and visualizations
- Support for both synthetic and real-world PV data

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-qml-pv-forecasting.git
cd enhanced-qml-pv-forecasting
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training the model:
```python
from src.training.Training_and_Evaluation_Script_for_Enhanced_QML_Models import ExperimentRunner

# Initialize experiment runner
experiment_runner = ExperimentRunner()

# Run experiments with your data
experiment_runner.run_comparative_study(X_train, X_val, X_test, y_train, y_val, y_test)
```

2. Using the model for predictions:
```python
from src.models.Enhanced_Quantum_Feature_Maps_for_PV_Power_Forecasting import EnhancedPVForecastingModel

# Initialize model
model = EnhancedPVForecastingModel(
    input_features=5,
    sequence_length=24,
    hidden_size=32,
    n_qubits=4,
    n_quantum_layers=2,
    encoding_type="fourier"
)

# Make predictions
predictions = model(input_data)
```

## Model Architecture

The project implements three variants of the HQLSTM model:

1. Base HQLSTM: Standard implementation with basic quantum circuits
2. Enhanced HQLSTM: Improved version with attention mechanisms and better feature encoding
3. Multi-Scale HQLSTM: Advanced version with multiple quantum layers and enhanced entanglement

## Results

The model performance is evaluated using multiple metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Variance Accounted For (VAF)
- R-squared Score
- Directional Accuracy

## Requirements

- Python 3.8+
- PyTorch
- PennyLane
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{enhanced_qml_pv_forecasting,
  author = {Mayan Sharma},
  title = {Enhanced Quantum Machine Learning for PV Power Forecasting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/enhanced-qml-pv-forecasting}
}
``` 
