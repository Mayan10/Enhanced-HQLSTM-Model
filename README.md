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
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Device Support

The model supports different computing devices:
- CPU: Default fallback device
- MPS (Metal Performance Shaders): For Apple Silicon Macs
- CUDA: For NVIDIA GPUs (experimental)

Note: Quantum operations are performed on CPU regardless of the device setting, while classical operations can utilize the available hardware acceleration.

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
from src.models.Enhanced_Quantum_Feature_Maps_for_PV_Power_Forecasting import PVForecastingModel

# Initialize model
model = PVForecastingModel(
    input_features=5,
    hidden_size=32,
    n_qubits=4
)

# Make predictions
predictions = model(input_data)
```

## Model Architecture

The model combines:
1. Quantum Feature Map: Encodes classical data into quantum states
2. Quantum Layer: Processes quantum information using parameterized circuits
3. Classical LSTM: Handles temporal dependencies
4. Attention Mechanism: Focuses on relevant time steps

## Training Process

1. Data Preprocessing:
   - Feature scaling
   - Sequence creation
   - Train/validation/test split

2. Model Training:
   - Automatic device selection
   - Gradient clipping for stability
   - Learning rate scheduling
   - Early stopping

3. Evaluation:
   - Multiple metrics (RMSE, MAE, VAF)
   - Visualization tools
   - Model comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
