"""
Basic example of using the Enhanced QML PV Forecasting package.
"""

from src import (
    EnhancedPVForecastingModel,
    PVDataProcessor,
    ExperimentRunner
)

def main():
    # Initialize data processor
    data_processor = PVDataProcessor(sequence_length=24)
    
    # Load and preprocess data
    X, y = data_processor.load_and_preprocess_data()
    
    # Split data into train/val/test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner()
    
    # Run comparative study
    experiment_runner.run_comparative_study(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    # Plot results
    experiment_runner.plot_results()

if __name__ == "__main__":
    main() 