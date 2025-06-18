"""
Basic example showing how to use our Enhanced Quantum Machine Learning 
solar power forecasting package. This demonstrates the complete pipeline 
from data loading to model training and evaluation.
"""

from src import (
    PVForecastingModel,
    PVDataProcessor,
    ExperimentRunner
)

def main():
    # Set up our data processor to handle solar power data
    data_processor = PVDataProcessor(sequence_length=24)
    
    # Load and preprocess our solar power data
    X, y = data_processor.load_and_preprocess_data()
    
    # Split our data into training, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Set up our experiment runner to handle multiple model configurations
    experiment_runner = ExperimentRunner()
    
    # Run our comparative study with different model architectures
    experiment_runner.run_comparative_study(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    # Generate beautiful plots of our results
    experiment_runner.plot_results()

if __name__ == "__main__":
    main() 