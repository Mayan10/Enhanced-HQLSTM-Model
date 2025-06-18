import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
from datetime import datetime
from tqdm import tqdm

# Import our enhanced quantum models
from src.models.Enhanced_Quantum_Feature_Maps_for_PV_Power_Forecasting import (
    PVForecastingModel,
    PVDataProcessor,
    calculate_metrics
)

# Let's use Apple Silicon's MPS if it's available for faster training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class PVDataProcessor:
    """
    Handles all the data preprocessing for our solar power forecasting.
    This makes sure our data is properly scaled and ready for the model.
    """
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
        
    def fit_transform(self, X, y):
        """Fit our scalers to the data and transform it."""
        X_scaled = self.feature_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.fitted = True
        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)
    
    def transform(self, X, y=None):
        """Transform new data using our fitted scalers."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        X_scaled = self.feature_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        if y is not None:
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
            return X_tensor, y_tensor
        
        return X_tensor
    
    def inverse_transform_target(self, y_scaled):
        """Convert our scaled predictions back to the original scale."""
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().numpy()
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

class ModelTrainer:
    """
    A trainer class that handles the complete training process for our enhanced quantum models.
    This includes training, validation, early stopping, and model saving.
    """
    
    def __init__(self, model: PVForecastingModel, processor: PVDataProcessor, 
                 device: torch.device = device):  # Use the global device
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Set up beautiful, publication-quality plotting style for our results."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 75, lr: float = 0.001, patience: int = 20) -> Dict:
        """Train our model with proper optimization and monitoring."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        metrics_history = []
        patience_counter = 0
        
        print(f"\nTraining on {self.device}...")
        print("=" * 50)
        
        # Create a progress bar to track our training epochs
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            # Training phase - update our model parameters
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            # Progress bar for training batches within each epoch
            train_batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                                  position=1, leave=False)
            
            for batch_X, batch_y in train_batch_pbar:
                # Move our data to the right device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                
                # Clip gradients to prevent exploding gradients during training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(predictions.detach().cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
                
                # Update our batch progress bar with current loss info
                train_batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{train_loss/len(train_batch_pbar):.4f}'
                })
            
            train_loss /= len(train_loader)
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
            
            # Validation phase - check how well we're generalizing
            val_loss, val_metrics = self._validate(val_loader, criterion)
            
            # Adjust learning rate based on validation performance
            scheduler.step(val_loss)
            
            # Early stopping - save the best model and stop if we're not improving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                # Save our best model to disk
                torch.save(self.model.state_dict(), f'best_model.pth')
                print(f"\nNew best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Store our metrics for later analysis
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            metrics_history.append(val_metrics)
            
            # Update our epoch progress bar with current performance
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_rmse': f'{val_metrics["RMSE"]:.4f}',
                'val_vaf': f'{val_metrics["VAF"]:.4f}'
            })
            
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        # Load the best model we found during training
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics_history': metrics_history
        }
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict]:
        """Validate our model and return loss and performance metrics."""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation", position=1, leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
                # Move tensors to CPU before converting to numpy for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
                # Update validation progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        # Convert our predictions back to original scale for proper evaluation
        predictions_orig = self.processor.inverse_transform_target(np.array(all_predictions))
        targets_orig = self.processor.inverse_transform_target(np.array(all_targets))
        
        # Calculate comprehensive performance metrics
        metrics = calculate_metrics(targets_orig, predictions_orig)
        
        return val_loss, metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate our model on the test set to get final performance metrics.
        """
        print("\nEvaluating on test set...")
        print("=" * 50)
        
        _, metrics = self._validate(test_loader, nn.MSELoss())
        
        print("\nTest Results:")
        print(f"VAF (Variance Accounted For): {metrics['VAF']:.4f}")
        print(f"R²: {metrics['R2']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        
        return metrics
    
    def save_model(self, path: str):
        """
        Save our trained model and all its components for later use.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor': self.processor,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"\nModel saved to {path}")
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], 
                           metrics_history: List[Dict], save_path: str = None):
        """Plot beautiful training curves and metrics for publication."""
        # Move any tensors to CPU before plotting
        if isinstance(train_losses, torch.Tensor):
            train_losses = train_losses.cpu().numpy()
        if isinstance(val_losses, torch.Tensor):
            val_losses = val_losses.cpu().numpy()
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot training and validation losses
        axes[0].plot(train_losses, label='Training Loss', color='blue')
        axes[0].plot(val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Losses')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot RMSE over time
        rmse_values = [m['RMSE'] for m in metrics_history]
        axes[1].plot(rmse_values, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Square Error')
        axes[1].grid(True)
        
        # Plot VAF over time
        vaf_values = [m['VAF'] for m in metrics_history]
        axes[2].plot(vaf_values, color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('VAF')
        axes[2].set_title('Variance Accounted For')
        axes[2].grid(True)
        
        # Plot MAE over time
        mae_values = [m['MAE'] for m in metrics_history]
        axes[3].plot(mae_values, color='orange')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('MAE')
        axes[3].set_title('Mean Absolute Error')
        axes[3].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ExperimentRunner:
    """
    A comprehensive experiment runner that handles multiple model configurations,
    training, evaluation, and result visualization for our solar power forecasting study.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_histories = {}
        self.synthetic_predictions = {}  # Store synthetic predictions for plotting
        
    def pre_training_check(self, X_train, X_val, X_test, y_train, y_val, y_test) -> bool:
        """
        Run comprehensive checks before training to ensure everything is set up correctly.
        This helps us catch issues early and avoid wasting time on broken configurations.
        """
        print("Running pre-training checks...")
        print("=" * 50)
        
        # Check data shapes and types
        print("1. Checking data shapes and types...")
        try:
            assert X_train.ndim == 3, f"X_train should be 3D, got {X_train.ndim}D"
            assert X_val.ndim == 3, f"X_val should be 3D, got {X_val.ndim}D"
            assert X_test.ndim == 3, f"X_test should be 3D, got {X_test.ndim}D"
            assert y_train.ndim == 1, f"y_train should be 1D, got {y_train.ndim}D"
            assert y_val.ndim == 1, f"y_val should be 1D, got {y_val.ndim}D"
            assert y_test.ndim == 1, f"y_test should be 1D, got {y_test.ndim}D"
            print("   ✓ Data shapes are correct")
        except AssertionError as e:
            print(f"   ✗ Data shape error: {e}")
            return False
        
        # Check for NaN or infinite values
        print("2. Checking for NaN or infinite values...")
        try:
            assert not np.isnan(X_train).any(), "NaN values found in X_train"
            assert not np.isnan(X_val).any(), "NaN values found in X_val"
            assert not np.isnan(X_test).any(), "NaN values found in X_test"
            assert not np.isnan(y_train).any(), "NaN values found in y_train"
            assert not np.isnan(y_val).any(), "NaN values found in y_val"
            assert not np.isnan(y_test).any(), "NaN values found in y_test"
            assert not np.isinf(X_train).any(), "Infinite values found in X_train"
            assert not np.isinf(X_val).any(), "Infinite values found in X_val"
            assert not np.isinf(X_test).any(), "Infinite values found in X_test"
            assert not np.isinf(y_train).any(), "Infinite values found in y_train"
            assert not np.isinf(y_val).any(), "Infinite values found in y_val"
            assert not np.isinf(y_test).any(), "Infinite values found in y_test"
            print("   ✓ No NaN or infinite values found")
        except AssertionError as e:
            print(f"   ✗ Data quality error: {e}")
            return False
        
        # Check data ranges
        print("3. Checking data ranges...")
        try:
            print(f"   X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
            print(f"   y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
            print(f"   y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
            print("   ✓ Data ranges look reasonable")
        except Exception as e:
            print(f"   ✗ Data range check failed: {e}")
            return False
        
        # Check target variance
        print("4. Checking target variance...")
        y_train_var = y_train.var()
        if y_train_var < 1e-6:
            print(f"   ⚠ Warning: Very low target variance ({y_train_var:.2e})")
            print("   This might cause training issues")
        else:
            print(f"   ✓ Target variance is reasonable: {y_train_var:.4f}")
        
        # Test model creation
        print("5. Testing model creation...")
        try:
            test_model = PVForecastingModel(input_features=X_train.shape[-1], hidden_size=32, n_qubits=2)
            print("   ✓ Model creation successful")
        except Exception as e:
            print(f"   ✗ Model creation failed: {e}")
            return False
        
        # Test forward pass
        print("6. Testing forward pass...")
        try:
            test_input = torch.randn(2, X_train.shape[1], X_train.shape[2])
            with torch.no_grad():
                test_output = test_model(test_input)
            assert test_output.shape == (2, 1), f"Expected shape (2, 1), got {test_output.shape}"
            print("   ✓ Forward pass successful")
        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            return False
        
        # Test data processor
        print("7. Testing data processor...")
        try:
            processor = PVDataProcessor()
            X_scaled, y_scaled = processor.fit_transform(X_train[:10], y_train[:10])
            X_test_scaled, y_test_scaled = processor.transform(X_val[:5], y_val[:5])
            y_orig = processor.inverse_transform_target(y_scaled)
            print("   ✓ Data processor working correctly")
        except Exception as e:
            print(f"   ✗ Data processor failed: {e}")
            return False
        
        print("\nAll pre-training checks passed! ✓")
        return True

def main():
    """
    Main function to run our enhanced quantum model training and evaluation.
    This orchestrates the entire pipeline from data loading to final results.
    """
    print("Starting Enhanced Quantum Model Training and Evaluation")
    print("=" * 50)
    
    # Set up our experiment runner
    runner = ExperimentRunner()
    
    # Load and preprocess our solar power data
    print("\nLoading and preprocessing data...")
    data_processor = PVDataProcessor()
    
    # Find all our data files in the raw data directory
    data_files = []
    for root, dirs, files in os.walk('data/raw'):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(data_files)} data files to process")
    
    # Load and combine all our data files
    all_data = []
    for file in data_files:
        print(f"Loading {file}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    # Combine all our data into one big dataset
    data = pd.concat(all_data, ignore_index=True)
    
    # Let's validate our data to make sure it looks good
    print("\nData Validation:")
    print("=" * 50)
    print(f"Total samples: {len(data)}")
    print(f"Date range: {data['measured_on'].min()} to {data['measured_on'].max()}")
    
    # Check for any missing values that might cause problems
    print("\nMissing values per feature:")
    print(data.isnull().sum())
    
    # Clean up our data by removing any rows with missing values
    data = data.dropna()
    print(f"\nSamples after cleaning: {len(data)}")
    
    # Let's see what our feature ranges look like after cleaning
    print("\nFeature ranges after cleaning:")
    for col in data.columns:
        if col != 'measured_on':
            print(f"{col}: [{data[col].min():.2f}, {data[col].max():.2f}]")
    
    # Check for outliers that might skew our training
    print("\nOutlier detection:")
    for col in data.columns:
        if col != 'measured_on':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outliers} outliers detected")
    
    # Prepare our features and target for the model
    feature_columns = [col for col in data.columns if col not in ['measured_on', 'dc_power__422']]
    target_column = 'dc_power__422'
    
    # Create time sequences for our LSTM model
    sequence_length = 24  # 24 hours of historical data
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][feature_columns].values)
        y.append(data.iloc[i + sequence_length][target_column])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split our data into training, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Run our comprehensive pre-training checks
    if not runner.pre_training_check(X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nPre-training checks failed. Exiting...")
        return
    
    # If everything looks good, let's start training our models!
    print("\nStarting training process...")
    runner.run_comparative_study(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()