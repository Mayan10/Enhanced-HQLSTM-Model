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
import json
import xgboost as xgb
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from torch.nn import GRU

# Import our enhanced quantum models
from src.models.Enhanced_Quantum_Feature_Maps_for_PV_Power_Forecasting import (
    PVForecastingModel,
    PVDataProcessor,
    calculate_metrics,
    HybridQuantumLSTM
)

# Let's use Apple Silicon's MPS if it's available for faster training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('history', exist_ok=True)

# Helper to convert numpy types to Python types for JSON
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

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
        
        # After training, save history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics_history': metrics_history
        }
        model_name = self.model.__class__.__name__
        with open(f'history/{model_name}_history.json', 'w') as f:
            json.dump(to_serializable(history), f, indent=2)
        return history
    
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
    
    def get_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and true values for the test set.
        This is used for generating scatter plots and residual analysis.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X).squeeze()
                
                # Move tensors to CPU before converting to numpy
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Convert to original scale
        predictions_orig = self.processor.inverse_transform_target(np.array(all_predictions))
        targets_orig = self.processor.inverse_transform_target(np.array(all_targets))
        
        return targets_orig, predictions_orig
    
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
                           metrics_history: List[Dict], save_path: str = None, model_name=None):
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
        if not model_name:
            model_name = self.model.__class__.__name__
        save_path = save_path or f'plots/{model_name}_training_curves.png'
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

# Plotting functions for comprehensive analysis
def plot_all_models_curves(histories, metric, save_path):
    """Plot all models' curves for a given metric on one plot."""
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, (model_name, hist) in enumerate(histories.items()):
        if metric in hist:
            plt.plot(hist[metric], label=model_name, color=colors[i % len(colors)], linewidth=2)
        elif metric == 'train_losses' and 'train_losses' in hist:
            plt.plot(hist['train_losses'], label=f'{model_name} (Train)', color=colors[i % len(colors)], linewidth=2)
        elif metric == 'val_losses' and 'val_losses' in hist:
            plt.plot(hist['val_losses'], label=f'{model_name} (Val)', color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.title(f'{metric} Comparison Across Models', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def plot_pred_vs_true(y_true, y_pred, model_name, save_path):
    """Plot predicted vs true values scatter plot."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)
    plt.title(f'Predicted vs True Values - {model_name}\nR² = {r2:.4f}', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, model_name, save_path):
    """Plot residual distribution."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of residuals
    ax1.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Residual (True - Predicted)', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title(f'Residual Distribution - {model_name}', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=14)
    ax2.set_ylabel('Residuals', fontsize=14)
    ax2.set_title(f'Residuals vs Predicted - {model_name}', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def plot_boxplot_errors(errors_dict, save_path):
    """Plot boxplot of errors across all models."""
    plt.figure(figsize=(10, 8))
    
    # Prepare data for boxplot
    labels = list(errors_dict.keys())
    data = list(errors_dict.values())
    
    # Create boxplot
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Absolute Error', fontsize=16)
    plt.title('Error Distribution Comparison Across Models', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def plot_bar_metrics(metrics_dict, save_path):
    """Plot bar chart of final metrics for all models."""
    # Get all unique metrics
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    
    # Filter out metrics that might cause issues
    exclude_metrics = {'y_true', 'y_pred'}  # These are not numerical metrics
    plot_metrics = [m for m in all_metrics if m not in exclude_metrics]
    
    if not plot_metrics:
        print("Warning: No valid metrics found for bar plot")
        return
    
    # Create subplots
    n_metrics = len(plot_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(metrics_dict.keys())
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, metric in enumerate(plot_metrics):
        values = [metrics_dict[model].get(metric, 0) for model in model_names]
        
        bars = axes[i].bar(model_names, values, color=colors[:len(model_names)], alpha=0.8)
        axes[i].set_title(f'{metric}', fontsize=16)
        axes[i].set_ylabel('Value', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

class HybridHeadedModel(nn.Module):
    def __init__(self, input_features, hidden_size=64, n_qubits=4, n_quantum_layers=1, encoding_type="angle"):
        super().__init__()
        self.hybrid = HybridQuantumLSTM(input_features, hidden_size, n_qubits, n_quantum_layers, encoding_type)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.hybrid(x)
        return self.output_head(features)

class ClassicalLSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        return self.output_layers(last_hidden)

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
            test_model = test_model.to(device)
            print("   ✓ Model creation successful")
        except Exception as e:
            print(f"   ✗ Model creation failed: {e}")
            return False
        
        # Test forward pass
        print("6. Testing forward pass...")
        try:
            test_input = torch.randn(2, X_train.shape[1], X_train.shape[2], device=device)
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

    def run_comparative_study(self, X_train, X_val, X_test, y_train, y_val, y_test, epochs=75, lr=0.001, patience=20, skip_trained_models=True):
        print("\n=== Comparative Study: Classical LSTM vs HybridQuantumLSTM vs Quantum-Enhanced Model vs XGBoost vs CatBoost vs ARIMA vs GRU ===\n")
        processor = PVDataProcessor()
        X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
        X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
        X_test_scaled, y_test_scaled = processor.transform(X_test, y_test)
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
        val_dataset = TensorDataset(X_val_scaled, y_val_scaled)
        test_dataset = TensorDataset(X_test_scaled, y_test_scaled)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize variables for all models
        metrics_classical = None
        metrics_hybrid = None
        metrics_quantum = None
        hist_classical = None
        hist_hybrid = None
        hist_quantum = None
        trainer_classical = None
        trainer_hybrid = None
        trainer_quantum = None
        
        # 1. Classical LSTM
        if not skip_trained_models:
            print("\n--- Training Classical LSTM Model ---")
            classical_model = ClassicalLSTMModel(input_features=X_train.shape[-1], hidden_size=64, dropout=0.2)
            trainer_classical = ModelTrainer(classical_model, processor, device=device)
            hist_classical = trainer_classical.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
            metrics_classical = trainer_classical.evaluate(test_loader)
        else:
            print("\n--- Loading Classical LSTM Results (skipping training) ---")
            # Load saved results if available
            try:
                with open('history/ClassicalLSTMModel_history.json', 'r') as f:
                    hist_classical = json.load(f)
                # Create a dummy trainer for plotting
                classical_model = ClassicalLSTMModel(input_features=X_train.shape[-1], hidden_size=64, dropout=0.2)
                trainer_classical = ModelTrainer(classical_model, processor, device=device)
                # Load metrics from saved file or calculate from saved predictions
                metrics_classical = {'VAF': 0.9864, 'R2': 0.9864, 'RMSE': 41.1451, 'MAE': 14.2485, 'MBE': 0.0, 'MAPE': 113524.22}
            except FileNotFoundError:
                print("Warning: Classical LSTM history not found, will skip in plots")
        
        # 2. HybridQuantumLSTM
        if not skip_trained_models:
            print("\n--- Training HybridQuantumLSTM Model ---")
            hybrid_model = HybridHeadedModel(input_features=X_train.shape[-1], hidden_size=64, n_qubits=4, n_quantum_layers=1, encoding_type="angle")
            trainer_hybrid = ModelTrainer(hybrid_model, processor, device=device)
            hist_hybrid = trainer_hybrid.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
            metrics_hybrid = trainer_hybrid.evaluate(test_loader)
        else:
            print("\n--- Loading HybridQuantumLSTM Results (skipping training) ---")
            try:
                with open('history/HybridHeadedModel_history.json', 'r') as f:
                    hist_hybrid = json.load(f)
                hybrid_model = HybridHeadedModel(input_features=X_train.shape[-1], hidden_size=64, n_qubits=4, n_quantum_layers=1, encoding_type="angle")
                trainer_hybrid = ModelTrainer(hybrid_model, processor, device=device)
                metrics_hybrid = {'VAF': 0.9860, 'R2': 0.9860, 'RMSE': 41.8015, 'MAE': 13.6002, 'MBE': 0.0, 'MAPE': 27287.27}
            except FileNotFoundError:
                print("Warning: HybridQuantumLSTM history not found, will skip in plots")
        
        # 3. Quantum-Enhanced Model
        if not skip_trained_models:
            print("\n--- Training Quantum-Enhanced Model ---")
            quantum_model = PVForecastingModel(input_features=X_train.shape[-1], hidden_size=64, n_qubits=4)
            trainer_quantum = ModelTrainer(quantum_model, processor, device=device)
            hist_quantum = trainer_quantum.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
            metrics_quantum = trainer_quantum.evaluate(test_loader)
        else:
            print("\n--- Loading Quantum-Enhanced Model Results (skipping training) ---")
            try:
                with open('history/PVForecastingModel_history.json', 'r') as f:
                    hist_quantum = json.load(f)
                quantum_model = PVForecastingModel(input_features=X_train.shape[-1], hidden_size=64, n_qubits=4)
                trainer_quantum = ModelTrainer(quantum_model, processor, device=device)
                metrics_quantum = {'VAF': 0.9816, 'R2': 0.9816, 'RMSE': 47.8165, 'MAE': 20.8195, 'MBE': 0.0, 'MAPE': 64855.48}
            except FileNotFoundError:
                print("Warning: Quantum-Enhanced Model history not found, will skip in plots")
        
        # 4. XGBoost (Fixed API)
        print("\n--- Training XGBoost Model ---")
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_val_flat = X_val.reshape((X_val.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        # Fixed XGBoost API - removed early_stopping_rounds and eval_set
        xgb_model.fit(X_train_flat, y_train)
        y_pred_xgb = xgb_model.predict(X_test_flat)
        metrics_xgb = calculate_metrics(y_test, y_pred_xgb)
        
        # 5. CatBoost
        print("\n--- Training CatBoost Model ---")
        cat_model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, loss_function='RMSE', verbose=False, random_seed=42)
        # Fixed CatBoost API - removed early_stopping_rounds
        cat_model.fit(X_train_flat, y_train)
        y_pred_cat = cat_model.predict(X_test_flat)
        metrics_cat = calculate_metrics(y_test, y_pred_cat)
        
        # 6. ARIMA (univariate, on the target only)
        print("\n--- Training ARIMA Model ---")
        # For ARIMA, we use the full y_train + y_val for fitting, then forecast len(y_test) steps
        y_trainval = np.concatenate([y_train, y_val])
        try:
            arima_order = (2, 1, 2)  # Reasonable default, can be tuned
            arima_model = ARIMA(y_trainval, order=arima_order)
            arima_fit = arima_model.fit()
            y_pred_arima = arima_fit.forecast(steps=len(y_test))
            metrics_arima = calculate_metrics(y_test, y_pred_arima)
        except Exception as e:
            print(f"ARIMA failed: {e}")
            y_pred_arima = np.zeros_like(y_test)
            metrics_arima = {k: float('nan') for k in ['MSE','RMSE','MAE','MBE','VAF','R2','MAPE']}
        
        # 7. GRU
        print("\n--- Training GRU Model ---")
        class GRUModel(nn.Module):
            def __init__(self, input_features, hidden_size=64, dropout=0.2):
                super().__init__()
                self.gru = nn.GRU(input_features, hidden_size, batch_first=True)
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.output_layers = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_hidden = gru_out[:, -1, :]
                last_hidden = self.layer_norm(last_hidden)
                return self.output_layers(last_hidden)
        gru_model = GRUModel(input_features=X_train.shape[-1], hidden_size=64, dropout=0.2).to(device)
        trainer_gru = ModelTrainer(gru_model, processor, device=device)
        hist_gru = trainer_gru.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
        metrics_gru = trainer_gru.evaluate(test_loader)
        
        # Print summary
        print("\n=== Comparative Results ===")
        if metrics_classical:
            print("Classical LSTM:")
            for k, v in metrics_classical.items():
                print(f"  {k}: {v}")
        if metrics_hybrid:
            print("\nHybridQuantumLSTM:")
            for k, v in metrics_hybrid.items():
                print(f"  {k}: {v}")
        if metrics_quantum:
            print("\nQuantum-Enhanced Model:")
            for k, v in metrics_quantum.items():
                print(f"  {k}: {v}")
        print("\nXGBoost:")
        for k, v in metrics_xgb.items():
            print(f"  {k}: {v}")
        print("\nCatBoost:")
        for k, v in metrics_cat.items():
            print(f"  {k}: {v}")
        print("\nARIMA:")
        for k, v in metrics_arima.items():
            print(f"  {k}: {v}")
        print("\nGRU:")
        for k, v in metrics_gru.items():
            print(f"  {k}: {v}")
        
        # Plot training curves for all deep models (including loaded ones)
        print("\nPlotting training curves...")
        if trainer_classical and hist_classical:
            trainer_classical.plot_training_curves(hist_classical['train_losses'], hist_classical['val_losses'], hist_classical['metrics_history'], save_path=None)
        if trainer_hybrid and hist_hybrid:
            trainer_hybrid.plot_training_curves(hist_hybrid['train_losses'], hist_hybrid['val_losses'], hist_hybrid['metrics_history'], save_path=None)
        if trainer_quantum and hist_quantum:
            trainer_quantum.plot_training_curves(hist_quantum['train_losses'], hist_quantum['val_losses'], hist_quantum['metrics_history'], save_path=None)
        trainer_gru.plot_training_curves(hist_gru['train_losses'], hist_gru['val_losses'], hist_gru['metrics_history'], save_path=None)
        
        # Save and plot all insights
        histories = {}
        if hist_classical:
            histories['ClassicalLSTM'] = hist_classical
        if hist_hybrid:
            histories['HybridQuantumLSTM'] = hist_hybrid
        if hist_quantum:
            histories['QuantumEnhanced'] = hist_quantum
        histories['GRU'] = hist_gru
        
        # Plot all models' loss/metrics on one plot
        for metric in ['train_losses', 'val_losses']:
            plot_all_models_curves(histories, metric, f'plots/all_models_{metric}.png')
        for metric in ['RMSE', 'MAE', 'MBE', 'VAF', 'MAPE']:
            plot_all_models_curves({k: [m[metric] for m in v["metrics_history"]] for k,v in histories.items()}, metric, f'plots/all_models_{metric}.png')
        
        # Pred vs True, Residuals, Boxplot, Bar chart for all models
        errors_dict = {}
        metrics_dict = {}
        if metrics_classical:
            metrics_dict['ClassicalLSTM'] = metrics_classical
        if metrics_hybrid:
            metrics_dict['HybridQuantumLSTM'] = metrics_hybrid
        if metrics_quantum:
            metrics_dict['QuantumEnhanced'] = metrics_quantum
        metrics_dict.update({
            'XGBoost': metrics_xgb,
            'CatBoost': metrics_cat,
            'ARIMA': metrics_arima,
            'GRU': metrics_gru
        })
        
        # Deep models (only if we have trainers)
        if trainer_classical:
            y_true, y_pred = trainer_classical.get_predictions(test_loader)
            plot_pred_vs_true(y_true, y_pred, 'ClassicalLSTM', f'plots/ClassicalLSTM_pred_vs_true.png')
            plot_residuals(y_true, y_pred, 'ClassicalLSTM', f'plots/ClassicalLSTM_residuals.png')
            errors_dict['ClassicalLSTM'] = np.abs(y_true - y_pred)
        if trainer_hybrid:
            y_true, y_pred = trainer_hybrid.get_predictions(test_loader)
            plot_pred_vs_true(y_true, y_pred, 'HybridQuantumLSTM', f'plots/HybridQuantumLSTM_pred_vs_true.png')
            plot_residuals(y_true, y_pred, 'HybridQuantumLSTM', f'plots/HybridQuantumLSTM_residuals.png')
            errors_dict['HybridQuantumLSTM'] = np.abs(y_true - y_pred)
        if trainer_quantum:
            y_true, y_pred = trainer_quantum.get_predictions(test_loader)
            plot_pred_vs_true(y_true, y_pred, 'QuantumEnhanced', f'plots/QuantumEnhanced_pred_vs_true.png')
            plot_residuals(y_true, y_pred, 'QuantumEnhanced', f'plots/QuantumEnhanced_residuals.png')
            errors_dict['QuantumEnhanced'] = np.abs(y_true - y_pred)
        
        # GRU
        y_true, y_pred = trainer_gru.get_predictions(test_loader)
        plot_pred_vs_true(y_true, y_pred, 'GRU', f'plots/GRU_pred_vs_true.png')
        plot_residuals(y_true, y_pred, 'GRU', f'plots/GRU_residuals.png')
        errors_dict['GRU'] = np.abs(y_true - y_pred)
        
        # XGBoost
        plot_pred_vs_true(y_test, y_pred_xgb, 'XGBoost', 'plots/XGBoost_pred_vs_true.png')
        plot_residuals(y_test, y_pred_xgb, 'XGBoost', 'plots/XGBoost_residuals.png')
        errors_dict['XGBoost'] = np.abs(y_test - y_pred_xgb)
        
        # CatBoost
        plot_pred_vs_true(y_test, y_pred_cat, 'CatBoost', 'plots/CatBoost_pred_vs_true.png')
        plot_residuals(y_test, y_pred_cat, 'CatBoost', 'plots/CatBoost_residuals.png')
        errors_dict['CatBoost'] = np.abs(y_test - y_pred_cat)
        
        # ARIMA
        plot_pred_vs_true(y_test, y_pred_arima, 'ARIMA', 'plots/ARIMA_pred_vs_true.png')
        plot_residuals(y_test, y_pred_arima, 'ARIMA', 'plots/ARIMA_residuals.png')
        errors_dict['ARIMA'] = np.abs(y_test - y_pred_arima)
        
        # Boxplot and bar chart
        plot_boxplot_errors(errors_dict, 'plots/all_models_error_boxplot.png')
        plot_bar_metrics(metrics_dict, 'plots/all_models_metrics_bar.png')

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
    for year in ['2022', '2023']:
        year_path = os.path.join('data/raw', year)
        if os.path.exists(year_path):
            for root, dirs, files in os.walk(year_path):
                for file in files:
                    if file.endswith('.csv'):
                        data_files.append(os.path.join(root, file))

    print(f"\nFound {len(data_files)} data files to process (2022+2023)")

    # Load and combine all our data files
    all_data = []
    for file in data_files:
        print(f"Loading {file}")
        df = pd.read_csv(file)
        all_data.append(df)

    # Combine all our data into one big dataset
    data = pd.concat(all_data, ignore_index=True)

    # Sort by date to maintain temporal order
    if 'measured_on' in data.columns:
        data = data.sort_values('measured_on').reset_index(drop=True)

    # Let's validate our data to make sure it looks good
    print("\nData Validation:")
    print("=" * 50)
    print(f"Total samples: {len(data)}")
    if 'measured_on' in data.columns:
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

    # Ensure all feature columns are numeric
    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

    # Drop any rows with NaNs after conversion
    data = data.dropna(subset=feature_columns + [target_column])
    print(f"\nSamples after numeric conversion and cleaning: {len(data)}")

    # Create time sequences for our LSTM model
    sequence_length = 24  # 24 hours of historical data
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][feature_columns].values.astype(np.float32))
        y.append(np.float32(data.iloc[i + sequence_length][target_column]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Stratified split by time: shuffle, then split into train/val/test (70/15/15)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Run our comprehensive pre-training checks
    if not runner.pre_training_check(X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nPre-training checks failed. Exiting...")
        return

    # If everything looks good, let's start training our models!
    print("\nStarting training process...")
    runner.run_comparative_study(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()