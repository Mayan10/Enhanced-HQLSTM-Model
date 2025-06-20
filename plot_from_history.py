import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# Helper to load history JSON files

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_all_models_curves(histories, metric, save_path):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (model_name, hist) in enumerate(histories.items()):
        # If hist is a list (metric curves), plot directly
        if isinstance(hist, list):
            plt.plot(hist, label=model_name, color=colors[i % len(colors)], linewidth=2)
        # If hist is a dict (loss curves), look for the metric key
        elif isinstance(hist, dict):
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

def plot_training_curves(train_losses: List[float], val_losses: List[float], metrics_history: List[Dict], save_path: str, model_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Losses')
    axes[0].legend()
    axes[0].grid(True)
    rmse_values = [m['RMSE'] for m in metrics_history]
    axes[1].plot(rmse_values, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Root Mean Square Error')
    axes[1].grid(True)
    vaf_values = [m['VAF'] for m in metrics_history]
    axes[2].plot(vaf_values, color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('VAF')
    axes[2].set_title('Variance Accounted For')
    axes[2].grid(True)
    mae_values = [m['MAE'] for m in metrics_history]
    axes[3].plot(mae_values, color='orange')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('MAE')
    axes[3].set_title('Mean Absolute Error')
    axes[3].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def plot_boxplot_errors(errors_dict, save_path):
    plt.figure(figsize=(10, 8))
    labels = list(errors_dict.keys())
    data = list(errors_dict.values())
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
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
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    exclude_metrics = {'y_true', 'y_pred'}
    plot_metrics = [m for m in all_metrics if m not in exclude_metrics]
    if not plot_metrics:
        print("Warning: No valid metrics found for bar plot")
        return
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
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height, f'{value:.4f}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def main():
    # Map model names to history files
    model_files = {
        'ClassicalLSTM': 'history/ClassicalLSTMModel_history.json',
        'HybridQuantumLSTM': 'history/HybridHeadedModel_history.json',
        'QuantumEnhanced': 'history/PVForecastingModel_history.json',
    }
    histories = {}
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            histories[model_name] = load_history(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping {model_name}")
    # Plot training curves for each model
    for model_name, hist in histories.items():
        plot_training_curves(hist['train_losses'], hist['val_losses'], hist['metrics_history'], f'plots/{model_name}_training_curves.png', model_name)
    # Plot all models' loss/metrics on one plot
    for metric in ['train_losses', 'val_losses']:
        plot_all_models_curves(histories, metric, f'plots/all_models_{metric}.png')
    for metric in ['RMSE', 'MAE', 'VAF', 'MAPE']:
        plot_all_models_curves({k: [m[metric] for m in v["metrics_history"]] for k,v in histories.items()}, metric, f'plots/all_models_{metric}.png')
    # Plot boxplot and bar chart for final metrics
    # Use last epoch's metrics for each model
    metrics_dict = {k: v['metrics_history'][-1] for k, v in histories.items()}
    # For boxplot, use absolute errors from last epoch if available
    errors_dict = {}
    for model_name, hist in histories.items():
        if 'metrics_history' in hist and 'y_true' in hist['metrics_history'][-1] and 'y_pred' in hist['metrics_history'][-1]:
            y_true = np.array(hist['metrics_history'][-1]['y_true'])
            y_pred = np.array(hist['metrics_history'][-1]['y_pred'])
            errors_dict[model_name] = np.abs(y_true - y_pred)
    if errors_dict:
        plot_boxplot_errors(errors_dict, 'plots/all_models_error_boxplot.png')
    plot_bar_metrics(metrics_dict, 'plots/all_models_metrics_bar.png')
    print('All plots generated in the plots/ directory.')

if __name__ == "__main__":
    main() 