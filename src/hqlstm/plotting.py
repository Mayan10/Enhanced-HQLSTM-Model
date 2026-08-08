"""Publication-style plots for training curves and comparative model evaluation."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

MODEL_COLORS = ["blue", "red", "green", "purple", "orange", "brown", "teal"]


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "png",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics_history: List[Dict],
    model_name: str,
    save_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    axes[0].plot(train_losses, label="Training Loss", color="blue")
    axes[0].plot(val_losses, label="Validation Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Losses")
    axes[0].legend()
    axes[0].grid(True)

    for ax, metric, color in zip(axes[1:], ["RMSE", "VAF", "MAE"], ["green", "purple", "orange"]):
        values = [m[metric] for m in metrics_history]
        ax.plot(values, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True)

    fig.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_all_models_curves(histories: Dict, metric: str, save_path: str):
    """Plot one curve per model for a shared metric (loss curves or per-epoch metrics)."""
    plt.figure(figsize=(12, 8))

    for i, (model_name, hist) in enumerate(histories.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        if isinstance(hist, list):
            plt.plot(hist, label=model_name, color=color, linewidth=2)
        elif isinstance(hist, dict) and metric in hist:
            plt.plot(hist[metric], label=model_name, color=color, linewidth=2)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.title(f"{metric} Comparison Across Models", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20, color="blue")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    r2 = r2_score(y_true, y_pred)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Predicted Values", fontsize=16)
    plt.title(f"Predicted vs True Values - {model_name}\nR2 = {r2:.4f}", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(residuals, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax1.set_xlabel("Residual (True - Predicted)", fontsize=14)
    ax1.set_ylabel("Frequency", fontsize=14)
    ax1.set_title(f"Residual Distribution - {model_name}", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color="green")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted Values", fontsize=14)
    ax2.set_ylabel("Residuals", fontsize=14)
    ax2.set_title(f"Residuals vs Predicted - {model_name}", fontsize=16)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_boxplot_errors(errors_dict: Dict[str, np.ndarray], save_path: str):
    plt.figure(figsize=(10, 8))

    labels = list(errors_dict.keys())
    data = list(errors_dict.values())
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)

    palette = ["lightblue", "lightgreen", "lightcoral", "khaki", "plum", "peachpuff", "lightgrey"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)

    plt.ylabel("Absolute Error", fontsize=16)
    plt.title("Error Distribution Comparison Across Models", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_bar_metrics(metrics_dict: Dict[str, Dict], save_path: str):
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())

    plot_metrics = sorted(all_metrics - {"y_true", "y_pred"})
    if not plot_metrics:
        print("Warning: no valid metrics found for bar plot")
        return

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 8))
    if len(plot_metrics) == 1:
        axes = [axes]

    model_names = list(metrics_dict.keys())
    palette = ["lightblue", "lightgreen", "lightcoral", "khaki", "plum", "peachpuff", "lightgrey"]

    for ax, metric in zip(axes, plot_metrics):
        values = [metrics_dict[model].get(metric, 0) for model in model_names]
        bars = ax.bar(model_names, values, color=palette[: len(model_names)], alpha=0.8)
        ax.set_title(metric, fontsize=16)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
