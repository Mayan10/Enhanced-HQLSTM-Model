#!/usr/bin/env python3
"""Regenerate training-curve and comparison plots from saved history/*.json files,
without retraining. Useful after editing plot styling or if plots/ was cleared."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hqlstm import plotting  # noqa: E402

DEFAULT_MODEL_FILES = {
    "ClassicalLSTM": "ClassicalLSTMModel_history.json",
    "HybridQuantumLSTM": "HybridHeadedModel_history.json",
    "QuantumEnhanced": "PVForecastingModel_history.json",
    "GRU": "GRUModel_history.json",
}


def load_history(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", default="history")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    plotting.set_plot_style()

    histories = {}
    for model_name, filename in DEFAULT_MODEL_FILES.items():
        path = os.path.join(args.history_dir, filename)
        if os.path.exists(path):
            histories[model_name] = load_history(path)
        else:
            print(f"Skipping {model_name}: {path} not found")

    if not histories:
        print(f"No history files found in {args.history_dir}")
        return

    for model_name, hist in histories.items():
        plotting.plot_training_curves(
            hist["train_losses"], hist["val_losses"], hist["metrics_history"],
            model_name=model_name,
            save_path=os.path.join(args.plots_dir, f"{model_name}_training_curves.png"),
        )

    for metric in ["train_losses", "val_losses"]:
        plotting.plot_all_models_curves(
            histories, metric, os.path.join(args.plots_dir, f"all_models_{metric}.png")
        )
    for metric in ["RMSE", "MAE", "VAF", "MAPE"]:
        plotting.plot_all_models_curves(
            {name: [m[metric] for m in hist["metrics_history"]] for name, hist in histories.items()},
            metric, os.path.join(args.plots_dir, f"all_models_{metric}.png"),
        )

    final_metrics = {name: hist["metrics_history"][-1] for name, hist in histories.items()}
    plotting.plot_bar_metrics(final_metrics, os.path.join(args.plots_dir, "all_models_metrics_bar.png"))

    print(f"Plots written to {args.plots_dir}/")


if __name__ == "__main__":
    main()
