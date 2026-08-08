"""Orchestrates the full comparative study across neural and non-neural models."""

import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import baselines, plotting
from .data import PVDataProcessor
from .device import DEVICE
from .models import ClassicalLSTMModel, GRUModel, HybridHeadedModel, PVForecastingModel
from .training import ModelTrainer


class ExperimentRunner:
    """Trains (or loads) each model in the comparison, evaluates it on a held-out test set,
    and produces the full set of comparative plots."""

    def __init__(self, plots_dir: str = "plots", checkpoint_dir: str = "checkpoints",
                 history_dir: str = "history"):
        self.plots_dir = plots_dir
        self.checkpoint_dir = checkpoint_dir
        self.history_dir = history_dir
        os.makedirs(plots_dir, exist_ok=True)
        plotting.set_plot_style()

    def pre_training_check(self, X_train, X_val, X_test, y_train, y_val, y_test) -> bool:
        """Validate data shapes, ranges, and a test model/forward pass before committing to training."""
        print("Running pre-training checks...")
        print("=" * 50)

        try:
            assert X_train.ndim == 3 and X_val.ndim == 3 and X_test.ndim == 3
            assert y_train.ndim == 1 and y_val.ndim == 1 and y_test.ndim == 1
        except AssertionError:
            print("Data shape check failed: expected 3D features and 1D targets")
            return False

        for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                           ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Data quality check failed: NaN or infinite values in {name}")
                return False

        print(f"X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}], "
              f"mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

        if y_train.var() < 1e-6:
            print(f"Warning: very low target variance ({y_train.var():.2e})")

        try:
            test_model = PVForecastingModel(
                input_features=X_train.shape[-1], hidden_size=32, n_qubits=2
            ).to(DEVICE)
            test_input = torch.randn(2, X_train.shape[1], X_train.shape[2], device=DEVICE)
            with torch.no_grad():
                test_output = test_model(test_input)
            assert test_output.shape == (2, 1)
        except Exception as exc:
            print(f"Model forward pass check failed: {exc}")
            return False

        try:
            processor = PVDataProcessor()
            X_scaled, y_scaled = processor.fit_transform(X_train[:10], y_train[:10])
            processor.transform(X_val[:5], y_val[:5])
            processor.inverse_transform_target(y_scaled)
        except Exception as exc:
            print(f"Data processor check failed: {exc}")
            return False

        print("All pre-training checks passed.")
        return True

    def _get_or_train(self, model, train_loader, val_loader, test_loader,
                       processor, epochs, lr, patience, skip_if_trained):
        """Return (trainer, history, metrics) for a model, training it unless a checkpoint
        with matching weights can be loaded from disk."""
        trainer = ModelTrainer(
            model, processor, device=DEVICE,
            checkpoint_dir=self.checkpoint_dir, history_dir=self.history_dir,
        )

        if skip_if_trained and trainer.load_checkpoint():
            print(f"\nLoaded existing checkpoint for {trainer.model_name}, skipping training")
            history = trainer.load_history() if os.path.exists(trainer.history_path) else None
        else:
            print(f"\n--- Training {trainer.model_name} ---")
            history = trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)

        metrics = trainer.evaluate(test_loader)
        return trainer, history, metrics

    def run_comparative_study(
        self, X_train, X_val, X_test, y_train, y_val, y_test,
        epochs: int = 75, lr: float = 0.001, patience: int = 20, skip_trained_models: bool = True,
    ):
        print("\n=== Comparative Study ===\n")
        processor = PVDataProcessor()
        X_train_scaled, y_train_scaled = processor.fit_transform(X_train, y_train)
        X_val_scaled, y_val_scaled = processor.transform(X_val, y_val)
        X_test_scaled, y_test_scaled = processor.transform(X_test, y_test)

        train_loader = DataLoader(
            TensorDataset(X_train_scaled, y_train_scaled), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_scaled, y_val_scaled), batch_size=32, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test_scaled, y_test_scaled), batch_size=32, shuffle=False
        )

        n_features = X_train.shape[-1]
        train_args = (train_loader, val_loader, test_loader, processor, epochs, lr, patience,
                      skip_trained_models)

        trainer_classical, hist_classical, metrics_classical = self._get_or_train(
            ClassicalLSTMModel(input_features=n_features, hidden_size=64, dropout=0.2), *train_args
        )
        trainer_hybrid, hist_hybrid, metrics_hybrid = self._get_or_train(
            HybridHeadedModel(input_features=n_features, hidden_size=64, n_qubits=4,
                               n_quantum_layers=1, encoding_type="angle"),
            *train_args,
        )
        trainer_quantum, hist_quantum, metrics_quantum = self._get_or_train(
            PVForecastingModel(input_features=n_features, hidden_size=64, n_qubits=4), *train_args
        )
        trainer_gru, hist_gru, metrics_gru = self._get_or_train(
            GRUModel(input_features=n_features, hidden_size=64, dropout=0.2), *train_args
        )

        print("\n--- Training XGBoost ---")
        metrics_xgb, y_pred_xgb = baselines.run_xgboost(X_train, y_train, X_test, y_test)

        print("\n--- Training CatBoost ---")
        metrics_cat, y_pred_cat = baselines.run_catboost(X_train, y_train, X_test, y_test)

        print("\n--- Training ARIMA ---")
        metrics_arima, y_pred_arima = baselines.run_arima(y_train, y_val, y_test)

        self._report_summary({
            "ClassicalLSTM": metrics_classical,
            "HybridQuantumLSTM": metrics_hybrid,
            "QuantumEnhanced": metrics_quantum,
            "GRU": metrics_gru,
            "XGBoost": metrics_xgb,
            "CatBoost": metrics_cat,
            "ARIMA": metrics_arima,
        })

        self._plot_all(
            trainers={
                "ClassicalLSTM": (trainer_classical, hist_classical),
                "HybridQuantumLSTM": (trainer_hybrid, hist_hybrid),
                "QuantumEnhanced": (trainer_quantum, hist_quantum),
                "GRU": (trainer_gru, hist_gru),
            },
            metrics_dict={
                "ClassicalLSTM": metrics_classical,
                "HybridQuantumLSTM": metrics_hybrid,
                "QuantumEnhanced": metrics_quantum,
                "GRU": metrics_gru,
                "XGBoost": metrics_xgb,
                "CatBoost": metrics_cat,
                "ARIMA": metrics_arima,
            },
            baseline_predictions={
                "XGBoost": y_pred_xgb,
                "CatBoost": y_pred_cat,
                "ARIMA": y_pred_arima,
            },
            test_loader=test_loader,
            y_test=y_test,
        )

    @staticmethod
    def _report_summary(metrics_dict: Dict[str, Dict]):
        print("\n=== Comparative Results ===")
        for name, metrics in metrics_dict.items():
            print(f"\n{name}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

    def _plot_all(self, trainers, metrics_dict, baseline_predictions, test_loader, y_test):
        print("\nPlotting training curves...")
        for name, (trainer, history) in trainers.items():
            if history is None:
                continue
            plotting.plot_training_curves(
                history["train_losses"], history["val_losses"], history["metrics_history"],
                model_name=name, save_path=f"{self.plots_dir}/{name}_training_curves.png",
            )

        histories = {name: hist for name, (_, hist) in trainers.items() if hist is not None}
        for metric in ["train_losses", "val_losses"]:
            plotting.plot_all_models_curves(histories, metric, f"{self.plots_dir}/all_models_{metric}.png")
        for metric in ["RMSE", "MAE", "MBE", "VAF", "MAPE"]:
            plotting.plot_all_models_curves(
                {name: [m[metric] for m in hist["metrics_history"]] for name, hist in histories.items()},
                metric, f"{self.plots_dir}/all_models_{metric}.png",
            )

        errors_dict = {}
        for name, (trainer, _) in trainers.items():
            y_true, y_pred = trainer.get_predictions(test_loader)
            plotting.plot_pred_vs_true(y_true, y_pred, name, f"{self.plots_dir}/{name}_pred_vs_true.png")
            plotting.plot_residuals(y_true, y_pred, name, f"{self.plots_dir}/{name}_residuals.png")
            errors_dict[name] = np.abs(y_true - y_pred)

        for name, y_pred in baseline_predictions.items():
            plotting.plot_pred_vs_true(y_test, y_pred, name, f"{self.plots_dir}/{name}_pred_vs_true.png")
            plotting.plot_residuals(y_test, y_pred, name, f"{self.plots_dir}/{name}_residuals.png")
            errors_dict[name] = np.abs(y_test - y_pred)

        plotting.plot_boxplot_errors(errors_dict, f"{self.plots_dir}/all_models_error_boxplot.png")
        plotting.plot_bar_metrics(metrics_dict, f"{self.plots_dir}/all_models_metrics_bar.png")
