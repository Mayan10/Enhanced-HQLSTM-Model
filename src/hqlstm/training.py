"""Training loop, validation, and checkpointing for a single model."""

import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PVDataProcessor
from .device import DEVICE
from .metrics import calculate_metrics, to_serializable


class ModelTrainer:
    """Trains a model with early stopping and per-model checkpointing, and evaluates it."""

    def __init__(
        self,
        model: nn.Module,
        processor: PVDataProcessor,
        device: torch.device = DEVICE,
        checkpoint_dir: str = "checkpoints",
        history_dir: str = "history",
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.best_model_state = None
        self.best_val_loss = float("inf")

        self.checkpoint_dir = checkpoint_dir
        self.history_dir = history_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self.model.__class__.__name__

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")

    @property
    def history_path(self) -> str:
        return os.path.join(self.history_dir, f"{self.model_name}_history.json")

    def load_checkpoint(self) -> bool:
        """Load this model's best saved weights, if a checkpoint exists. Returns success."""
        if not os.path.exists(self.checkpoint_path):
            return False
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        return True

    def load_history(self) -> Dict:
        with open(self.history_path, "r") as f:
            return json.load(f)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 75,
        lr: float = 0.001,
        patience: int = 20,
    ) -> Dict:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        train_losses, val_losses, metrics_history = [], [], []
        patience_counter = 0

        print(f"\nTraining {self.model_name} on {self.device}")
        print("=" * 50)

        epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        for epoch in epoch_pbar:
            self.model.train()
            train_loss = 0.0

            batch_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", position=1, leave=False
            )
            for batch_X, batch_y in batch_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader)

            val_loss, val_metrics = self._validate(val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                torch.save(self.best_model_state, self.checkpoint_path)
            else:
                patience_counter += 1

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            metrics_history.append(val_metrics)

            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_rmse": f"{val_metrics['RMSE']:.4f}",
                    "val_vaf": f"{val_metrics['VAF']:.4f}",
                }
            )

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics_history": metrics_history,
        }
        with open(self.history_path, "w") as f:
            json.dump(to_serializable(history), f, indent=2)

        return history

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict]:
        self.model.eval()
        val_loss = 0.0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X).squeeze()
                val_loss += criterion(predictions, batch_y).item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)

        predictions_orig = self.processor.inverse_transform_target(np.array(all_predictions))
        targets_orig = self.processor.inverse_transform_target(np.array(all_targets))
        metrics = calculate_metrics(targets_orig, predictions_orig)

        return val_loss, metrics

    def evaluate(self, test_loader: DataLoader) -> Dict:
        print(f"\nEvaluating {self.model_name} on test set")
        print("=" * 50)

        _, metrics = self._validate(test_loader, nn.MSELoss())

        print(f"VAF: {metrics['VAF']:.4f}  R2: {metrics['R2']:.4f}  "
              f"RMSE: {metrics['RMSE']:.4f}  MAE: {metrics['MAE']:.4f}  MAPE: {metrics['MAPE']:.2f}%")

        return metrics

    def get_predictions(self, test_loader: DataLoader):
        """Return (true, predicted) values on the original scale for the test set."""
        self.model.eval()
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X).squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        predictions_orig = self.processor.inverse_transform_target(np.array(all_predictions))
        targets_orig = self.processor.inverse_transform_target(np.array(all_targets))

        return targets_orig, predictions_orig
