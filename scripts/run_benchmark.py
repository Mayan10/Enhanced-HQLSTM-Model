#!/usr/bin/env python3
"""Run the full comparative benchmark: load PV data, run pre-training checks, then train
(or load cached checkpoints for) every model and produce the comparison plots."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hqlstm.data import load_pv_dataset  # noqa: E402
from hqlstm.experiment import ExperimentRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="directory containing <year>/ CSV folders")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retrain", action="store_true",
        help="retrain every neural model even if a checkpoint already exists",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="use a random subset of this many sequences instead of the full dataset "
             "(for a quick end-to-end run before committing to full-scale training)",
    )
    args = parser.parse_args()

    print("Loading and preparing PV dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_pv_dataset(
        raw_dir=args.data_dir, sequence_length=args.sequence_length, seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    runner = ExperimentRunner()

    if not runner.pre_training_check(X_train, X_val, X_test, y_train, y_val, y_test):
        print("Pre-training checks failed, aborting.")
        return

    runner.run_comparative_study(
        X_train, X_val, X_test, y_train, y_val, y_test,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        skip_trained_models=not args.retrain,
    )


if __name__ == "__main__":
    main()
