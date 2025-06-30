"""Command-line wrapper for training the tabular MLP."""
from __future__ import annotations

import argparse

from src.training.train_tabular import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on early pregnancy data")
    parser.add_argument("--features", type=str, required=True, help="CSV of sample features")
    parser.add_argument("--cfrna", type=str, required=True, help="Excel file with cfRNA labels")
    parser.add_argument("--cfdna", type=str, required=True, help="Excel file with cfDNA labels")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--plot", type=str, default="", help="Path to save ROC curve")
    args = parser.parse_args()
    train_main(args)

