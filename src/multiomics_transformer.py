"""Command-line wrapper for the PTB transformer."""
from __future__ import annotations

import argparse

from src.training.train_transformer import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PTB transformer model")
    parser.add_argument("--cfrna", type=str, required=True, help="Path to cfRNA data")
    parser.add_argument("--cfdna", type=str, required=True, help="Path to cfDNA data")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--plot", type=str, default="", help="Path to save ROC curve image")
    args = parser.parse_args()
    train_main(args)

