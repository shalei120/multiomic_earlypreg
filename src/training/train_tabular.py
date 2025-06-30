from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch

from ..data.tabular_loader import load_tabular_data, TabularDataset
from ..models.mlp import MLP
from ..utils.metrics import evaluate_binary


def train_epoch(model: torch.nn.Module, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
    model.train()
    crit = torch.nn.BCELoss()
    total = 0.0
    for x, y in loader:
        optim.zero_grad()
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        optim.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


def main(args: argparse.Namespace) -> None:
    X, y = load_tabular_data(Path(args.features), Path(args.cfrna), Path(args.cfdna))
    dataset = TabularDataset(X, y)
    test_ratio = 0.25 if len(dataset) >= 8 else 0.5
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_ratio,
        stratify=y,
        random_state=42,
    )
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    model = MLP(X.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optim)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    auc = evaluate_binary(model, test_loader, Path(args.plot) if args.plot else None)
    print(f"Test AUROC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on tabular features")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--cfrna", type=str, required=True)
    parser.add_argument("--cfdna", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--plot", type=str, default="")
    main(parser.parse_args())
