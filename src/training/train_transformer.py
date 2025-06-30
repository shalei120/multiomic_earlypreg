from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.sequence_loader import (
    load_cfrna_sequences,
    load_cfdna_sequences,
    OmicsDataset,
    train_test_split_stratified,
    ModelConfig,
)
from ..models.transformer import PTBTransformer, collate_tokens
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
    vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    cfrna_seqs, cfrna_labels = load_cfrna_sequences(Path(args.cfrna))
    cfdna_seqs, cfdna_labels = load_cfdna_sequences(Path(args.cfdna))
    sequences = cfrna_seqs + cfdna_seqs
    labels = cfrna_labels + cfdna_labels

    dataset = OmicsDataset(sequences, labels, vocab)
    train_ds, test_ds = train_test_split_stratified(dataset, 0.2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_tokens)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_tokens)

    model = PTBTransformer(ModelConfig())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optim)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    auc = evaluate_binary(model, test_loader, Path(args.plot) if args.plot else None)
    print(f"Test AUROC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-omics transformer")
    parser.add_argument("--cfrna", type=str, required=True)
    parser.add_argument("--cfdna", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--plot", type=str, default="")
    main(parser.parse_args())
