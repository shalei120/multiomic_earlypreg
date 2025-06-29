# MIT License
# Copyright (c) 2025 Lei Sha
"""Transformer-based model for multi-omics PTB prediction.

This script implements preprocessing utilities for cfRNA and cfDNA data
and defines a transformer neural network to integrate these modalities.
The architecture is based on the description provided in the
project documentation.

The code relies on PyTorch for model implementation. Data loading
functions are placeholders and should be replaced with project-specific
logic for reading sequencing data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Data preprocessing utilities
# -------------------------------------------------------------

def tokenize_sequence(seq: str, vocab: dict[str, int]) -> List[int]:
    """Convert a nucleotide sequence to integer tokens."""
    return [vocab.get(ch, vocab["N"]) for ch in seq]


def load_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Load sequences and labels from a CSV file.

    The expected format is ``sequence,label`` per line with a header row.

    Parameters
    ----------
    path : Path
        Path to the CSV file containing sequences and labels.

    Returns
    -------
    Tuple[List[str], List[int]]
        Lists of sequences and integer labels.
    """
    sequences: List[str] = []
    labels: List[int] = []
    with path.open() as fh:
        next(fh)  # skip header
        for line in fh:
            seq, lab = line.strip().split(",")
            sequences.append(seq)
            labels.append(int(lab))
    return sequences, labels


def load_cfrna_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Wrapper for backward compatibility."""
    return load_sequences(path)


def load_cfdna_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Wrapper for backward compatibility."""
    return load_sequences(path)


class OmicsDataset(Dataset):
    """PyTorch dataset for tokenized omics sequences and labels."""

    def __init__(self, sequences: List[str], labels: List[int], vocab: dict[str, int]):
        self.tokens = [torch.tensor(tokenize_sequence(s, vocab)) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def train_test_split(dataset: Dataset, test_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """Split a dataset into train and test subsets."""
    indices = np.random.permutation(len(dataset))
    test_size = int(len(dataset) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# -------------------------------------------------------------
# Model definition
# -------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 5  # A,C,G,T,N
    embed_dim: int = 768
    num_layers: int = 6
    num_heads: int = 4
    ffn_dim: int = 2000
    dropout: float = 0.1


class PTBTransformer(nn.Module):
    """Transformer encoder followed by a linear classifier."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.cls = nn.Linear(config.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        enc = self.transformer(emb)
        pooled = enc.mean(dim=1)
        out = torch.sigmoid(self.cls(pooled))
        return out.squeeze(-1)


# -------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences in the batch to equal length and stack labels."""
    seqs, labels = zip(*batch)
    lengths = [len(x) for x in seqs]
    max_len = max(lengths)
    padded = [torch.cat([x, x.new_zeros(max_len - len(x))]) for x in seqs]
    return torch.stack(padded), torch.stack(labels)


def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()
    for seqs, labels in loader:
        optim.zero_grad()
        preds = model(seqs)
        loss = criterion(preds, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(seqs)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, plot_path: Path | None = None) -> float:
    """Evaluate on the test set and optionally save a ROC curve."""
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for seqs, labs in loader:
            preds.append(model(seqs))
            labels.append(labs)
    preds_t = torch.cat(preds).cpu().numpy()
    labels_t = torch.cat(labels).cpu().numpy()
    auc = roc_auc_score(labels_t, preds_t)
    if plot_path is not None:
        fpr, tpr, _ = roc_curve(labels_t, preds_t)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    return auc


# -------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    cfrna_seqs, cfrna_labels = load_cfrna_sequences(Path(args.cfrna))
    cfdna_seqs, cfdna_labels = load_cfdna_sequences(Path(args.cfdna))
    sequences = cfrna_seqs + cfdna_seqs
    labels = cfrna_labels + cfdna_labels

    dataset = OmicsDataset(sequences, labels, vocab)
    train_ds, test_ds = train_test_split(dataset, test_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    config = ModelConfig()
    model = PTBTransformer(config)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optim)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    auc = evaluate(model, test_loader, plot_path=Path(args.plot) if args.plot else None)
    print(f"Test AUROC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PTB transformer model")
    parser.add_argument("--cfrna", type=str, required=True, help="Path to cfRNA data")
    parser.add_argument("--cfdna", type=str, required=True, help="Path to cfDNA data")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--plot", type=str, default="", help="Path to save ROC curve image")
    main(parser.parse_args())
