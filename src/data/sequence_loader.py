from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split as sk_split

__all__ = [
    "tokenize_sequence",
    "load_sequences",
    "load_cfrna_sequences",
    "load_cfdna_sequences",
    "OmicsDataset",
    "train_test_split_stratified",
    "ModelConfig",
]


def tokenize_sequence(seq: str, vocab: dict[str, int]) -> List[int]:
    """Convert a nucleotide sequence string to integer tokens."""
    return [vocab.get(ch, vocab["N"]) for ch in seq]


def load_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Load sequences and labels from a CSV file with ``sequence,label`` format."""
    seqs: List[str] = []
    labels: List[int] = []
    with path.open() as fh:
        next(fh)  # skip header
        for line in fh:
            seq, lab = line.strip().split(",")
            seqs.append(seq)
            labels.append(int(lab))
    return seqs, labels


def load_cfrna_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Load cfRNA sequences or gene expression tables."""
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols = [c.lower() for c in df.columns]
    if "sequence" in cols and "label" in cols:
        seqs = df["sequence"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
    else:
        seqs = df.iloc[:, 0].astype(str).tolist()
        labels = (df.iloc[:, 1] > 0).astype(int).tolist()
    return seqs, labels


def load_cfdna_sequences(path: Path) -> Tuple[List[str], List[int]]:
    """Load cfDNA sequences from CSV."""
    return load_sequences(path)


class OmicsDataset(Dataset):
    """Dataset of tokenized sequences and labels."""

    def __init__(self, sequences: List[str], labels: List[int], vocab: dict[str, int]):
        self.tokens = [torch.tensor(tokenize_sequence(s, vocab)) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def train_test_split_stratified(dataset: OmicsDataset, test_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """Split dataset with stratification to avoid missing labels."""
    indices = np.arange(len(dataset))
    labels = dataset.labels.numpy()
    train_idx, test_idx = sk_split(indices, test_size=test_ratio, stratify=labels, random_state=42)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


@dataclass
class ModelConfig:
    vocab_size: int = 5  # A C G T N
    embed_dim: int = 768
    num_layers: int = 8
    num_heads: int = 4
    ffn_dim: int = 2048
    dropout: float = 0.1
