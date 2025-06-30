from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["load_tabular_data", "TabularDataset"]


def load_tabular_data(features_csv: Path, cfrna_csv: Path, cfdna_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load gene-count features and labels based on sample identifiers."""
    feat_df = pd.read_csv(features_csv)
    X = feat_df.set_index("gene_id").T

    rna_df = pd.read_csv(cfrna_csv)
    dna_df = pd.read_csv(cfdna_csv)
    lbl_df = pd.merge(rna_df, dna_df, on="sample_id", how="outer", suffixes=("_rna", "_dna"))
    label = lbl_df["label_rna"].fillna(lbl_df.get("label_dna"))

    samples = lbl_df["sample_id"].astype(str)
    X = X.loc[samples].values
    y = label.values.astype(int)
    return X, y


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
