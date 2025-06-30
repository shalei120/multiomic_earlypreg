from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from .plotting import plot_roc_curve

__all__ = ["evaluate_binary"]


def evaluate_binary(model: torch.nn.Module, loader, plot_path: Optional[Path] = None) -> float:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            labels.append(y)
    p = torch.cat(preds).cpu().numpy()
    l = torch.cat(labels).cpu().numpy()
    auc = roc_auc_score(l, p)
    if plot_path:
        fpr, tpr, _ = roc_curve(l, p)
        plot_roc_curve(fpr, tpr, auc, plot_path)
    return auc
