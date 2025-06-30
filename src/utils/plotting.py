from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

__all__ = ["plot_roc_curve"]


def plot_roc_curve(fpr, tpr, auc: float, path: Path) -> None:
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
