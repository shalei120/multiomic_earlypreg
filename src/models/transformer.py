from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn

__all__ = ["ModelConfig", "PTBTransformer", "collate_tokens"]


@dataclass
class ModelConfig:
    vocab_size: int = 5
    embed_dim: int = 768
    num_layers: int = 8
    num_heads: int = 4
    ffn_dim: int = 2048
    dropout: float = 0.1


def collate_tokens(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs, labels = zip(*batch)
    max_len = max(len(x) for x in seqs)
    padded = [torch.cat([x, x.new_zeros(max_len - len(x))]) for x in seqs]
    return torch.stack(padded), torch.stack(labels)


class PTBTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers)
        self.cls = nn.Linear(config.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out = self.encoder(emb)
        pooled = out.mean(dim=1)
        return torch.sigmoid(self.cls(pooled)).squeeze(-1)
