"""Embedding paths: dense (default) and chrono_stub.

After [B,T,D], Session 8 attention math is identical for both paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class DenseEmbedding(nn.Module):
    def __init__(self, vocab_size: int = config.VOCAB_SIZE, d_model: int = config.D_MODEL) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


class ChronoStubEmbedding(nn.Module):
    """Assumption A3: stub stand-in for Session 7 Chrono/Fourier path.

    Maps token ids through a small MLP so shape matches dense without a V5 table.
    """

    def __init__(self, vocab_size: int = config.VOCAB_SIZE, d_model: int = config.D_MODEL) -> None:
        super().__init__()
        self.code_dim = 32
        self.lookup = nn.Embedding(vocab_size, self.code_dim, padding_idx=config.PAD_ID)
        self.proj = nn.Linear(self.code_dim, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.lookup(input_ids))


def build_embedding(mode: str | None = None) -> nn.Module:
    mode = mode or config.EMBED_MODE
    if mode == "chrono_stub":
        return ChronoStubEmbedding()
    if mode != "dense":
        raise ValueError(f"Unknown ASSIGN008_EMBED={mode!r}; use dense|chrono_stub")
    return DenseEmbedding()


def v5_dense_param_count() -> int:
    """Account only — never allocate V5_REF_V x V5_REF_D."""
    return config.V5_REF_V * config.V5_REF_D
