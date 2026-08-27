"""Output / unembedding heads: dense tied/untied and low-rank factored."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseOutputHead(nn.Module):
    """z = h @ W^T  with W shape [V, D] (nn.Linear weight)."""

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        return self.proj.weight

    @weight.setter
    def weight(self, value: nn.Parameter) -> None:
        self.proj.weight = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D] -> logits [B, T, V]
        return self.proj(hidden)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LowRankOutputHead(nn.Module):
    """z = (h @ W_down) @ W_up with rank r << D."""

    def __init__(self, d_model: int, vocab_size: int, rank: int = 16, bias: bool = False):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compare_tied_untied_counts(d_model: int, vocab_size: int) -> dict[str, int]:
    """
    Accounting on a dense toy embed (assignment checklist #6).
    Tied shares one VxD table; untied pays embed + head.
    """
    embed = vocab_size * d_model
    head = vocab_size * d_model
    return {
        "tied_params": embed,  # single shared VxD
        "untied_params": embed + head,
        "d_model": d_model,
        "vocab_size": vocab_size,
    }
