"""Hand-written chunked cross-entropy (default chunk_size=1024)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedCrossEntropy(nn.Module):
    """
    Compute CE over flattened tokens in chunks of size C.
    Same math as full CE; peak logits activation ~ C * V instead of T * V.
    """

    def __init__(self, chunk_size: int = 1024, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [N, V] or [B, T, V]
        targets: [N] or [B, T]
        Returns mean CE over non-ignored tokens.
        """
        if logits.dim() == 3:
            b, t, v = logits.shape
            logits = logits.reshape(b * t, v)
            targets = targets.reshape(b * t)
        n, v = logits.shape
        total = logits.new_zeros(())
        count = 0
        for i in range(0, n, self.chunk_size):
            sl = slice(i, min(i + self.chunk_size, n))
            chunk_logits = logits[sl]
            chunk_targets = targets[sl]
            # Per-token losses; ignore_index handled by F.cross_entropy
            loss_sum = F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            valid = (chunk_targets != self.ignore_index).sum().item()
            total = total + loss_sum
            count += valid
        if count == 0:
            return logits.new_zeros(())
        return total / count


def full_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.dim() == 3:
        b, t, v = logits.shape
        logits = logits.reshape(b * t, v)
        targets = targets.reshape(b * t)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="mean")


def shift_logits_and_targets(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard causal LM: predict tokens[:, 1:] from logits[:, :-1]."""
    return logits[:, :-1, :].contiguous(), tokens[:, 1:].contiguous()
