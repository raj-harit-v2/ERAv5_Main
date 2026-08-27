"""Head stability helpers (document / optional smoke — not graded core)."""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_cap(logits: torch.Tensor, c: float = 30.0) -> torch.Tensor:
    """Gemma-style logit soft-capping: z <- c * tanh(z / c)."""
    return c * torch.tanh(logits / c)


def z_loss_penalty(logits: torch.Tensor, lam: float = 1e-4) -> torch.Tensor:
    """L += lam * (log Z)^2 where Z = sum_j exp(z_j)."""
    log_z = torch.logsumexp(logits, dim=-1)
    return lam * (log_z ** 2).mean()


def center_output_weights(weight: torch.Tensor) -> torch.Tensor:
    """
    Subtract row-mean of unembedding weights.
    Pins mean logit near 0; does NOT pin log Z (centering mistake correction).
    weight: [V, D]
    """
    return weight - weight.mean(dim=0, keepdim=True)


def explain_centering_vs_logz() -> str:
    return (
        "Output centering pins mean logit ≈ 0 but does not constrain logit spread; "
        "log Z can still grow. Use z-loss or soft-capping to bound log Z."
    )
