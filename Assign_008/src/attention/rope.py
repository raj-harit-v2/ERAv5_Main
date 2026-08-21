"""Minimal RoPE helpers (expanded 2D rotation; no pmatrix)."""

from __future__ import annotations

import torch


def rope_2d(x0: torch.Tensor, x1: torch.Tensor, m: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """R_m(x0,x1) = (cos(mθ)x0 - sin(mθ)x1, sin(mθ)x0 + cos(mθ)x1)."""
    c = torch.cos(torch.tensor(m * theta, dtype=x0.dtype, device=x0.device))
    s = torch.sin(torch.tensor(m * theta, dtype=x0.dtype, device=x0.device))
    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1
    return y0, y1


def apply_rope_pair_dim(x: torch.Tensor, theta: float = 1.0) -> torch.Tensor:
    """Apply RoPE to last even dims of [B,H,T,D] by rotating consecutive pairs."""
    *lead, d = x.shape
    assert d % 2 == 0
    out = x.clone()
    t = x.size(-2)
    for pos in range(t):
        for i in range(0, d, 2):
            a = out[..., pos, i]
            b = out[..., pos, i + 1]
            a2, b2 = rope_2d(a, b, pos, theta / (i // 2 + 1))
            out[..., pos, i] = a2
            out[..., pos, i + 1] = b2
    return out
