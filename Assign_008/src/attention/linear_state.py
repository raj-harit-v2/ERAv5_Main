"""Linear attention state (softmax removed).

Assumption A4: session convention S = sum v k^T , y = S q
"""

from __future__ import annotations

import torch


def additive_write(s_prev: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """S_i = S_{i-1} + v k^T. Shapes: S [d_v,d_k], v [d_v], k [d_k]."""
    return s_prev + torch.outer(v, k)


def read_state(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """y = S q."""
    return s @ q


def direct_no_softmax(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """y = sum_j (q·k_j) v_j for toy vectors."""
    scores = keys @ q  # [T]
    return values.T @ scores  # [d_v]


def build_state(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """S = sum_j v_j k_j^T ; keys/values [T, d]."""
    d_v = values.size(1)
    d_k = keys.size(1)
    s = torch.zeros(d_v, d_k, dtype=values.dtype, device=values.device)
    for j in range(keys.size(0)):
        s = additive_write(s, values[j], keys[j])
    return s
