"""Delta rule write: S_i = S_{i-1} + (v - S_{i-1} k) k^T.

Assumption A5 / Correction: S k = v identity needs ||k||_2 = 1 (or equivalent).
"""

from __future__ import annotations

import torch


def delta_write(s_prev: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    v_hat = s_prev @ k
    delta = v - v_hat
    return s_prev + torch.outer(delta, k)


def normalize_key(k: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return k / (k.norm() + eps)
