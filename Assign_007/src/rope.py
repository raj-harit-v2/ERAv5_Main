"""Rotary Position Embeddings (RoPE) — Su et al., https://arxiv.org/abs/2104.09864"""
from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, half]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, Dh]; cos/sin: [T, Dh/2]"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # broadcast cos/sin to [1,1,T,Dh/2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    rot = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return rot.flatten(-2)
