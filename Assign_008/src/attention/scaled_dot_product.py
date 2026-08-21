"""Scaled dot-product attention with causal mask.

Correction: Full Document OCR is broken — use closed forms from formulas Cursor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(t: int, device: torch.device | None = None) -> torch.Tensor:
    """M[i,j] = 0 if j<=i else -inf (added before softmax)."""
    m = torch.zeros(t, t, device=device)
    future = torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
    m = m.masked_fill(future, float("-inf"))
    return m


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q,k,v: [B, H, T, d_k]
    returns out [B,H,T,d_v], weights [B,H,T,T]
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        t = scores.size(-1)
        scores = scores + causal_mask(t, device=scores.device)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        out, weights = scaled_dot_product_attention(q, k, v, causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.w_o(out), weights
