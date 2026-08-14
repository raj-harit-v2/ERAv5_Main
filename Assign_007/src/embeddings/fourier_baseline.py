"""Fourier baseline codec (Problem #4) — shared by CANINE/VQ upgrades."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def znormalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class FourierCodepointEmbedding(nn.Module):
    """Problem #4 baseline: each Unicode codepoint -> Fourier wave; sum to make token code."""

    def __init__(self, d_model: int, code_dim: int = 512, n_freq: int = 16, max_chars: int = 64):
        super().__init__()
        if code_dim % 2 != 0:
            raise ValueError("code_dim must be even (sin/cos pairs)")
        self.code_dim = code_dim
        self.n_freq = n_freq
        self.max_chars = max_chars
        self.proj = nn.Linear(code_dim, d_model, bias=True)
        self.requires_token_text = True
        half = code_dim // 2
        self.register_buffer(
            "base_freqs",
            torch.linspace(1.0, float(n_freq), steps=half),
            persistent=False,
        )

    def _omega(self, codepoint: int) -> float:
        return 0.5 + ((codepoint * 2654435761) % 8000) / 1000.0

    def wave(self, codepoint: int, position: int) -> torch.Tensor:
        omega = self._omega(codepoint)
        p = float(position + 1)
        angles = self.base_freqs * omega * p * 0.1
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)

    def encode_string(self, text: str) -> torch.Tensor:
        chars = list(text)[: self.max_chars]
        L = len(chars)
        code = torch.zeros(self.code_dim, dtype=torch.float32)
        if L == 0:
            return code
        for p, ch in enumerate(chars):
            code = code + self.wave(ord(ch), p)
        code = code / math.sqrt(L)
        return znormalize(code)

    def encode_batch_strings(self, batch_tokens: list[list[str]]) -> torch.Tensor:
        B = len(batch_tokens)
        T = max((len(r) for r in batch_tokens), default=0)
        out = torch.zeros(B, T, self.code_dim)
        for bi, row in enumerate(batch_tokens):
            for ti, tok in enumerate(row):
                out[bi, ti] = self.encode_string(tok)
        return out

    def forward(self, token_ids: torch.Tensor, token_strings: list[list[str]] | None = None) -> torch.Tensor:
        if token_strings is None:
            raise ValueError("FourierCodepointEmbedding requires token_strings")
        codes = self.encode_batch_strings(token_strings).to(token_ids.device)
        return self.proj(codes)

    def projection_parameters(self) -> list[nn.Parameter]:
        return list(self.proj.parameters())
