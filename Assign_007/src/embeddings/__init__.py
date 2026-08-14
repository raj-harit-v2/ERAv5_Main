"""Embedding modules: dense, Kronecker, Fourier baseline, CANINE upgrade, gated VQ."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.embeddings.canine_vq import FourierCanineEmbedding, FourierVQEmbedding
from src.embeddings.fourier_baseline import FourierCodepointEmbedding, znormalize


class DenseEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.requires_token_text = False

    def forward(self, token_ids: torch.Tensor, token_strings: list[list[str]] | None = None) -> torch.Tensor:
        return self.table(token_ids)

    def projection_parameters(self) -> list[nn.Parameter]:
        return list(self.table.parameters())


class KroneckerByteEmbedding(nn.Module):
    """Session-7 style: frozen 256 x pos_dim byte grid + trainable Linear."""

    def __init__(self, d_model: int, pos_dim: int = 32):
        super().__init__()
        self.pos_dim = pos_dim
        self.code_dim = 256 * pos_dim
        self.proj = nn.Linear(self.code_dim, d_model, bias=True)
        self.requires_token_text = True
        self.register_buffer("byte_eye", torch.eye(256), persistent=False)
        self.register_buffer("pos_eye", torch.eye(pos_dim), persistent=False)

    def encode_string(self, text: str) -> torch.Tensor:
        raw = text.encode("utf-8")
        L = min(len(raw), self.pos_dim)
        code = torch.zeros(self.code_dim, dtype=torch.float32)
        if L == 0:
            return code
        acc = torch.zeros(256, self.pos_dim, dtype=torch.float32)
        for p in range(L):
            acc[raw[p], p] = 1.0
        flat = acc.reshape(-1) / math.sqrt(L)
        return znormalize(flat)

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
            raise ValueError("KroneckerByteEmbedding requires token_strings (seam-crossing)")
        codes = self.encode_batch_strings(token_strings).to(token_ids.device)
        return self.proj(codes)

    def projection_parameters(self) -> list[nn.Parameter]:
        return list(self.proj.parameters())


def build_embedding(
    kind: str,
    vocab_size: int,
    d_model: int,
    pad_id: int,
    pos_dim: int,
    fourier_code_dim: int,
    fourier_n_freq: int,
    max_chars: int,
    canine_stride: int = 2,
    vq_num_codes: int = 256,
) -> nn.Module:
    kind = kind.lower()
    if kind == "dense":
        return DenseEmbedding(vocab_size, d_model, pad_id=pad_id)
    if kind == "kronecker":
        return KroneckerByteEmbedding(d_model, pos_dim=pos_dim)
    if kind == "fourier":
        return FourierCodepointEmbedding(
            d_model, code_dim=fourier_code_dim, n_freq=fourier_n_freq, max_chars=max_chars
        )
    if kind in ("fourier_canine", "canine"):
        return FourierCanineEmbedding(
            d_model,
            code_dim=fourier_code_dim,
            n_freq=fourier_n_freq,
            max_chars=max_chars,
            stride=canine_stride,
        )
    if kind in ("fourier_vq", "vq"):
        return FourierVQEmbedding(
            d_model,
            code_dim=fourier_code_dim,
            n_freq=fourier_n_freq,
            max_chars=max_chars,
            num_codes=vq_num_codes,
            use_canine=True,
        )
    raise ValueError(f"unknown embedding kind: {kind}")


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def adamw_train_state_gb(n_params: int) -> float:
    return n_params * 16 / 1e9
