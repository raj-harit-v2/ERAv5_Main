"""CANINE-like char encoder on Fourier waves + optional VQ bridge for Problem #5."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings.fourier_baseline import FourierCodepointEmbedding, znormalize


class FourierCanineEmbedding(nn.Module):
    """
    Upgrade over naive Fourier sum:
    - per-char Fourier waves
    - local Conv1d over char axis (CANINE-like)
    - stride downsample + mean pool → one token code
    - Linear → d_model
    Ref: https://arxiv.org/abs/2103.06874
    """

    def __init__(
        self,
        d_model: int,
        code_dim: int = 512,
        n_freq: int = 32,
        max_chars: int = 64,
        stride: int = 2,
        conv_channels: int | None = None,
    ):
        super().__init__()
        self.base = FourierCodepointEmbedding(d_model=d_model, code_dim=code_dim, n_freq=n_freq, max_chars=max_chars)
        self.code_dim = code_dim
        self.max_chars = max_chars
        self.stride = max(1, stride)
        ch = conv_channels or min(code_dim, 256)
        self.char_proj = nn.Linear(code_dim, ch)
        self.local = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, stride=self.stride),
            nn.GELU(),
        )
        self.proj = nn.Linear(ch, d_model, bias=True)
        self.requires_token_text = True

    def char_wave_matrix(self, text: str) -> torch.Tensor:
        chars = list(text)[: self.max_chars]
        if not chars:
            return torch.zeros(1, self.code_dim)
        return torch.stack([self.base.wave(ord(ch), p) for p, ch in enumerate(chars)], dim=0)

    def encode_string(self, text: str) -> torch.Tensor:
        waves = self.char_wave_matrix(text)
        x = self.char_proj(waves).transpose(0, 1).unsqueeze(0)
        h = self.local(x)
        pooled = h.mean(dim=-1).squeeze(0)
        return znormalize(pooled)

    def encode_batch_strings(self, batch_tokens: list[list[str]]) -> torch.Tensor:
        B = len(batch_tokens)
        T = max((len(r) for r in batch_tokens), default=0)
        ch = self.proj.in_features
        out = torch.zeros(B, T, ch)
        for bi, row in enumerate(batch_tokens):
            for ti, tok in enumerate(row):
                out[bi, ti] = self.encode_string(tok)
        return out

    def forward(self, token_ids: torch.Tensor, token_strings: list[list[str]] | None = None) -> torch.Tensor:
        if token_strings is None:
            raise ValueError("FourierCanineEmbedding requires token_strings")
        codes = self.encode_batch_strings(token_strings).to(token_ids.device)
        return self.proj(codes)

    def projection_parameters(self) -> list[nn.Parameter]:
        return list(self.proj.parameters()) + list(self.char_proj.parameters()) + list(self.local.parameters())


class VectorQuantizer(nn.Module):
    """Straight-through VQ (VQ-VAE) for Problem #5 — gated off by default."""

    def __init__(self, code_dim: int, num_codes: int = 256, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.beta = beta
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.last_loss: torch.Tensor | None = None
        self.last_perplexity: float | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(-1, self.code_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        z_q = self.codebook(idx).view_as(z)
        self.last_loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z_q.detach(), z)
        enc = F.one_hot(idx, self.num_codes).float().mean(0)
        self.last_perplexity = float(torch.exp(-torch.sum(enc * torch.log(enc + 1e-10))).item())
        return z + (z_q - z).detach()


class FourierVQEmbedding(nn.Module):
    """Problem #5 prep only: CANINE-Fourier → VQ → projection. Not the #4 submission claim."""

    def __init__(
        self,
        d_model: int,
        code_dim: int = 512,
        n_freq: int = 32,
        max_chars: int = 64,
        num_codes: int = 256,
        use_canine: bool = True,
    ):
        super().__init__()
        self.use_canine = use_canine
        if use_canine:
            self.front = FourierCanineEmbedding(d_model, code_dim, n_freq, max_chars)
            self.vq = VectorQuantizer(self.front.proj.in_features, num_codes=num_codes)
            self.proj = self.front.proj
        else:
            self.front = FourierCodepointEmbedding(d_model, code_dim, n_freq, max_chars)
            self.vq = VectorQuantizer(code_dim, num_codes=num_codes)
            self.proj = self.front.proj
        self.requires_token_text = True

    def forward(self, token_ids: torch.Tensor, token_strings: list[list[str]] | None = None) -> torch.Tensor:
        if token_strings is None:
            raise ValueError("FourierVQEmbedding requires token_strings")
        codes = self.front.encode_batch_strings(token_strings).to(token_ids.device)
        return self.proj(self.vq(codes))

    def projection_parameters(self) -> list[nn.Parameter]:
        params = list(self.proj.parameters()) + list(self.vq.parameters())
        if self.use_canine:
            params += list(self.front.char_proj.parameters()) + list(self.front.local.parameters())
        return params

    @property
    def vq_loss(self) -> torch.Tensor | None:
        return self.vq.last_loss
