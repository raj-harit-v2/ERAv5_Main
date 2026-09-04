"""Minimal GPT-style model for Session 10 training-loop instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniGPTConfig:
    vocab_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 128


class MiniGPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Small decoder-only LM: embed -> blocks -> lm_head."""

    def __init__(self, config: MiniGPTConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniGPTConfig()
        c = self.config
        self.token_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(
            MiniGPTBlock(c.n_embd, c.n_head) for _ in range(c.n_layer)
        )
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss_on_batch(self, idx: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        logits_flat = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
        targets = idx[:, 1:].reshape(-1)
        return F.cross_entropy(logits_flat, targets)
