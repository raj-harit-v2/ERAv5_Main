"""Tiny demo model for 32-rank ZeRO / DP lab (not 30B)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DemoConfig:
    vocab_size: int = 128
    d_model: int = 64
    n_layers: int = 2
    seq_len: int = 16
    batch: int = 4


class TinyDecoder(nn.Module):
    """Small decoder-style stack: embed → Linear blocks → lm head."""

    def __init__(self, cfg: DemoConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DemoConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        blocks = []
        for _ in range(c.n_layers):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(c.d_model),
                    nn.Linear(c.d_model, c.d_model * 2),
                    nn.GELU(),
                    nn.Linear(c.d_model * 2, c.d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = self.ln_f(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def synthetic_batch(
    cfg: DemoConfig,
    rank: int = 0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-seeded synthetic tokens so each rank sees a distinct micro-batch."""
    g = torch.Generator(device="cpu")
    g.manual_seed(10_000 + int(rank))
    x = torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq_len), generator=g)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq_len), generator=g)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, t, v = logits.shape
    return nn.functional.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t))
