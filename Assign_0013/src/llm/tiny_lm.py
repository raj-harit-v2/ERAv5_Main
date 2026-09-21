"""~20M decoder-only LM: standard residual or reversible Euler/midpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.llm.reversible import euler_step, midpoint_step

Mode = Literal["standard", "euler", "midpoint"]


@dataclass(frozen=True)
class LMConfig:
    """Locked shape targeting 18e6–22e6 parameters with tied embeddings."""

    vocab_size: int = 8192
    n_embd: int = 384
    n_layer: int = 10
    n_head: int = 6
    n_inner: int = 1536
    dropout: float = 0.0
    h: float = 0.25
    mode: Mode = "standard"


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.n_inner, bias=False)
        self.proj = nn.Linear(cfg.n_inner, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class FBlock(nn.Module):
    """Whole-block vector field f_theta: attention then FFN."""

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attn(self.ln1(x))
        return self.mlp(self.ln2(y))


class StandardBlock(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyReversibleLM(nn.Module):
    def __init__(self, cfg: LMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LMConfig()
        c = self.cfg
        if c.mode != "standard" and c.dropout != 0.0:
            raise ValueError("reversible modes require dropout=0 (FULL §16)")
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        if c.mode == "standard":
            self.blocks = nn.ModuleList([StandardBlock(c) for _ in range(c.n_layer)])
        else:
            self.blocks = nn.ModuleList([FBlock(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor) -> Tensor:
        c = self.cfg
        x = self.embed(idx)
        if c.mode == "standard":
            for blk in self.blocks:
                x = blk(x)
            h = x
        elif c.mode == "euler":
            p_curr = x
            for blk in self.blocks:
                p_curr = euler_step(p_curr, blk, c.h)
            h = p_curr
        elif c.mode == "midpoint":
            p_prev = x
            p_curr = x
            for blk in self.blocks:
                p_next = midpoint_step(p_prev, p_curr, blk, c.h)
                p_prev = p_curr
                p_curr = p_next
            h = p_curr
        else:
            raise ValueError(f"unknown mode {c.mode}")
        return self.head(self.ln_f(h))

    def count_parameters(self) -> int:
        return count_parameters(self)
