"""~20M dense decoder-only LM (standard residual; Assign_0014 Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LMConfig:
    """Locked shape targeting 18e6-22e6 parameters with tied embeddings."""

    vocab_size: int = 8192
    n_embd: int = 384
    n_layer: int = 10
    n_head: int = 6
    n_inner: int = 1536
    dropout: float = 0.0


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
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_embd)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Dense FFN (GeLU). Upcycle maps this into MoE experts."""

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.n_inner, bias=False)
        self.proj = nn.Linear(cfg.n_inner, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


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


class TinyDenseLM(nn.Module):
    def __init__(self, cfg: LMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LMConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList([StandardBlock(c) for _ in range(c.n_layer)])
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
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def count_parameters(self) -> int:
        return count_parameters(self)
