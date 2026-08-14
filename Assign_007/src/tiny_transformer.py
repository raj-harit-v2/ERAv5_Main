"""Tiny causal Transformer with RoPE (default) or absolute PE."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rope import apply_rope, build_rope_cache


class RopeCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,T,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = build_rope_cache(T, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if key_padding is not None:
            # key_padding: [B,T] True = pad
            att = att.masked_fill(key_padding[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RopeCausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding=key_padding)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        embed: nn.Module,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        pad_id: int = 0,
        position_policy: str = "rope",
    ):
        super().__init__()
        self.embed = embed
        self.position_policy = position_policy
        self.pos = nn.Embedding(seq_len, d_model) if position_policy == "absolute" else None
        self.blocks = nn.ModuleList([Block(d_model, n_heads, seq_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # untied
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    def forward(
        self,
        token_ids: torch.Tensor,
        token_strings: list[list[str]] | None = None,
    ) -> torch.Tensor:
        B, T = token_ids.shape
        if T > self.seq_len:
            token_ids = token_ids[:, : self.seq_len]
            T = self.seq_len
            if token_strings is not None:
                token_strings = [row[:T] for row in token_strings]
        if getattr(self.embed, "requires_token_text", False):
            x = self.embed(token_ids, token_strings)
        else:
            x = self.embed(token_ids)
        if self.pos is not None:
            pos_ids = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
            x = x + self.pos(pos_ids)
        key_padding = token_ids.eq(self.pad_id)
        for blk in self.blocks:
            x = blk(x, key_padding=key_padding)
        x = self.ln_f(x)
        return self.lm_head(x)

    @property
    def stack(self):
        """Compat for diagnosis that looks at first layer grads."""
        return self

    @property
    def layers(self):
        return self.blocks

    def loss(
        self,
        token_ids: torch.Tensor,
        token_strings: list[list[str]] | None = None,
    ) -> torch.Tensor:
        logits = self.forward(token_ids[:, :-1], None if token_strings is None else [r[:-1] for r in token_strings])
        targets = token_ids[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_id,
        )
