"""Tiny causal Transformer LM for Session 9 harness (hermetic, no HF)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.output_head import DenseOutputHead


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoLM(nn.Module):
    """
    Small decoder-only LM.
    Default: untied output head. Use tie_weights=True to share embed with head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq: int = 256,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tie_weights = tie_weights
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = DenseOutputHead(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.max_seq = max_seq
        self._init_near_uniform()

    def _init_near_uniform(self) -> None:
        """Small weights so untrained logits ≈ 0 → PPL near V (assignment gate)."""
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

    def forward(
        self, tokens: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        if t > self.max_seq:
            raise ValueError(f"seq_len {t} > max_seq {self.max_seq}")
        pos = torch.arange(t, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits

    def count_head_params(self) -> int:
        return sum(p.numel() for p in self.lm_head.parameters())

    def count_embed_params(self) -> int:
        return self.tok_emb.weight.numel()
