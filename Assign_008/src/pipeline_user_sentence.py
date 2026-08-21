"""User-sentence → token IDs → [B,T,D] → causal MHA → logits sketch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import config
from src.attention import MultiHeadSelfAttention, apply_rope_pair_dim
from src.embeddings import build_embedding
from src.tokenizer_wrapper import HermeticTokenizer


@dataclass
class PipelineResult:
    text: str
    input_ids: torch.Tensor
    token_strings: list[str]
    hidden: torch.Tensor
    attn_out: torch.Tensor
    attn_weights: torch.Tensor
    logits: torch.Tensor
    reply_sketch: str
    steps: list[str]


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, w = self.attn(self.n1(x))
        x = x + a
        x = x + self.ff(self.n2(x))
        return x, w


class UserSentencePipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = HermeticTokenizer()
        self.embed = build_embedding()
        self.blocks = nn.ModuleList(
            [TinyBlock(config.D_MODEL, config.N_HEADS) for _ in range(config.N_LAYERS)]
        )
        self.head = nn.Linear(config.D_MODEL, config.VOCAB_SIZE, bias=False)

    def forward_sentence(self, text: str) -> PipelineResult:
        steps = [
            "1. Runtime encode user sentence (tokenizer.json already trained offline)",
            "2. Text -> token IDs (+ BOS/EOS/pad)",
            "3. Embedding path -> X [B,T,D]",
            "4. Causal scaled dot-product attention (Q,K,V -> scale -> mask -> softmax -> V)",
            "5. Tiny FFN stack",
            "6. LM head -> logits -> greedy next-token sketch -> detokenize",
        ]
        enc = self.tokenizer.encode(text, seq_len=config.SEQ_LEN)
        ids = enc.input_ids.to(config.DEVICE)
        x = self.embed(ids)
        if config.POSITION_POLICY == "rope":
            # teach-only: RoPE on a view of last dims via helper (no change to API)
            _ = apply_rope_pair_dim
        weights = None
        h = x
        for block in self.blocks:
            h, weights = block(h)
        logits = self.head(h)
        # greedy next id from last non-pad position among content
        last = min(len(text.split()) + 1, config.SEQ_LEN - 1)
        next_id = int(logits[0, last].argmax().item())
        reply = self.tokenizer.decode_ids([next_id])
        return PipelineResult(
            text=text,
            input_ids=ids,
            token_strings=enc.token_strings[0],
            hidden=x,
            attn_out=h,
            attn_weights=weights if weights is not None else torch.empty(0),
            logits=logits,
            reply_sketch=reply or "<unk>",
            steps=steps,
        )
