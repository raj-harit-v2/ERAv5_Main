"""Tiny PyTorch LM for smoke-scale training."""
from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = cfg.SMOKE_VOCAB_SIZE,
        d_model: int = cfg.SMOKE_D_MODEL,
        n_layers: int = cfg.SMOKE_N_LAYERS,
        n_heads: int = cfg.SMOKE_N_HEADS,
        max_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t = input_ids.shape
        if t > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            t = self.max_len
            if loss_mask is not None:
                loss_mask = loss_mask[:, : self.max_len]
            if position_ids is not None:
                position_ids = position_ids[:, : self.max_len]
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask[:, :t, :t]

        if position_ids is None:
            position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        position_ids = position_ids.clamp(0, self.max_len - 1)

        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        # Default causal; optional block-diagonal (1=allow) → True means masked/blocked for Transformer
        causal = torch.triu(torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1)
        if attention_mask is not None and attention_mask.dim() == 3:
            # attention_mask: 1 = may attend; nn.TransformerEncoder mask True = blocked
            allow = attention_mask[:, :t, :t].bool()
            mask = causal.unsqueeze(0) | (~allow)
            outs = []
            for i in range(b):
                outs.append(self.blocks(x[i : i + 1], mask=mask[i]))
            x = torch.cat(outs, dim=0)
        else:
            x = self.blocks(x, mask=causal)
        x = self.ln(x)
        logits = self.head(x)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_tok = nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(b, t - 1)

        if loss_mask is None:
            mask = torch.ones(b, t - 1, device=input_ids.device)
        else:
            mask = loss_mask[:, 1:].float()
        pad_mask = (shift_labels != 0).float()
        mask = mask * pad_mask
        denom = mask.sum().clamp_min(1.0)
        loss = (per_tok * mask).sum() / denom
        return {"loss": loss, "logits": logits, "per_token_loss": per_tok, "mask": mask}
