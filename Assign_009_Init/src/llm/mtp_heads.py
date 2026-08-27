"""Multi-token prediction: Head1 t+1, Head2 t+2 (assignment Part 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.output_head import DenseOutputHead


class MTPDualHeads(nn.Module):
    """
    Shared trunk hidden h_t.
    Head1 predicts token t+1; Head2 predicts token t+2.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head1 = DenseOutputHead(d_model, vocab_size, bias=False)
        self.head2 = DenseOutputHead(d_model, vocab_size, bias=False)

    def forward_losses(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        ignore_index: int = -100,
    ) -> dict[str, torch.Tensor]:
        """
        hidden: [B, T, D]
        tokens: [B, T]
        """
        # Head1: positions 0..T-2 predict tokens 1..T-1
        logits1 = self.head1(hidden[:, :-1, :])
        targets1 = tokens[:, 1:].contiguous()
        loss1 = F.cross_entropy(
            logits1.reshape(-1, logits1.size(-1)),
            targets1.reshape(-1),
            ignore_index=ignore_index,
        )
        # Head2: positions 0..T-3 predict tokens 2..T-1
        logits2 = self.head2(hidden[:, :-2, :])
        targets2 = tokens[:, 2:].contiguous()
        loss2 = F.cross_entropy(
            logits2.reshape(-1, logits2.size(-1)),
            targets2.reshape(-1),
            ignore_index=ignore_index,
        )
        return {
            "loss1": loss1,
            "loss2": loss2,
            "loss_sum": loss1 + loss2,
            "logits1": logits1,
            "logits2": logits2,
        }
