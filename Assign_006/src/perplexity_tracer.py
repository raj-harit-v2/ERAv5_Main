"""Per-token perplexity tracing with tiered storage."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import config as cfg
from src.utils import write_json


class PerplexityTracer:
    def __init__(self, full_trace_limit: int = cfg.PPL_FULL_TRACE_SHARDS):
        self.full_trace_limit = full_trace_limit
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self.full_traces: list[dict[str, Any]] = []
        self.aggregates: list[dict[str, Any]] = []
        self.mastered_shards: list[str] = []
        self.eos_ppl: list[float] = []

    def record(
        self,
        *,
        shard_id: str,
        lane: str,
        curriculum_stage: str,
        per_token_loss: torch.Tensor,
        input_ids: torch.Tensor,
        eos_id: int,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """per_token_loss: [B, T-1]; ppl_t = exp(ce_t)."""
        with torch.no_grad():
            ce = per_token_loss.detach().float().cpu()
            ppl = torch.exp(ce.clamp(max=20))
            if mask is not None:
                m = mask.detach().float().cpu()
                vals = ppl[m > 0]
                avg_ppl = float(vals.mean()) if vals.numel() else float(ppl.mean())
            else:
                avg_ppl = float(ppl.mean())

            # EOS positions in labels = input_ids[:, 1:]
            labels = input_ids[:, 1:].detach().cpu()
            eos_mask = labels == eos_id
            if eos_mask.any():
                eos_vals = ppl[eos_mask]
                eos_avg = float(eos_vals.mean())
                self.eos_ppl.append(eos_avg)
            else:
                eos_avg = None

        key = (lane, curriculum_stage)
        self._counts[key] += 1
        store_full = self._counts[key] <= self.full_trace_limit

        rec: dict[str, Any] = {
            "shard_id": shard_id,
            "lane": lane,
            "curriculum_stage": curriculum_stage,
            "avg_ppl": avg_ppl,
            "ppl_at_eos": eos_avg,
            "mastered": avg_ppl < cfg.PPL_THRESHOLD_SKIP,
            "full_trace": store_full,
        }
        if rec["mastered"]:
            self.mastered_shards.append(shard_id)

        if store_full:
            rec["ppl_trace"] = ppl.flatten()[:256].tolist()  # capped
            self.full_traces.append(rec)
        else:
            self.aggregates.append(rec)
        return rec

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "full_traces": self.full_traces,
                "aggregates": self.aggregates,
                "mastered_shards": self.mastered_shards,
                "eos_ppl_series": self.eos_ppl,
                "threshold": cfg.PPL_THRESHOLD_SKIP,
            },
        )


def expected_initial_loss(vocab_size: int) -> float:
    return math.log(vocab_size)
