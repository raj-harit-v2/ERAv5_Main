"""Metrics, token budget, peak memory, and cloud-cost helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


N_PARAMS_MIN = 18_000_000
N_PARAMS_MAX = 22_000_000
FULL_TOKEN_BUDGET = 50_000_000


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "mps"
    return "cpu"


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def tokens_per_step(batch: int, seq_len: int) -> int:
    return int(batch) * int(seq_len)


def steps_for_budget(token_budget: int, batch: int, seq_len: int) -> int:
    tps = tokens_per_step(batch, seq_len)
    if tps <= 0:
        raise ValueError("batch * seq_len must be > 0")
    return (int(token_budget) + tps - 1) // tps


def usd_cost(wall_s: float, usd_per_hour: float) -> float:
    return (float(wall_s) / 3600.0) * float(usd_per_hour)


def params_in_range(n: int) -> bool:
    return N_PARAMS_MIN <= int(n) <= N_PARAMS_MAX


@dataclass
class RunMetrics:
    variant: str
    n_params: int
    seq_len: int
    batch: int
    tokens_seen: int
    steps: int
    final_loss: float
    tokens_per_s: float
    peak_mem_bytes: int
    gpu_name: str
    host: str
    wall_s: float
    usd: float
    h: float
    dropout: float
    weight_decay: float
    optimizer: str
    smoke: bool
    winner: Optional[str] = None
    usd_per_hour: float = 0.0
    rate_source: str = ""
    usd_shadow: float = 0.0
    version: str = ""

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "n_params": self.n_params,
            "seq_len": self.seq_len,
            "batch": self.batch,
            "tokens_seen": self.tokens_seen,
            "steps": self.steps,
            "final_loss": self.final_loss,
            "tokens_per_s": self.tokens_per_s,
            "peak_mem_bytes": self.peak_mem_bytes,
            "gpu_name": self.gpu_name,
            "host": self.host,
            "wall_s": self.wall_s,
            "usd": self.usd,
            "usd_per_hour": self.usd_per_hour,
            "rate_source": self.rate_source,
            "usd_shadow": self.usd_shadow,
            "version": self.version,
            "h": self.h,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "smoke": self.smoke,
            "winner": self.winner,
        }
