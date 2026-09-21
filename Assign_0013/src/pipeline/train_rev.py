"""Train loop for the 20M reversible / standard lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.llm.tiny_lm import LMConfig, TinyReversibleLM
from src.utils.metrics import (
    RunMetrics,
    gpu_name,
    peak_memory_bytes,
    pick_device,
    reset_peak_memory,
    tokens_per_step,
    usd_cost,
)

OPTIMIZER_NAME = "AdamW"


def _is_oom(err: BaseException) -> bool:
    text = str(err).lower()
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in text


def build_model(mode: str, h: float, dropout: float, device: torch.device) -> TinyReversibleLM:
    cfg = LMConfig(mode=mode, h=h, dropout=dropout)  # type: ignore[arg-type]
    model = TinyReversibleLM(cfg)
    return model.to(device)


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one(
    mode: str,
    batch: int,
    seq_len: int,
    token_budget: int,
    *,
    h: float = 0.25,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    usd_per_hour: float = 0.0,
    host: str = "local",
    device: Optional[torch.device] = None,
    log_every: int = 5,
    ckpt_every_tokens: int = 5_000_000,
    ckpt_dir: Optional[Path] = None,
    seed: int = 13,
    smoke: bool = True,
    rate_source: str = "",
    shadow_usd_per_hour: float = 0.12,
    version: str = "",
) -> tuple[RunMetrics, List[Dict[str, float]]]:
    device = device or pick_device()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    drop = 0.0 if mode != "standard" else dropout
    wd = 0.0 if mode != "standard" else weight_decay
    model = build_model(mode, h=h, dropout=drop, device=device)
    n_params = model.count_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tps = tokens_per_step(batch, seq_len)

    reset_peak_memory(device)
    history: List[Dict[str, float]] = []
    tokens_seen = 0
    step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()
    model.train()

    while tokens_seen < token_budget:
        tok = torch.randint(0, model.cfg.vocab_size, (batch, seq_len + 1), device=device)
        x = tok[:, :-1]
        y = tok[:, 1:]
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size),
            y.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += tps
        last_loss = float(loss.item())
        elapsed = time.perf_counter() - t0
        tok_s = tokens_seen / elapsed if elapsed > 0 else 0.0
        rec = {
            "step": float(step),
            "tokens_seen": float(tokens_seen),
            "loss": last_loss,
            "tokens_per_s": float(tok_s),
        }
        if step == 1 or step % log_every == 0 or tokens_seen >= token_budget:
            history.append(rec)
        if ckpt_dir is not None and tokens_seen % ckpt_every_tokens < tps:
            save_checkpoint(
                ckpt_dir / f"{mode}_step{step}.pt",
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "tokens_seen": tokens_seen,
                    "step": step,
                    "mode": mode,
                    "batch": batch,
                    "seq_len": seq_len,
                },
            )

    wall_s = time.perf_counter() - t0
    metrics = RunMetrics(
        variant=mode,
        n_params=n_params,
        seq_len=seq_len,
        batch=batch,
        tokens_seen=tokens_seen,
        steps=step,
        final_loss=last_loss,
        tokens_per_s=tokens_seen / wall_s if wall_s > 0 else 0.0,
        peak_mem_bytes=peak_memory_bytes(device),
        gpu_name=gpu_name(device),
        host=host,
        wall_s=wall_s,
        usd=usd_cost(wall_s, usd_per_hour),
        usd_per_hour=float(usd_per_hour),
        rate_source=rate_source,
        usd_shadow=usd_cost(wall_s, shadow_usd_per_hour),
        version=version,
        h=h,
        dropout=drop,
        weight_decay=wd,
        optimizer=OPTIMIZER_NAME,
        smoke=smoke,
    )
    return metrics, history


def probe_max_batch(
    mode: str,
    seq_len: int,
    *,
    h: float = 0.25,
    device: Optional[torch.device] = None,
    start: int = 1,
    max_try: int = 128,
    cpu_fixed: int = 2,
    cpu_scale: int = 2,
) -> tuple[int, str]:
    """Largest batch that completes one forward+backward. CUDA uses OOM; CPU uses a 2x protocol."""
    device = device or pick_device()
    if device.type != "cuda":
        b_fixed = max(1, int(cpu_fixed))
        if mode == "standard":
            return b_fixed, "cpu_or_mps_no_oom_probe; B_fixed protocol default"
        return b_fixed * cpu_scale, "cpu_or_mps_no_oom_probe; B_max = 2 * B_fixed protocol"

    best = start
    b = start
    while b <= max_try:
        try:
            train_one(
                mode,
                batch=b,
                seq_len=seq_len,
                token_budget=b * seq_len,
                h=h,
                device=device,
                log_every=1,
                smoke=True,
            )
            best = b
            b *= 2
        except Exception as err:
            if _is_oom(err):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            raise
    # Fine search between best and 2*best
    lo, hi = best, min(max_try, best * 2)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            train_one(
                mode,
                batch=mid,
                seq_len=seq_len,
                token_budget=mid * seq_len,
                h=h,
                device=device,
                log_every=1,
                smoke=True,
            )
            lo = mid
        except Exception as err:
            if _is_oom(err):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                hi = mid
            else:
                raise
    return lo, "cuda_oom_binary_search"


def dump_run_json(path: Path, metrics: RunMetrics, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.as_dict()
    payload["history"] = history
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
