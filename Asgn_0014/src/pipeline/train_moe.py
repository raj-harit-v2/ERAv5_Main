"""Train loops for dense Phase 1 and MoE Phase 2 (Assign_0014)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.llm.moe import TinyMoELM, upcycle_dense_to_moe
from src.llm.tiny_lm import LMConfig, TinyDenseLM
from src.utils.corpus import build_batch_source
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


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def dump_run_json(path: Path, metrics: RunMetrics, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.as_dict()
    payload["history"] = history
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_dense(
    batch: int,
    seq_len: int,
    token_budget: int,
    *,
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
    cfg: Optional[LMConfig] = None,
    batch_source: Optional[Any] = None,
    data_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[TinyDenseLM, RunMetrics, List[Dict[str, float]]]:
    device = device or pick_device()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = TinyDenseLM(cfg or LMConfig(dropout=dropout)).to(device)
    n_params = model.count_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tps = tokens_per_step(batch, seq_len)
    if batch_source is None:
        root = Path(__file__).resolve().parents[2]
        batch_source, data_meta = build_batch_source(root, seed=seed)
    data_meta = data_meta or {}

    reset_peak_memory(device)
    history: List[Dict[str, float]] = []
    tokens_seen = 0
    step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()
    model.train()

    while tokens_seen < token_budget:
        tok = batch_source.next_batch(batch, seq_len, device)
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
            "max_vio": 0.0,
            "dead_experts": 0.0,
            "phase": 0.0,
        }
        if step == 1 or step % log_every == 0 or tokens_seen >= token_budget:
            history.append(rec)
        if ckpt_dir is not None and tokens_seen % ckpt_every_tokens < tps:
            save_checkpoint(
                ckpt_dir / f"dense_step{step}.pt",
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "tokens_seen": tokens_seen,
                    "step": step,
                    "variant": "dense",
                    "batch": batch,
                    "seq_len": seq_len,
                },
            )

    wall_s = time.perf_counter() - t0
    metrics = RunMetrics(
        variant="dense",
        n_params=n_params,
        n_active=n_params,
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
        dropout=dropout,
        weight_decay=weight_decay,
        optimizer=OPTIMIZER_NAME,
        smoke=smoke,
        use_ce_aux=False,
        router_dtype="n/a",
    )
    _ = data_meta  # stamped by moe_lab from shared build_batch_source
    return model, metrics, history


def train_moe(
    model: TinyMoELM,
    batch: int,
    seq_len: int,
    token_budget: int,
    *,
    tokens_offset: int = 0,
    loss_at_convert: float = float("nan"),
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    usd_per_hour: float = 0.0,
    host: str = "local",
    device: Optional[torch.device] = None,
    log_every: int = 5,
    ckpt_every_tokens: int = 5_000_000,
    ckpt_dir: Optional[Path] = None,
    seed: int = 14,
    smoke: bool = True,
    rate_source: str = "",
    shadow_usd_per_hour: float = 0.12,
    version: str = "",
    batch_source: Optional[Any] = None,
    data_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[TinyMoELM, RunMetrics, List[Dict[str, float]]]:
    device = device or pick_device()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    # Ensure routers stay FP32
    for blk in model.blocks:
        blk.moe.router.to(dtype=torch.float32)

    n_params = model.count_parameters()
    n_active = model.count_active_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tps = tokens_per_step(batch, seq_len)
    if batch_source is None:
        root = Path(__file__).resolve().parents[2]
        batch_source, data_meta = build_batch_source(root, seed=seed)
    data_meta = data_meta or {}

    reset_peak_memory(device)
    history: List[Dict[str, float]] = []
    tokens_phase = 0
    step = 0
    last_loss = float("nan")
    last_stats = {"max_vio": 0.0, "dead_experts": 0.0}
    # Floor from config (env); auto-extend so short ladder and long full both stay
    # in probabilistic selection for the whole MoE phase (v015 failed after step 2000).
    configured = int(getattr(model.moe_cfg, "warmup_steps", 2000) or 0)
    phase_steps = max(1, int(math.ceil(float(token_budget) / float(tps))))
    warmup_steps = max(configured, phase_steps)
    t0 = time.perf_counter()
    model.train()

    while tokens_phase < token_budget:
        # Probabilistic selection for the MoE phase (covers full budget after auto-scale)
        model.set_probabilistic(step < warmup_steps)
        tok = batch_source.next_batch(batch, seq_len, device)
        x = tok[:, :-1]
        y = tok[:, 1:]
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, model.dense_cfg.vocab_size),
            y.reshape(-1),
        )
        # Graded path: CE only (no aux). use_ce_aux locked False.
        if model.moe_cfg.use_ce_aux:
            raise RuntimeError("CE aux load loss is forbidden on graded Assgn_014 path")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_all_biases()
        last_stats = model.routing_stats()
        step += 1
        tokens_phase += tps
        tokens_seen = tokens_offset + tokens_phase
        last_loss = float(loss.item())
        elapsed = time.perf_counter() - t0
        tok_s = tokens_phase / elapsed if elapsed > 0 else 0.0
        rec = {
            "step": float(step),
            "tokens_seen": float(tokens_seen),
            "loss": last_loss,
            "tokens_per_s": float(tok_s),
            "max_vio": float(last_stats["max_vio"]),
            "dead_experts": float(last_stats["dead_experts"]),
            "phase": 1.0,
            "probabilistic": 1.0 if step <= warmup_steps else 0.0,
            "warmup_steps_effective": float(warmup_steps),
        }
        if step == 1 or step % log_every == 0 or tokens_phase >= token_budget:
            history.append(rec)
        if ckpt_dir is not None and tokens_phase % ckpt_every_tokens < tps:
            save_checkpoint(
                ckpt_dir / f"moe_step{step}.pt",
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "tokens_seen": tokens_seen,
                    "step": step,
                    "variant": "moe",
                    "batch": batch,
                    "seq_len": seq_len,
                },
            )

    model.set_probabilistic(False)
    wall_s = time.perf_counter() - t0
    router_dtype = str(next(model.blocks[0].moe.router.parameters()).dtype).replace("torch.", "")
    metrics = RunMetrics(
        variant="moe",
        n_params=n_params,
        n_active=n_active,
        seq_len=seq_len,
        batch=batch,
        tokens_seen=tokens_phase,
        steps=step,
        final_loss=last_loss,
        tokens_per_s=tokens_phase / wall_s if wall_s > 0 else 0.0,
        peak_mem_bytes=peak_memory_bytes(device),
        gpu_name=gpu_name(device),
        host=host,
        wall_s=wall_s,
        usd=usd_cost(wall_s, usd_per_hour),
        usd_per_hour=float(usd_per_hour),
        rate_source=rate_source,
        usd_shadow=usd_cost(wall_s, shadow_usd_per_hour),
        version=version,
        dropout=model.moe_cfg.dropout,
        weight_decay=weight_decay,
        optimizer=OPTIMIZER_NAME,
        smoke=smoke,
        max_vio=float(last_stats["max_vio"]),
        dead_experts=float(last_stats["dead_experts"]),
        loss_at_convert=float(loss_at_convert),
        use_ce_aux=False,
        router_dtype=router_dtype,
        n_experts=model.moe_cfg.n_experts,
        top_k=model.moe_cfg.top_k,
        warmup_steps_effective=warmup_steps,
    )
    return model, metrics, history


def convert_dense_model(
    dense: TinyDenseLM,
    *,
    n_experts: int = 8,
    top_k: int = 2,
    bias_gamma: float = 0.001,
    drop_redraw_frac: float = 0.5,
    seed: int = 14,
    warmup_steps: int = 2000,
    use_three_tier: bool = True,
) -> TinyMoELM:
    return upcycle_dense_to_moe(
        dense,
        n_experts=n_experts,
        top_k=top_k,
        bias_gamma=bias_gamma,
        drop_redraw_frac=drop_redraw_frac,
        seed=seed,
        warmup_steps=warmup_steps,
        use_three_tier=use_three_tier,
    )
