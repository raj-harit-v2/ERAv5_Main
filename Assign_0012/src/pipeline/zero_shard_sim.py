"""Explicit ZeRO-1/2/3 shard ownership for the tiny demo model (no DeepSpeed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from src.utils.zero_accounting import bytes_per_param, gib_from_bytes_per_param


@dataclass
class ShardPlan:
    stage: str
    world_size: int
    n_params: int
    param_names: List[str]
    owner_of_flat_index: List[int]
    bytes_per_param: float
    local_opt_params: int
    local_grad_params: int
    local_model_params: int
    estimated_local_bytes: float


def _flat_param_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters()]


def _numel_list(model: nn.Module) -> List[int]:
    return [p.numel() for _, p in model.named_parameters()]


def build_owner_map(n_flat: int, world_size: int) -> List[int]:
    """Assign flat parameter-index ownership round-robin across ranks."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return [i % world_size for i in range(n_flat)]


def plan_for_stage(model: nn.Module, stage: str, world_size: int, rank: int = 0) -> ShardPlan:
    """Build a shard plan describing what this rank would keep under each stage."""
    names = _flat_param_names(model)
    numels = _numel_list(model)
    n_params = int(sum(numels))
    owners = build_owner_map(len(names), world_size)
    bpp = bytes_per_param(stage, world_size)

    # Ownership is by parameter tensor (named), approximating slice ownership.
    owned = [i for i, o in enumerate(owners) if o == rank]
    owned_params = int(sum(numels[i] for i in owned)) if owned else 0

    stage_u = stage.upper().replace(" ", "-")
    if stage_u in ("DP", "DATA-PARALLELISM"):
        local_opt = n_params
        local_grad = n_params
        local_model = n_params
        # 16 B/param fully local
        est = 16.0 * n_params
    elif stage_u in ("ZERO-1", "ZEORO-1", "Z1", "ZERO1"):
        local_opt = owned_params
        local_grad = n_params
        local_model = n_params
        # 2B param + 2B grad + 12B opt/W
        est = (2.0 + 2.0) * n_params + 12.0 * owned_params
    elif stage_u in ("ZERO-2", "Z2", "ZERO2"):
        local_opt = owned_params
        local_grad = owned_params
        local_model = n_params
        est = 2.0 * n_params + (2.0 + 12.0) * owned_params
    elif stage_u in ("ZERO-3", "Z3", "ZERO3"):
        local_opt = owned_params
        local_grad = owned_params
        local_model = owned_params
        est = 16.0 * owned_params
    else:
        raise ValueError(f"unknown stage: {stage!r}")

    return ShardPlan(
        stage=stage,
        world_size=world_size,
        n_params=n_params,
        param_names=names,
        owner_of_flat_index=owners,
        bytes_per_param=bpp,
        local_opt_params=local_opt,
        local_grad_params=local_grad,
        local_model_params=local_model,
        estimated_local_bytes=est,
    )


def simulate_zero_memory_table(
    model: nn.Module,
    world_size: int,
    stages: Sequence[str] = ("DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"),
) -> List[Dict[str, float | str | int]]:
    """Per-stage demo-scale memory estimate for rank 0 ownership."""
    rows = []
    for stage in stages:
        plan = plan_for_stage(model, stage, world_size, rank=0)
        rows.append(
            {
                "stage": plan.stage,
                "world_size": world_size,
                "n_params": plan.n_params,
                "bytes_per_param_formula": round(plan.bytes_per_param, 6),
                "local_model_params": plan.local_model_params,
                "local_grad_params": plan.local_grad_params,
                "local_opt_params": plan.local_opt_params,
                "estimated_local_mib": round(plan.estimated_local_bytes / (1024**2), 4),
                "overlay_gib_30b": round(gib_from_bytes_per_param(30e9, plan.bytes_per_param), 4),
            }
        )
    return rows


def partition_optimizer_state(
    model: nn.Module,
    world_size: int,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """ZeRO-1 style: keep FP32 master/m/v only for owned parameters."""
    owners = build_owner_map(len(list(model.parameters())), world_size)
    state: Dict[str, torch.Tensor] = {}
    for idx, (name, p) in enumerate(model.named_parameters()):
        if owners[idx] != rank:
            continue
        state[f"{name}.master"] = p.detach().float().clone()
        state[f"{name}.m"] = torch.zeros_like(state[f"{name}.master"])
        state[f"{name}.v"] = torch.zeros_like(state[f"{name}.master"])
    return state


@torch.no_grad()
def zero3_gather_param(param: torch.Tensor, owner_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    """Educational stub: reconstruct full param from equal shards (concat dim0)."""
    # For demo we store equal dim-0 shards; if not divisible, pad.
    shards = [torch.zeros_like(owner_shard) for _ in range(world_size)]
    # In real distributed, all_gather fills shards; here caller supplies only local.
    shards[0] = owner_shard
    return torch.cat(shards, dim=0)[: param.shape[0]]
