"""Distributed helpers for Assign_0012 (gloo-first, optional NCCL)."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    rank: int
    world_size: int
    backend: str
    hostname: str
    is_distributed: bool


def prefer_backend() -> str:
    """gloo on CPU/Windows; NCCL only when CUDA multi-GPU is available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "nccl"
    return "gloo"


def env_world_size() -> Optional[int]:
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        if key in os.environ:
            return int(os.environ[key])
    return None


def env_rank() -> Optional[int]:
    for key in ("RANK", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return None


def init_distributed(backend: Optional[str] = None, timeout_s: int = 600) -> DistContext:
    """Initialize process group if launched under torchrun/spawn; else single-process context.

    If the process group is already initialized (e.g. a Windows file-store
    spawn launcher), reuse it.
    """
    hostname = socket.gethostname()

    if dist.is_available() and dist.is_initialized():
        be = backend or prefer_backend()
        return DistContext(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            backend=be,
            hostname=hostname,
            is_distributed=True,
        )

    ws = env_world_size()
    rk = env_rank()
    if ws is None or rk is None or ws < 2:
        return DistContext(
            rank=0,
            world_size=1,
            backend="none",
            hostname=hostname,
            is_distributed=False,
        )

    be = backend or prefer_backend()
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(backend=be, timeout=torch.distributed.default_pg_timeout)
    return DistContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        backend=be,
        hostname=hostname,
        is_distributed=True,
    )


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def smoke_all_reduce(device: Optional[torch.device] = None) -> tuple[float, bool]:
    """Each rank contributes 1.0; sum must equal world_size."""
    if device is None:
        device = torch.device("cpu")
    t = torch.ones(1, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ok = abs(float(t.item()) - float(dist.get_world_size())) < 1e-5
        return float(t.item()), ok
    return 1.0, True


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


def max_abs_param_diff_vs_rank0(model: torch.nn.Module) -> float:
    """Max |w_i - w_0| across all parameters (FULL §3 replica identity check)."""
    if not (dist.is_available() and dist.is_initialized()):
        return 0.0
    diffs = []
    for p in model.parameters():
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        diffs.append((p.detach() - ref).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def peak_rss_mb() -> float:
    """Best-effort peak RSS in MiB (Windows/Linux)."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        # Linux: ru_maxrss in KB; macOS: bytes
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix" and sys_platform_is_linux():
            return float(ru) / 1024.0
        return float(ru) / (1024 * 1024)
    except Exception:
        return -1.0


def sys_platform_is_linux() -> bool:
    return os.name == "posix" and "linux" in __import__("sys").platform
