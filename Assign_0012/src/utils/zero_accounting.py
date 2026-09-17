"""ZeRO / DP memory accounting matching Session 12 formulas.

```text
DP:     16
ZeRO-1: 4 + 12/W
ZeRO-2: 2 + 14/W
ZeRO-3: 16/W
GiB = N * bytes_per_param / 1024**3
card = 80e9 / 1024**3 ≈ 74.5 GiB
P_GB = N * 2 / 1e9
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

N_30B: float = 30e9
CARD_GIB: float = (80e9) / (1024**3)  # ≈ 74.5058
HEADROOM_FRAC: float = 0.95
DEFAULT_WORLD_SIZES: tuple[int, ...] = (8, 16, 32, 64)

# FULL §7 published targets (GiB), tolerance 0.2 GiB in tests
FULL_LADDER_GIB: Dict[int, Dict[str, float]] = {
    8: {"DP": 447.0, "ZeRO-1": 153.7, "ZeRO-2": 104.8, "ZeRO-3": 55.9},
    16: {"DP": 447.0, "ZeRO-1": 132.7, "ZeRO-2": 80.3, "ZeRO-3": 27.9},
    32: {"DP": 447.0, "ZeRO-1": 122.2, "ZeRO-2": 68.1, "ZeRO-3": 14.0},
    64: {"DP": 447.0, "ZeRO-1": 117.0, "ZeRO-2": 62.0, "ZeRO-3": 7.0},
}

FULL_BYTES_W8: Dict[str, float] = {
    "DP": 16.00,
    "ZeRO-1": 5.50,
    "ZeRO-2": 3.75,
    "ZeRO-3": 2.00,
}

# FULL §5 compute / InfiniBand 2P reference
H100_COMPUTE_S: float = 7.10
B200_COMPUTE_S: float = 3.12
IB_2P_S: float = 2.40
IB_3P_S: float = 3.60
NVLINK_2P_S: float = 0.27
NVLINK_3P_S: float = 0.40


@dataclass(frozen=True)
class StageRow:
    stage: str
    world_size: int
    bytes_per_param: float
    gib_per_gpu: float
    cluster_gib: float
    comm: str
    p_gb: float
    fit: str  # FITS | OOM | NEVER_FIT
    headroom_ok: bool


def bytes_per_param(stage: str, world_size: int) -> float:
    """Return mixed-precision Adam training bytes per parameter for a stage."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    key = stage.strip().upper().replace(" ", "-")
    if key in ("DP", "DATA-PARALLELISM", "DATA_PARALLELISM"):
        return 16.0
    if key in ("ZERO-1", "ZERO1", "ZEORO-1", "Z1"):
        return 4.0 + 12.0 / world_size
    if key in ("ZERO-2", "ZERO2", "Z2"):
        return 2.0 + 14.0 / world_size
    if key in ("ZERO-3", "ZERO3", "Z3"):
        return 16.0 / world_size
    raise ValueError(f"unknown stage: {stage!r}")


def normalize_stage(stage: str) -> str:
    key = stage.strip().upper().replace(" ", "-")
    aliases = {
        "DP": "DP",
        "DATA-PARALLELISM": "DP",
        "DATA_PARALLELISM": "DP",
        "ZERO-1": "ZeRO-1",
        "ZERO1": "ZeRO-1",
        "Z1": "ZeRO-1",
        "ZERO-2": "ZeRO-2",
        "ZERO2": "ZeRO-2",
        "Z2": "ZeRO-2",
        "ZERO-3": "ZeRO-3",
        "ZERO3": "ZeRO-3",
        "Z3": "ZeRO-3",
    }
    if key not in aliases:
        raise ValueError(f"unknown stage: {stage!r}")
    return aliases[key]


def gib_from_bytes_per_param(n_params: float, bpp: float) -> float:
    return (n_params * bpp) / (1024**3)


def p_gb(n_params: float) -> float:
    """P in marketing GB (N * 2 / 1e9), FULL §2 style."""
    return (n_params * 2.0) / 1e9


def comm_label(stage: str) -> str:
    s = normalize_stage(stage)
    return "3P" if s == "ZeRO-3" else "2P"


def fit_flag(stage: str, gib: float, world_size: int, card_gib: float = CARD_GIB) -> str:
    """DP and ZeRO-1 never fit at 30B (replicated 4 B/weight floor)."""
    s = normalize_stage(stage)
    if s in ("DP", "ZeRO-1"):
        return "NEVER_FIT"
    if gib <= card_gib:
        return "FITS"
    return "OOM"


def headroom_ok(gib: float, card_gib: float = CARD_GIB, frac: float = HEADROOM_FRAC) -> bool:
    return gib <= (frac * card_gib)


def stage_row(
    stage: str,
    world_size: int,
    n_params: float = N_30B,
    card_gib: float = CARD_GIB,
) -> StageRow:
    s = normalize_stage(stage)
    bpp = bytes_per_param(s, world_size)
    gib = gib_from_bytes_per_param(n_params, bpp)
    fit = fit_flag(s, gib, world_size, card_gib=card_gib)
    return StageRow(
        stage=s,
        world_size=world_size,
        bytes_per_param=bpp,
        gib_per_gpu=gib,
        cluster_gib=gib * world_size,
        comm=comm_label(s),
        p_gb=p_gb(n_params),
        fit=fit,
        headroom_ok=headroom_ok(gib, card_gib=card_gib),
    )


def memory_ladder(
    world_sizes: Sequence[int] = DEFAULT_WORLD_SIZES,
    n_params: float = N_30B,
    stages: Sequence[str] = ("DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"),
) -> List[StageRow]:
    rows: List[StageRow] = []
    for w in world_sizes:
        for stage in stages:
            rows.append(stage_row(stage, w, n_params=n_params))
    return rows


def ladder_as_dicts(rows: Iterable[StageRow]) -> List[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "stage": r.stage,
                "world_size": r.world_size,
                "bytes_per_param": round(r.bytes_per_param, 4),
                "gib_per_gpu": round(r.gib_per_gpu, 4),
                "cluster_gib": round(r.cluster_gib, 4),
                "comm": r.comm,
                "p_gb": round(r.p_gb, 4),
                "fit": r.fit,
                "headroom_ok": r.headroom_ok,
            }
        )
    return out


def replicated_floor_gib(n_params: float = N_30B) -> float:
    """4 B/weight (FP16 param + FP16 grad) always replicated under DP and ZeRO-1."""
    return gib_from_bytes_per_param(n_params, 4.0)


def activation_gib_estimate(
    batch: int,
    seq_len: int,
    hidden: int,
    layers: int,
    bytes_per_act: float = 2.0,
) -> float:
    """Rough activation footprint for long-context headroom discussion (FULL §7)."""
    return (batch * seq_len * hidden * layers * bytes_per_act) / (1024**3)


def comm_fraction(compute_s: float, comm_s: float) -> float:
    if compute_s <= 0:
        raise ValueError("compute_s must be > 0")
    return comm_s / compute_s
