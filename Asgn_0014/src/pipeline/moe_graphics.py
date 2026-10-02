"""PRIMARY graphics: loss trajectories, expert load health, memory+speed."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY_LOSS = "assgn014_loss_trajectories.png"
PRIMARY_LOAD = "assgn014_expert_load_health.png"
PRIMARY_SPEED = "assgn014_memory_speed_dashboard.png"

COLORS = {
    "dense": "#1f4e79",
    "moe": "#548235",
    "convert": "#c45911",
}


def plot_loss_trajectories(
    dense_hist: Sequence[Dict],
    moe_hist: Sequence[Dict],
    convert_tokens: float,
    out_path: Path,
    *,
    n_dense: int = 0,
    n_moe: int = 0,
    dpi: int = 150,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    if dense_hist:
        ax.plot(
            [h["tokens_seen"] for h in dense_hist],
            [h["loss"] for h in dense_hist],
            marker="o",
            linewidth=2.0,
            label=f"dense N={n_dense}",
            color=COLORS["dense"],
        )
    if moe_hist:
        ax.plot(
            [h["tokens_seen"] for h in moe_hist],
            [h["loss"] for h in moe_hist],
            marker="s",
            linewidth=2.0,
            label=f"moe N={n_moe}",
            color=COLORS["moe"],
        )
    if convert_tokens > 0:
        ax.axvline(
            convert_tokens,
            color=COLORS["convert"],
            linestyle="--",
            linewidth=1.5,
            label=f"convert @{int(convert_tokens)}",
        )
    ax.set_xlabel("tokens_seen")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss trajectories — dense then upcycled MoE\n(must continue falling after convert)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_expert_load_health(
    moe_hist: Sequence[Dict],
    load_hist: Optional[np.ndarray],
    out_path: Path,
    *,
    dpi: int = 150,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))

    if moe_hist:
        xs = [h["tokens_seen"] for h in moe_hist]
        ax0.plot(xs, [h.get("max_vio", 0.0) for h in moe_hist], color="#c45911", linewidth=2.0, label="MaxVio")
        ax0.plot(xs, [h.get("dead_experts", 0.0) for h in moe_hist], color="#7030a0", linewidth=2.0, label="dead_experts")
    ax0.set_xlabel("tokens_seen")
    ax0.set_ylabel("routing health")
    ax0.set_title("MaxVio and dead experts (MoE phase)")
    ax0.grid(True, alpha=0.35)
    ax0.legend(fontsize=8)

    if load_hist is not None and len(load_hist) > 0:
        ax1.bar(range(len(load_hist)), load_hist, color="#548235")
        ax1.set_xlabel("expert id")
        ax1.set_ylabel("mean load share")
        ax1.set_title("Final mean expert load (across layers)")
    else:
        ax1.text(0.5, 0.5, "no load histogram", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Final mean expert load")
    ax1.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_memory_speed(
    dense_m: Dict,
    moe_m: Dict,
    out_path: Path,
    *,
    dpi: int = 150,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    labels = ["dense", "moe"]
    mem = [
        float(dense_m.get("peak_mem_bytes", 0)) / (1024**2),
        float(moe_m.get("peak_mem_bytes", 0)) / (1024**2),
    ]
    speed = [
        float(dense_m.get("tokens_per_s", 0)),
        float(moe_m.get("tokens_per_s", 0)),
    ]
    colors = [COLORS["dense"], COLORS["moe"]]
    ax0.bar(labels, mem, color=colors)
    ax0.set_ylabel("peak memory (MiB)")
    ax0.set_title("Peak memory (memory follows TOTAL params)")
    ax0.grid(True, axis="y", alpha=0.35)

    ax1.bar(labels, speed, color=colors)
    ax1.set_ylabel("tokens / sec")
    ax1.set_title("Throughput")
    ax1.grid(True, axis="y", alpha=0.35)

    # Long-context literacy overlay note (not measured KV)
    fig.suptitle(
        "Memory & speed — denser TOTAL after upcycle; active FLOPs controlled by top-k",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def generate_all_graphics(
    reports: Path,
    *,
    dense_hist: List[Dict],
    moe_hist: List[Dict],
    dense_metrics: Dict,
    moe_metrics: Dict,
    convert_tokens: float,
    load_hist: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    reports.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": plot_loss_trajectories(
            dense_hist,
            moe_hist,
            convert_tokens,
            reports / PRIMARY_LOSS,
            n_dense=int(dense_metrics.get("n_params", 0)),
            n_moe=int(moe_metrics.get("n_params", 0)),
        ),
        "load": plot_expert_load_health(moe_hist, load_hist, reports / PRIMARY_LOAD),
        "speed": plot_memory_speed(dense_metrics, moe_metrics, reports / PRIMARY_SPEED),
    }
    return paths
