"""PRIMARY graphics for Assign_0012 (memory ladder, W=32 dashboard, compute/comm)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.zero_accounting import (
    B200_COMPUTE_S,
    CARD_GIB,
    H100_COMPUTE_S,
    HEADROOM_FRAC,
    IB_2P_S,
    IB_3P_S,
    N_30B,
    activation_gib_estimate,
    bytes_per_param,
    gib_from_bytes_per_param,
    memory_ladder,
    normalize_stage,
)


STAGES = ("DP", "ZeRO-1", "ZeRO-2", "ZeRO-3")
COLORS = {
    "DP": "#1f4e79",
    "ZeRO-1": "#2e75b6",
    "ZeRO-2": "#548235",
    "ZeRO-3": "#c45911",
}


def plot_memory_ladder(
    out_path: Path,
    world_sizes: Sequence[int] = (8, 16, 32, 64),
    n_params: float = N_30B,
    dpi: int = 150,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    xs = list(world_sizes)
    for stage in STAGES:
        ys = [gib_from_bytes_per_param(n_params, bytes_per_param(stage, w)) for w in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=stage, color=COLORS[stage])

    ax.axhline(CARD_GIB, color="#c00000", linewidth=1.8, label=f"Card {CARD_GIB:.1f} GiB")
    ax.axhline(
        CARD_GIB * HEADROOM_FRAC,
        color="#c00000",
        linewidth=1.2,
        linestyle="--",
        label=f"94-95% headroom ({CARD_GIB * HEADROOM_FRAC:.1f} GiB)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(w) for w in xs])
    ax.set_xlabel("GPU count (world size)")
    ax.set_ylabel("GiB per GPU (static training state)")
    ax.set_title("Memory ladder — 30B overlay\nDP/ZeRO-1 never fit; ZeRO-2 from 32; ZeRO-3 from 8")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    # Annotations
    ax.annotate("Z3 FITS@8", xy=(8, gib_from_bytes_per_param(n_params, bytes_per_param("ZeRO-3", 8))),
                xytext=(10, 40), textcoords="data", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#c45911"))
    ax.annotate("Z2 FITS@32", xy=(32, gib_from_bytes_per_param(n_params, bytes_per_param("ZeRO-2", 32))),
                xytext=(40, 90), textcoords="data", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#548235"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_w32_dashboard(
    out_path: Path,
    world_size: int = 32,
    n_params: float = N_30B,
    dpi: int = 150,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stage in STAGES:
        bpp = bytes_per_param(stage, world_size)
        gib = gib_from_bytes_per_param(n_params, bpp)
        fit = "NEVER_FIT" if normalize_stage(stage) in ("DP", "ZeRO-1") else (
            "FITS" if gib <= CARD_GIB else "OOM"
        )
        comm = "3P" if stage == "ZeRO-3" else "2P"
        rows.append((stage, bpp, gib, comm, fit))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    stages = [r[0] for r in rows]
    gibs = [r[2] for r in rows]
    bpps = [r[1] for r in rows]
    colors = [COLORS[s] for s in stages]

    ax0 = axes[0]
    bars = ax0.bar(stages, gibs, color=colors, edgecolor="black", linewidth=0.6)
    ax0.axhline(CARD_GIB, color="#c00000", linewidth=1.6)
    ax0.axhline(CARD_GIB * HEADROOM_FRAC, color="#c00000", linestyle="--", linewidth=1.2)
    ax0.set_ylabel("GiB per GPU")
    ax0.set_title(f"W={world_size} static GiB (N=30B overlay)")
    for bar, row in zip(bars, rows):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{row[2]:.1f}\n{row[4]}\n{row[3]}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1 = axes[1]
    ax1.bar(stages, bpps, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Bytes per parameter")
    ax1.set_title(f"W={world_size} bytes/param formulas")
    for i, row in enumerate(rows):
        ax1.text(i, row[1] + 0.15, f"{row[1]:.4f}", ha="center", fontsize=8)

    fig.suptitle("W=32 per-rank dashboard — PRIMARY", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_compute_comm(
    out_path: Path,
    dpi: int = 150,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["H100", "B200"]
    compute = [H100_COMPUTE_S, B200_COMPUTE_S]
    frac_2p = [IB_2P_S / H100_COMPUTE_S, IB_2P_S / B200_COMPUTE_S]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, compute, width, label="Compute / step (s)", color="#1f4e79")
    b2 = ax.bar(x + width / 2, [IB_2P_S, IB_2P_S], width, label="InfiniBand 2P (s)", color="#c45911")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Compute vs communication — PRIMARY\nFaster cards raise comm/compute fraction")
    ax.legend(loc="upper right")
    for i, f in enumerate(frac_2p):
        ax.text(i, max(compute[i], IB_2P_S) + 0.25, f"comm/compute ≈ {100*f:.0f}%", ha="center", fontsize=9)
    ax.annotate(
        f"optional 3P IB = {IB_3P_S:.2f}s",
        xy=(1, IB_2P_S),
        xytext=(1.35, 5.0),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_ylim(0, 9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_activation_vs_seq(
    out_path: Path,
    seq_lens: Sequence[int] = (2048, 4096, 8192, 16384, 32768),
    batch: int = 2,
    hidden: int = 4096,
    layers: int = 32,
    dpi: int = 150,
) -> Path:
    """Recommended long-context overlay: activations on top of Z2@32 static."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z2_static = gib_from_bytes_per_param(N_30B, bytes_per_param("ZeRO-2", 32))
    acts = [activation_gib_estimate(batch, s, hidden, layers) for s in seq_lens]
    totals = [z2_static + a for a in acts]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(seq_lens, acts, marker="o", label="Activation GiB (estimate)", color="#2e75b6")
    ax.plot(seq_lens, totals, marker="s", label="Z2@32 static + activations", color="#548235")
    ax.axhline(CARD_GIB, color="#c00000", label=f"Card {CARD_GIB:.1f} GiB")
    ax.axhline(z2_static, color="#7030a0", linestyle="--", label=f"Z2@32 static {z2_static:.1f} GiB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length S")
    ax.set_ylabel("GiB")
    ax.set_title("Long-context pressure on ZeRO-2@32 headroom")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_graphics_manifest(reports_dir: Path, paths: dict) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "assgn012_graphics_manifest.md"
    lines = [
        "# Assgn012 Graphics Manifest",
        "",
        "| Plot | Path | What it proves | Priority |",
        "| :--- | :--- | :--- | :--- |",
        f"| Memory ladder | `{paths.get('ladder', '')}` | DP/Z1 never fit; Z2@32; Z3@8 | PRIMARY |",
        f"| W=32 dashboard | `{paths.get('dashboard', '')}` | Bytes/param + GiB + 2P/3P + FITS/OOM | PRIMARY |",
        f"| Compute vs comm | `{paths.get('compute_comm', '')}` | H100 ~34% vs B200 ~77% | PRIMARY |",
        f"| Activation vs S | `{paths.get('activation', '')}` | Long-context headroom on Z2@32 | Recommended |",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_all_graphics(reports_dir: Path) -> dict:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ladder = plot_memory_ladder(reports_dir / "assgn012_memory_ladder.png")
    dash = plot_w32_dashboard(reports_dir / "assgn012_w32_dashboard.png")
    cc = plot_compute_comm(reports_dir / "assgn012_compute_comm.png")
    act = plot_activation_vs_seq(reports_dir / "assgn012_activation_vs_seq.png")
    paths = {
        "ladder": str(ladder.name),
        "dashboard": str(dash.name),
        "compute_comm": str(cc.name),
        "activation": str(act.name),
    }
    manifest = write_graphics_manifest(reports_dir, paths)
    return {
        "ladder": ladder,
        "dashboard": dash,
        "compute_comm": cc,
        "activation": act,
        "manifest": manifest,
        "ok": all(p.is_file() and p.stat().st_size > 0 for p in (ladder, dash, cc)),
    }
