"""PRIMARY graphics: loss trajectories, memory vs seq/batch, speed+cost."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY_LOSS = "assgn013_loss_trajectories.png"
PRIMARY_MEM = "assgn013_memory_vs_seq_batch.png"
PRIMARY_SPEED = "assgn013_speed_cost_dashboard.png"

COLORS = {
    "standard": "#1f4e79",
    "euler": "#c45911",
    "midpoint": "#548235",
    "midpoint_max": "#7030a0",
    "euler_max": "#7030a0",
}

# FULL.txt §17 lecture overlay (30.2B, not measured on the 20M run)
OVERLAY_S = (8192, 32768, 131072)
OVERLAY_STORED = (127.5, 510.0, 2040.0)
OVERLAY_REV = (1.5, 5.9, 23.8)


def _label(run: Dict) -> str:
    variant = run["variant"]
    batch = run["batch"]
    if run.get("is_max_batch"):
        return f"{variant} B_max={batch}"
    return f"{variant} B={batch}"


def plot_loss_trajectories(runs: Sequence[Dict], out_path: Path, dpi: int = 150) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for run in runs:
        hist = run.get("history") or []
        if not hist:
            continue
        xs = [h["tokens_seen"] for h in hist]
        ys = [h["loss"] for h in hist]
        key = run["variant"]
        if run.get("is_max_batch"):
            key = f"{run['variant']}_max"
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            label=_label(run),
            color=COLORS.get(key, "#333333"),
        )
    n = runs[0]["n_params"] if runs else 0
    ax.set_xlabel("tokens_seen")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(
        f"Loss trajectories — 20M decoder (N={n})\n"
        "same B_fixed for standard/euler/midpoint; winner at B_max"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_memory(runs: Sequence[Dict], out_path: Path, dpi: int = 150) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))

    labels = [_label(r) for r in runs]
    mem_mib = [r["peak_mem_bytes"] / (1024 ** 2) for r in runs]
    colors = []
    for r in runs:
        key = r["variant"] if not r.get("is_max_batch") else f"{r['variant']}_max"
        colors.append(COLORS.get(key, "#333333"))
    ax0.bar(range(len(labels)), mem_mib, color=colors)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax0.set_ylabel("peak memory (MiB)")
    ax0.set_title("Measured peak (this machine)")
    ax0.grid(True, axis="y", alpha=0.35)

    ax1.plot(OVERLAY_S, OVERLAY_STORED, marker="o", linewidth=2.2, color="#1f4e79", label="stored act (30.2B overlay)")
    ax1.plot(OVERLAY_S, OVERLAY_REV, marker="s", linewidth=2.2, color="#548235", label="reversible act (30.2B overlay)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(list(OVERLAY_S))
    ax1.set_xticklabels([str(s) for s in OVERLAY_S])
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("GiB activations")
    ax1.set_title("FULL §17 overlay (not the 20M run)")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.legend(fontsize=8)
    fig.suptitle("Memory: measured 20M peaks vs lecture long-context overlay", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_speed_cost(runs: Sequence[Dict], out_path: Path, dpi: int = 150) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    labels = [_label(r) for r in runs]
    xs = range(len(labels))
    colors = []
    for r in runs:
        key = r["variant"] if not r.get("is_max_batch") else f"{r['variant']}_max"
        colors.append(COLORS.get(key, "#333333"))

    ax0.bar(xs, [r["tokens_per_s"] for r in runs], color=colors)
    ax0.set_xticks(list(xs))
    ax0.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax0.set_ylabel("tokens / s")
    ax0.set_title("Throughput")
    ax0.grid(True, axis="y", alpha=0.35)
    ax0.annotate(
        "paper: 114.49 vs 56.96 samples/s from batch 874 vs 52",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        va="top",
    )

    cost_vals = [float(r.get("usd") or 0.0) for r in runs]
    use_shadow = all(v == 0.0 for v in cost_vals) and any(
        float(r.get("usd_shadow") or 0.0) > 0.0 for r in runs
    )
    if use_shadow:
        cost_vals = [float(r.get("usd_shadow") or 0.0) for r in runs]
        cost_ylabel = "USD shadow (T4 0.12/h)"
        cost_title = "Cost (shadow; advertised was 0.00)"
    else:
        rate = runs[0].get("usd_per_hour", "") if runs else ""
        cost_ylabel = f"USD (hours x rate={rate})"
        cost_title = "Cost"

    ax1.bar(xs, cost_vals, color=colors)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel(cost_ylabel)
    ax1.set_title(cost_title)
    ax1.grid(True, axis="y", alpha=0.35)
    fig.suptitle("Speed and cost dashboard — 30-50% tax at B_fixed; batch gift at B_max", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def generate_all_graphics(runs: Sequence[Dict], reports: Path) -> Dict[str, Path]:
    reports.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": plot_loss_trajectories(runs, reports / PRIMARY_LOSS),
        "memory": plot_memory(runs, reports / PRIMARY_MEM),
        "speed": plot_speed_cost(runs, reports / PRIMARY_SPEED),
    }
    manifest = reports / "assgn013_graphics_manifest.md"
    lines = [
        "# Assgn013 PRIMARY graphics",
        "",
        "| File | Proves |",
        "| :--- | :--- |",
        f"| `{PRIMARY_LOSS}` | Euler vs midpoint quality at the same B_fixed; B_max overlay |",
        f"| `{PRIMARY_MEM}` | Measured peak vs batch; FULL §17 long-context overlay 127.5 vs 1.5 GiB |",
        f"| `{PRIMARY_SPEED}` | tokens/s and USD; compute tax vs batch gift |",
        "",
        "Lecture overlay numbers are labeled overlay and are not 20M measurements.",
        "",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")
    paths["manifest"] = manifest
    return paths
