"""Task 5: LR sweep at widths 256/512/1024; extrapolate eta at 4096 (SP only)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from src.llm.mini_gpt import MiniGPT, MiniGPTConfig


def _log_grid(lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [lo]
    ratios = []
    for i in range(n):
        t = i / (n - 1)
        ratios.append(math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo))))
    return ratios


def _train_short(
    width: int,
    lr: float,
    *,
    n_steps: int,
    seed: int,
    batch_size: int = 2,
    block_size: int = 32,
) -> float:
    torch.manual_seed(seed)
    cfg = MiniGPTConfig(
        vocab_size=128,
        n_embd=width,
        n_head=4,
        n_layer=2,
        block_size=block_size,
    )
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    losses: list[float] = []
    for _ in range(n_steps):
        batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size))
        opt.zero_grad()
        loss = model.loss_on_batch(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    tail = max(1, n_steps // 5)
    return sum(losses[-tail:]) / float(tail)


def run_lr_width_sweep(
    reports: Path,
    *,
    widths: tuple[int, ...] = (256, 512, 1024),
    n_etas: int = 8,
    n_steps: int = 40,
    seed: int = 7,
) -> dict[str, Any]:
    """Sweep LR under standard parameterization; SP extrapolate eta at 4096."""
    reports.mkdir(parents=True, exist_ok=True)
    etas = _log_grid(1e-4, 1e-2, n_etas)

    curves: dict[int, list[float]] = {}
    minima: dict[int, dict[str, float]] = {}
    for w in widths:
        losses = []
        for i, eta in enumerate(etas):
            loss = _train_short(w, eta, n_steps=n_steps, seed=seed + w + i)
            losses.append(loss)
        curves[w] = losses
        best_i = min(range(len(etas)), key=lambda i: losses[i])
        minima[w] = {"eta": etas[best_i], "loss": losses[best_i]}

    # SP: eta ∝ 1/width → eta_4096 ≈ eta_1024 / 4
    eta_1024 = minima[1024]["eta"]
    eta_4096_sp = eta_1024 / 4.0

    png_path = reports / "assgn011_lr_width_sweep.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {256: "#0891b2", 512: "#7c3aed", 1024: "#d97706"}
    for w in widths:
        ax.plot(etas, curves[w], marker="o", label=f"width={w}", color=colors[w], linewidth=1.6)
        m = minima[w]
        ax.scatter([m["eta"]], [m["loss"]], s=80, color=colors[w], zorder=5, edgecolors="black")
        ax.annotate(
            f"min η={m['eta']:.4g}",
            xy=(m["eta"], m["loss"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=colors[w],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate η")
    ax.set_ylabel(f"Mean late loss ({n_steps} steps, last 20%)")
    ax.set_title("LR sweep under standard parameterization — mark three minima")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md_path = reports / "assgn011_lr_sweep_table.md"
    lines = [
        "# Assgn011 — LR Width Sweep (Task 5)",
        "",
        f"Matched budget: {n_steps} steps per (width, eta) point, batch=2, block_size=32, AdamW beta2=0.95.",
        "Parameterization: **standard** (SP).",
        "",
        "| Width | Best η | Late loss |",
        "| ---: | ---: | ---: |",
    ]
    for w in widths:
        lines.append(f"| {w} | {minima[w]['eta']:.6g} | {minima[w]['loss']:.6f} |")
    lines.extend(
        [
            "",
            "## Extrapolation to width 4096",
            "",
            "```text",
            "SP:  eta_4096 ≈ eta_1024 / 4     (eta ∝ 1/width)",
            "```",
            "",
            "| Value | η at 4096 | Confidence | Why |",
            "| :--- | ---: | :--- | :--- |",
            f"| Standard parameterization (SP) | {eta_4096_sp:.6g} | **LOW** | "
            "FULL.txt Section 12: transferring a small-width η under SP can overstate by up to 16× "
            "(256→4096); without a μP day we do not claim high confidence |",
            "",
            f"Value we would use at width 4096 under SP: **{eta_4096_sp:.6g}**, confidence **LOW**.",
            "",
            f"Primary plot: `{png_path.name}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "ok": png_path.is_file() and png_path.stat().st_size > 0 and len(minima) == 3,
        "etas": etas,
        "minima": minima,
        "eta_4096_sp": eta_4096_sp,
        "png_path": str(png_path),
        "report_path": str(md_path),
        "n_steps": n_steps,
    }
