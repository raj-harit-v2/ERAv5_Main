"""Gradient accumulation averaging bug demo (Session 10 Task 3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def compute_accumulation_losses() -> dict[str, float]:
    mb1 = torch.tensor([2.0, 2.0, 2.0, 2.0])
    mb2 = torch.tensor([2.0, 2.0, 2.0, 2.0])
    mb3 = torch.tensor([5.0, 5.0])

    total_sum = mb1.sum() + mb2.sum() + mb3.sum()
    total_tokens = len(mb1) + len(mb2) + len(mb3)
    correct = (total_sum / total_tokens).item()

    buggy = torch.stack([mb1.mean(), mb2.mean(), mb3.mean()]).mean().item()
    discrepancy_pct = (buggy - correct) / correct * 100.0
    return {
        "correct": correct,
        "buggy": buggy,
        "discrepancy_pct": discrepancy_pct,
    }


def plot_accumulation_bug(out_path: Path, dpi: int = 150) -> dict:
    vals = compute_accumulation_losses()
    labels = ["Token-Weighted (Correct)", "Avg-of-Averages (Buggy)"]
    heights = [vals["correct"], vals["buggy"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, heights, color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.set_ylabel("Evaluated Global Loss")
    ax.set_title("Gradient Accumulation Bug — 15.4% Gap")
    ax.set_ylim(0, max(heights) * 1.2)
    for bar, v in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.4f}", ha="center", fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    ok = abs(vals["discrepancy_pct"] - 15.384615) < 0.15
    return {"path": str(out_path), **vals, "ok": ok}
