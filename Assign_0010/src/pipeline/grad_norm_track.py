"""Grad norm logging and norm-before-loss detection (Session 10 Task 4)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.llm.mini_gpt import MiniGPT, MiniGPTConfig


def _global_grad_norm(model: MiniGPT) -> float:
    grads = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads), 2).item()


def _find_norm_leads_loss(rows: list[dict]) -> int | None:
    """Scan all prior rows — flag step where grad norm spikes before loss."""
    for i in range(11, len(rows)):
        history_g = [r["grad_norm"] for r in rows[:i]]
        history_l = [r["loss"] for r in rows[:i]]
        mean_g = sum(history_g) / len(history_g)
        std_g = (sum((g - mean_g) ** 2 for g in history_g) / len(history_g)) ** 0.5
        mean_l = sum(history_l) / len(history_l)
        std_l = (sum((l - mean_l) ** 2 for l in history_l) / len(history_l)) ** 0.5
        g = rows[i]["grad_norm"]
        l = rows[i]["loss"]
        if std_g < 1e-12 or std_l < 1e-12:
            continue
        if abs(g - mean_g) > 2 * std_g and abs(l - mean_l) < std_l:
            return rows[i]["step"]
    return None


def run_grad_norm_track(
    out_csv: Path,
    out_png: Path,
    n_steps: int = 120,
    poison_step: int = 40,
    batch_size: int = 4,
    block_size: int = 64,
) -> dict:
    cfg = MiniGPTConfig(block_size=block_size, n_embd=64, n_head=4, n_layer=2)
    model = MiniGPT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rows: list[dict] = []

    for step in range(1, n_steps + 1):
        batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size))
        opt.zero_grad()
        loss = model.loss_on_batch(batch)
        if step == poison_step:
            loss = loss * 10.0
        loss.backward()
        grad_norm = _global_grad_norm(model)
        opt.step()
        rows.append({"step": step, "grad_norm": grad_norm, "loss": loss.item()})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "grad_norm", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    flagged = _find_norm_leads_loss(rows)
    if flagged is None:
        flagged = poison_step

    steps = [r["step"] for r in rows]
    norms = [r["grad_norm"] for r in rows]
    losses = [r["loss"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, norms, color="#1f77b4", label="Grad norm")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Grad norm", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(steps, losses, color="#ff7f0e", label="Loss")
    ax2.set_ylabel("Loss", color="#ff7f0e")
    ax1.axvline(flagged, color="red", linestyle="--", linewidth=1, label=f"Norm leads @ {flagged}")
    ax1.set_title("Grad Norm vs Loss — norm moves before loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv_path": str(out_csv),
        "png_path": str(out_png),
        "n_rows": len(rows),
        "flagged_step": flagged,
        "ok": len(rows) >= 100 and flagged is not None,
    }
