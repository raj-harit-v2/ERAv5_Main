"""Task 3: update-to-weight ratio logging and warmup-end detection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from src.llm.mini_gpt import MiniGPT, MiniGPTConfig
from src.utils.ratio_track import detect_warmup_end, group_ratio, layer_update_ratios, snapshot_params
from src.utils.schedules import linear_warmup_lr


def run_warmup_ratio_lab(
    reports: Path,
    *,
    n_steps: int = 300,
    warmup_steps: int = 50,
    peak_lr: float = 3e-3,
    batch_size: int = 4,
    block_size: int = 64,
    n_embd: int = 64,
    seed: int = 11,
) -> dict[str, Any]:
    reports.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    cfg = MiniGPTConfig(block_size=block_size, n_embd=n_embd, n_head=4, n_layer=2, vocab_size=256)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    rows: list[dict[str, Any]] = []
    steps: list[int] = []
    median_ratios: list[float] = []
    embed_series: list[float] = []
    block_series: list[float] = []
    head_series: list[float] = []

    for step in range(1, n_steps + 1):
        lr = linear_warmup_lr(step, peak_lr, warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        before = snapshot_params(model)
        batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size))
        opt.zero_grad()
        loss = model.loss_on_batch(batch)
        loss.backward()
        opt.step()

        ratios = layer_update_ratios(model, before)
        med = sorted(ratios.values())[len(ratios) // 2]
        emb = group_ratio(ratios, ("token_emb", "pos_emb")) or 0.0
        blk = group_ratio(ratios, ("blocks.",)) or 0.0
        head = group_ratio(ratios, ("lm_head",)) or 0.0

        steps.append(step)
        median_ratios.append(med)
        embed_series.append(emb)
        block_series.append(blk)
        head_series.append(head)

        for name, ratio in ratios.items():
            rows.append(
                {
                    "step": step,
                    "layer": name,
                    "ratio": f"{ratio:.8e}",
                    "lr": f"{lr:.8e}",
                    "loss": f"{loss.item():.6f}",
                }
            )

    csv_path = reports / "assgn011_update_weight_ratio.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "layer", "ratio", "lr", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    t_star = detect_warmup_end(steps, median_ratios, peak_step=warmup_steps, window=5, rel_tol=0.05)

    png_path = reports / "assgn011_update_weight_ratio.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, embed_series, label="embed (median)", color="#0891b2", linewidth=1.5)
    ax.plot(steps, block_series, label="blocks (median)", color="#7c3aed", linewidth=1.5)
    ax.plot(steps, head_series, label="lm_head (median)", color="#d97706", linewidth=1.5)
    ax.plot(steps, median_ratios, label="all-params median", color="#152033", linewidth=1.2, alpha=0.7)
    ax.axhline(1e-3, color="#159447", linestyle="--", linewidth=1, label="target ~1e-3")
    if t_star is not None:
        ax.axvline(t_star, color="#dc2626", linestyle="--", linewidth=1.2, label=f"T*={t_star}")
    ax.set_xlabel("Step")
    ax.set_ylabel("||delta_w|| / ||w||")
    ax.set_yscale("log")
    ax.set_title(f"Update-to-weight ratio (warmup={warmup_steps}, peak_lr={peak_lr})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Late-window median after warmup
    late = [r for s, r in zip(steps, median_ratios) if s >= warmup_steps]
    late_med = sorted(late)[len(late) // 2] if late else 0.0

    md_path = reports / "assgn011_warmup_end.md"
    md_lines = [
        "# Assgn011 — Warmup End / Update-to-Weight Ratio (Task 3)",
        "",
        f"Model: MiniGPT n_embd={n_embd}, n_layer=2, block_size={block_size}.",
        f"Optimizer: AdamW (beta1=0.9, beta2=0.95, wd=0.1), peak_lr={peak_lr}, warmup={warmup_steps} of {n_steps}.",
        "",
        "```text",
        "ratio_layer = ||w_after - w_before||_2 / (||w_after||_2 + eps)",
        "T* = first step after warmup peak where median ratio changes < 5% across a 5-step window",
        "```",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Warmup length W | {warmup_steps} |",
        f"| Detected T* | {t_star} |",
        f"| Late median ratio (after warmup) | {late_med:.4e} |",
        f"| Target | ~1e-3 |",
        f"| CSV rows | {len(rows)} |",
        "",
        "## Reasoning",
        "",
        f"Warmup stops changing the ratio once LR has reached peak (step {warmup_steps}) and the "
        f"per-layer update magnitudes settle. Detected T* = {t_star}. "
        "Healthy post-warmup ratios sit near 1e-3 (FULL.txt Section 9).",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "ok": csv_path.is_file() and t_star is not None and png_path.stat().st_size > 0,
        "t_star": t_star,
        "warmup_steps": warmup_steps,
        "n_rows": len(rows),
        "late_median_ratio": late_med,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "report_path": str(md_path),
    }
