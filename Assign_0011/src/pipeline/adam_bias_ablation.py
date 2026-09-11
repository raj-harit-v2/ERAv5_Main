"""Task 2: bias-correction ablation — plot first 20 steps both ways."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.utils.adam_hand import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPS,
    DEFAULT_ETA,
    adam_step_hand,
)


def run_bias_ablation(
    n_steps: int = 20,
    g: float = 0.5,
    *,
    eta: float = DEFAULT_ETA,
    beta1: float = DEFAULT_BETA1,
    beta2: float = DEFAULT_BETA2,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Constant-sign gradient Adam with / without bias correction."""
    steps_on: list[float] = []
    steps_off: list[float] = []
    w_on, m_on, v_on = 1.0, 0.0, 0.0
    w_off, m_off, v_off = 1.0, 0.0, 0.0

    for t in range(1, n_steps + 1):
        rec_on = adam_step_hand(
            w_on, g, m_on, v_on, t, eta=eta, beta1=beta1, beta2=beta2, eps=eps, bias_correction=True
        )
        rec_off = adam_step_hand(
            w_off, g, m_off, v_off, t, eta=eta, beta1=beta1, beta2=beta2, eps=eps, bias_correction=False
        )
        steps_on.append(rec_on.step)
        steps_off.append(rec_off.step)
        w_on, m_on, v_on = rec_on.w, rec_on.m, rec_on.v
        w_off, m_off, v_off = rec_off.w, rec_off.m, rec_off.v

    # Theoretical t=1 ratio for constant g: (1-beta1)/sqrt(1-beta2) ≈ 3.162
    theory_ratio = (1.0 - beta1) / math.sqrt(1.0 - beta2)
    measured_ratio = steps_off[0] / max(steps_on[0], 1e-30)

    # Smallest T such that for all t >= T, relative |on-off|/max(|on|,eps) < 1e-3
    crossover: int | None = None
    for t in range(1, n_steps + 1):
        ok_rest = True
        for k in range(t - 1, n_steps):
            denom = max(abs(steps_on[k]), 1e-30)
            rel = abs(steps_on[k] - steps_off[k]) / denom
            if rel >= 1e-3:
                ok_rest = False
                break
        if ok_rest:
            crossover = t
            break

    return {
        "steps_on": steps_on,
        "steps_off": steps_off,
        "theory_ratio_t1": theory_ratio,
        "measured_ratio_t1": measured_ratio,
        "crossover_t": crossover,
        "n_steps": n_steps,
        "eta": eta,
        "g": g,
    }


def write_bias_ablation_artifacts(reports: Path) -> dict[str, Any]:
    reports.mkdir(parents=True, exist_ok=True)
    data = run_bias_ablation()

    png_path = reports / "assgn011_bias_correction_20.png"
    xs = list(range(1, data["n_steps"] + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, data["steps_on"], label="with_bias_corr", color="#159447", linewidth=2)
    ax.plot(xs, data["steps_off"], label="without_bias_corr", color="#7c3aed", linewidth=2)
    ax.axhline(data["eta"], color="#65758b", linestyle=":", linewidth=1, label="eta (target)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Adam step size")
    ax.set_title("Bias correction ablation — first 20 steps (constant g=0.5)")
    ax.legend()
    ax.annotate(
        f"t=1 ratio off/on = {data['measured_ratio_t1']:.3f}\n(theory ≈ {data['theory_ratio_t1']:.3f})",
        xy=(1, data["steps_off"][0]),
        xytext=(4, data["steps_off"][0] * 0.85),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#7c3aed"),
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = reports / "assgn011_bias_correction_report.md"
    crossover = data["crossover_t"]
    if crossover is None:
        stop_note = (
            "Within the first 20 steps the relative difference never stays below 1e-3 for all "
            "remaining t. Study Guide notes the asymptotic gap shrinks over ~1000 steps; "
            "the 20-step plot is the primary deliverable."
        )
        crossover_str = "not reached within 20 steps"
    else:
        stop_note = f"Relative |step_on - step_off| / |step_on| stays below 1e-3 for all t >= {crossover}."
        crossover_str = str(crossover)

    lines = [
        "# Assgn011 — Bias Correction Ablation (Task 2)",
        "",
        "Constant gradient g = 0.5, eta = 0.001, beta1 = 0.9, beta2 = 0.999.",
        "",
        "```text",
        "t=1 without correction: step ≈ eta * (1 - beta1) / sqrt(1 - beta2) ≈ 3.162 * eta",
        "t=1 with correction:    step ≈ eta * 1.000",
        "```",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Measured off/on ratio at t=1 | {data['measured_ratio_t1']:.6f} |",
        f"| Theory (1-beta1)/sqrt(1-beta2) | {data['theory_ratio_t1']:.6f} |",
        f"| Crossover T (rel err < 1e-3 thereafter) | {crossover_str} |",
        f"| Primary plot | `{png_path.name}` |",
        "",
        "## When the difference stops mattering",
        "",
        stop_note,
        "",
        "Production runs never disable bias correction; this ablation is pedagogical only.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    ratio_ok = abs(data["measured_ratio_t1"] - data["theory_ratio_t1"]) < 1e-2
    png_ok = png_path.is_file() and png_path.stat().st_size > 0

    return {
        "ok": ratio_ok and png_ok,
        "measured_ratio_t1": data["measured_ratio_t1"],
        "theory_ratio_t1": data["theory_ratio_t1"],
        "crossover_t": crossover,
        "png_path": str(png_path),
        "report_path": str(report_path),
    }
