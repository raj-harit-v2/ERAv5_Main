"""Pure-Python / NumPy Adam step for hand verification (Task 1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# FULL.txt Section 6 textbook table
DEFAULT_GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]
DEFAULT_W0 = 1.0
DEFAULT_ETA = 0.001
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPS = 1e-8


@dataclass
class AdamStepRecord:
    t: int
    g: float
    m: float
    v: float
    m_hat: float
    v_hat: float
    step: float
    w: float


def adam_step_hand(
    w: float,
    g: float,
    m: float,
    v: float,
    t: int,
    *,
    eta: float = DEFAULT_ETA,
    beta1: float = DEFAULT_BETA1,
    beta2: float = DEFAULT_BETA2,
    eps: float = DEFAULT_EPS,
    bias_correction: bool = True,
) -> AdamStepRecord:
    """One scalar Adam update. Pure Python — no torch.optim."""
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    if bias_correction:
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
    else:
        m_hat = m
        v_hat = v
    step = eta * m_hat / (math.sqrt(v_hat) + eps)
    w_new = w - step
    return AdamStepRecord(
        t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat, step=step, w=w_new
    )


def run_adam_hand_table(
    grads: list[float] | None = None,
    *,
    w0: float = DEFAULT_W0,
    eta: float = DEFAULT_ETA,
    beta1: float = DEFAULT_BETA1,
    beta2: float = DEFAULT_BETA2,
    eps: float = DEFAULT_EPS,
    bias_correction: bool = True,
) -> list[AdamStepRecord]:
    grads = grads if grads is not None else list(DEFAULT_GRADS)
    w = w0
    m = 0.0
    v = 0.0
    rows: list[AdamStepRecord] = []
    for t, g in enumerate(grads, start=1):
        rec = adam_step_hand(
            w, g, m, v, t, eta=eta, beta1=beta1, beta2=beta2, eps=eps, bias_correction=bias_correction
        )
        m, v, w = rec.m, rec.v, rec.w
        rows.append(rec)
    return rows


def run_adam_pytorch_mirror(
    grads: list[float] | None = None,
    *,
    w0: float = DEFAULT_W0,
    eta: float = DEFAULT_ETA,
    beta1: float = DEFAULT_BETA1,
    beta2: float = DEFAULT_BETA2,
    eps: float = DEFAULT_EPS,
) -> list[dict[str, float]]:
    """Mirror hand Adam with torch.optim.Adam on a single Parameter."""
    grads = grads if grads is not None else list(DEFAULT_GRADS)
    w = nn.Parameter(torch.tensor([w0], dtype=torch.float64))
    opt = torch.optim.Adam([w], lr=eta, betas=(beta1, beta2), eps=eps)
    out: list[dict[str, float]] = []
    for g in grads:
        opt.zero_grad()
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        state = opt.state[w]
        m = float(state["exp_avg"].item())
        v = float(state["exp_avg_sq"].item())
        t = int(state["step"].item()) if hasattr(state["step"], "item") else int(state["step"])
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        step = eta * m_hat / (math.sqrt(v_hat) + eps)
        out.append(
            {
                "t": float(t),
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "step": step,
                "w": float(w.detach().item()),
            }
        )
    return out


def compare_hand_vs_pytorch(
    hand: list[AdamStepRecord],
    torch_rows: list[dict[str, float]],
    tol: float = 1e-6,
) -> dict[str, Any]:
    fields = ("m", "v", "m_hat", "v_hat", "step", "w")
    max_err = 0.0
    details: list[str] = []
    all_ok = True
    for h, p in zip(hand, torch_rows):
        for f in fields:
            hv = getattr(h, f)
            pv = p[f]
            err = abs(hv - pv)
            max_err = max(max_err, err)
            ok = err < tol
            if not ok:
                all_ok = False
            details.append(f"t={h.t} {f}: hand={hv:.10f} torch={pv:.10f} abs_err={err:.3e} {'PASS' if ok else 'FAIL'}")
    return {"ok": all_ok, "max_abs_err": max_err, "tol": tol, "details": details}


def write_adam_hand_artifacts(reports: Path) -> dict[str, Any]:
    """Run Task 1 and write table + PyTorch check artifacts."""
    reports.mkdir(parents=True, exist_ok=True)
    hand = run_adam_hand_table()
    torch_rows = run_adam_pytorch_mirror()
    cmp = compare_hand_vs_pytorch(hand, torch_rows)

    table_path = reports / "assgn011_adam_hand_table.md"
    lines = [
        "# Assgn011 — Adam Hand Table (Task 1)",
        "",
        "Setup: w0 = 1.0, eta = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, bias correction ON.",
        "Gradients: [0.50, 0.40, 0.60, 0.45, 0.55] (FULL.txt Section 6).",
        "",
        "```text",
        "m_t  = beta1 * m_{t-1} + (1 - beta1) * g_t",
        "v_t  = beta2 * v_{t-1} + (1 - beta2) * g_t^2",
        "mhat = m_t / (1 - beta1^t)",
        "vhat = v_t / (1 - beta2^t)",
        "step = eta * mhat / (sqrt(vhat) + eps)",
        "w_t  = w_{t-1} - step",
        "```",
        "",
        "| t | g | m | v | m_hat | v_hat | step | w |",
        "| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in hand:
        lines.append(
            f"| {r.t} | {r.g:.4f} | {r.m:.6f} | {r.v:.6f} | {r.m_hat:.6f} | "
            f"{r.v_hat:.6f} | {r.step:.6f} | {r.w:.6f} |"
        )
    lines.append("")
    table_path.write_text("\n".join(lines), encoding="utf-8")

    check_path = reports / "assgn011_adam_hand_check.txt"
    check_lines = [
        "Assgn011 Adam hand vs torch.optim.Adam",
        f"tolerance: {cmp['tol']}",
        f"max_abs_err: {cmp['max_abs_err']:.3e}",
        f"result: {'PASS' if cmp['ok'] else 'FAIL'}",
        "",
        *cmp["details"],
        "",
    ]
    check_path.write_text("\n".join(check_lines), encoding="utf-8")

    return {
        "ok": cmp["ok"],
        "max_abs_err": cmp["max_abs_err"],
        "n_steps": len(hand),
        "final_w": hand[-1].w if hand else None,
        "table_path": str(table_path),
        "check_path": str(check_path),
        "hand_rows": hand,
    }
