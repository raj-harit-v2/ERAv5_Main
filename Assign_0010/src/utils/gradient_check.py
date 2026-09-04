"""Numerical vs analytical gradient verification (Session 10 Task 2)."""

from __future__ import annotations

from pathlib import Path

import torch


def verify_two_weight_chain(out_path: Path, nudge: float = 1e-6) -> dict:
    """Classic chain from study guide: h=w1*x, y=w2*h, L=(y-t)^2."""
    dtype = torch.float64
    x = torch.tensor([2.0], dtype=dtype)
    w1 = torch.tensor([3.0], dtype=dtype, requires_grad=True)
    w2 = torch.tensor([4.0], dtype=dtype, requires_grad=True)
    target = torch.tensor([20.0], dtype=dtype)

    h = w1 * x
    y = w2 * h
    loss = (y - target) ** 2
    loss.backward()
    analytical = w1.grad.item()

    with torch.no_grad():
        w1_plus = w1 + nudge
        w1_minus = w1 - nudge
        loss_plus = (w2 * (w1_plus * x) - target) ** 2
        loss_minus = (w2 * (w1_minus * x) - target) ** 2
        numerical = (loss_plus.item() - loss_minus.item()) / (2 * nudge)

    diff = abs(analytical - numerical)
    passed = diff < 1e-4

    lines = [
        "--- Analytical Backpropagation (PyTorch) ---",
        f"Loss: {loss.item():.4f}",
        f"y: {y.item():.4f}",
        f"Analytical gradient dL/dw1: {analytical:.8f}",
        "",
        "--- Numerical Gradient (Manual Nudge) ---",
        f"Nudge size (central diff): {nudge}",
        f"Loss at w1+eps: {loss_plus.item():.6f}",
        f"Loss at w1-eps: {loss_minus.item():.6f}",
        f"Numerical gradient dL/dw1: {numerical:.8f}",
        "",
        f"Absolute difference: {diff:.8f}",
        f"RESULT: {'PASS' if passed else 'FAIL'} (tolerance 1e-4)",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "path": str(out_path),
        "analytical": analytical,
        "numerical": numerical,
        "diff": diff,
        "passed": passed,
    }
