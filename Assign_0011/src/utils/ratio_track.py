"""Per-layer update-to-weight ratio logging (Task 3)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def layer_update_ratios(
    model: nn.Module,
    before: dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute ||delta_w|| / (||w|| + eps) for every named parameter."""
    ratios: dict[str, float] = {}
    for name, p in model.named_parameters():
        prev = before[name]
        delta = (p.detach() - prev).norm(2).item()
        w_norm = p.detach().norm(2).item()
        ratios[name] = delta / (w_norm + eps)
    return ratios


def group_ratio(ratios: dict[str, float], prefixes: Iterable[str]) -> float | None:
    """Median ratio over parameters whose name starts with any of the prefixes."""
    vals = [v for name, v in ratios.items() if any(name.startswith(p) for p in prefixes)]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    if len(vals_sorted) % 2:
        return vals_sorted[mid]
    return 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])


def detect_warmup_end(
    steps: list[int],
    median_ratios: list[float],
    peak_step: int,
    window: int = 5,
    rel_tol: float = 0.05,
) -> int | None:
    """First step after peak LR where median ratio changes < rel_tol across a window."""
    if not steps or not median_ratios or len(steps) != len(median_ratios):
        return None
    # Find index where step >= peak_step
    start_idx = 0
    for i, s in enumerate(steps):
        if s >= peak_step:
            start_idx = i
            break
    for i in range(start_idx, len(median_ratios) - window):
        window_vals = median_ratios[i : i + window]
        base = window_vals[0]
        if base < 1e-12:
            continue
        if all(abs(v - base) / base < rel_tol for v in window_vals[1:]):
            return steps[i]
    # Fallback: peak step itself
    return peak_step if peak_step in steps else (steps[start_idx] if steps else None)
