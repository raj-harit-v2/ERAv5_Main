"""Sanity checks: initial loss, ppl threshold helpers."""
from __future__ import annotations

import math
from typing import Any


def initial_loss_ok(loss: float, vocab_size: int, tol: float = 1.0) -> tuple[bool, float]:
    expected = math.log(vocab_size)
    return abs(loss - expected) < tol, expected


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"n": 0}
    losses = [h["loss"] for h in history]
    return {
        "n": len(history),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
    }
