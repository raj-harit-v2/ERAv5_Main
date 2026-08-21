"""Effective document length under tokenizer fertility.

Assumption A8: Telugu ~3x is a teaching ratio, not a measured constant.
EffectiveContent(T, f_L) ≈ T / f_L
"""

from __future__ import annotations


def effective_content(t_tokens: int, fertility: float) -> float:
    if fertility <= 0:
        raise ValueError("fertility must be positive")
    return t_tokens / fertility
