"""GQA / MHA / MQA head-layout helpers.

Correction: transcript 'NQA' → MQA.
"""

from __future__ import annotations


def kv_reduction(h_q: int, h_kv: int) -> float:
    if h_kv <= 0:
        raise ValueError("h_kv must be positive")
    if h_q % h_kv != 0:
        raise ValueError("h_q must be divisible by h_kv for GQA")
    return h_q / h_kv


def layout_name(h_q: int, h_kv: int) -> str:
    if h_kv == h_q:
        return "MHA"
    if h_kv == 1:
        return "MQA"
    if 1 < h_kv < h_q:
        return "GQA"
    raise ValueError(f"Invalid head layout H_Q={h_q}, H_KV={h_kv}")
