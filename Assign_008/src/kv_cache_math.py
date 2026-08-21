"""Exact KV-cache byte formula (Full Document §10).

Correction: do NOT multiply an extra trailing x2 beyond P_b.
bytes = 2 * L * H_KV * d_head * T * B * P_b
"""

from __future__ import annotations

import config


def kv_cache_bytes(
    *,
    layers: int,
    h_kv: int,
    d_head: int,
    t: int,
    batch: int,
    p_b: int,
) -> int:
    return 2 * layers * h_kv * d_head * t * batch * p_b


def yardstick_one_user_32k() -> int:
    return kv_cache_bytes(
        layers=config.YARDSTICK_L,
        h_kv=config.YARDSTICK_H_KV,
        d_head=config.YARDSTICK_D_HEAD,
        t=32_768,
        batch=1,
        p_b=config.P_BYTES,
    )
