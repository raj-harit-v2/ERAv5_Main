"""Self-diagnosis smoke checks S0…"""

from __future__ import annotations

from dataclasses import dataclass

import config
from src.embeddings import v5_dense_param_count
from src.kv_cache_math import kv_cache_bytes, yardstick_one_user_32k


@dataclass
class Check:
    id: str
    ok: bool
    detail: str


def run_checks() -> list[Check]:
    checks: list[Check] = []
    # S0: never allocate V5 table
    n = v5_dense_param_count()
    checks.append(
        Check("S0_v5_account_only", n == config.V5_REF_V * config.V5_REF_D, f"|Theta_in|={n:,}")
    )
    # S1: smoke dims
    ok_dims = config.D_MODEL % config.N_HEADS == 0 and config.SEQ_LEN <= 64
    checks.append(Check("S1_smoke_dims", ok_dims, f"D={config.D_MODEL} H={config.N_HEADS} T={config.SEQ_LEN}"))
    # S2: KV formula yardstick ~6.44GB
    b = yardstick_one_user_32k()
    expected = 2 * 48 * 8 * 128 * 32768 * 1 * 2
    checks.append(Check("S2_kv_yardstick", b == expected, f"bytes={b:,}"))
    # S3: pedagogical ctx not used as tensor size
    checks.append(
        Check(
            "S3_pedagogical_not_allocated",
            config.PEDAGOGICAL_CTX == 256_000 and config.SEQ_LEN < 1000,
            f"PEDAGOGICAL_CTX={config.PEDAGOGICAL_CTX} SEQ_LEN={config.SEQ_LEN}",
        )
    )
    # S4: HF off
    checks.append(Check("S4_hf_off", not config.USE_HF, f"USE_HF={config.USE_HF}"))
    return checks
