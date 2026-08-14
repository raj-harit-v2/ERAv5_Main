"""embedding_policy_id record."""
from __future__ import annotations

from typing import Any

import config as cfg
from src.utils import write_json


def build_embedding_policy(tokenizer_hash: str) -> dict[str, Any]:
    short = tokenizer_hash[:12]
    policy_id = (
        f"fourier_v1|code{cfg.FOURIER_CODE_DIM}|freq{cfg.FOURIER_N_FREQ}"
        f"|{cfg.POSITION_POLICY}|canine={int(cfg.USE_FOURIER_CANINE)}"
        f"|vq={int(cfg.ENABLE_VQ_PROBLEM5)}|untied|tok_{short}"
    )
    return {
        "embedding_policy_id": policy_id,
        "problem_id": cfg.PROBLEM_ID,
        "embedding_type": "fourier_codepoint_sum",
        "upgrades": {
            "position_policy": cfg.POSITION_POLICY,
            "fourier_canine": cfg.USE_FOURIER_CANINE,
            "canine_stride": cfg.CANINE_STRIDE,
            "vq_problem5_enabled": cfg.ENABLE_VQ_PROBLEM5,
            "vq_num_codes": cfg.VQ_NUM_CODES if cfg.ENABLE_VQ_PROBLEM5 else None,
        },
        "controls": ["dense", "kronecker_byte", "fourier_baseline"],
        "primary_upgrade_arm": "fourier_canine" if cfg.USE_FOURIER_CANINE else "fourier",
        "char_unit": "unicode_codepoint",
        "code_dim": cfg.FOURIER_CODE_DIM,
        "n_freq": cfg.FOURIER_N_FREQ,
        "pos_dim_kronecker_baseline": cfg.POS_DIM_KRON,
        "max_chars_fourier": cfg.MAX_CHARS_FOURIER,
        "projection_trainable": True,
        "freeze_schedule": "diagnostic_mid_indic_focus_only",
        "tying": "untied",
        "position_policy": cfg.POSITION_POLICY,
        "requires_token_text": True,
        "tokenizer_hash": tokenizer_hash,
        "schema_version": cfg.SCHEMA_VERSION,
    }


def write_embedding_policy(policy: dict[str, Any], path) -> None:
    write_json(path, policy)
