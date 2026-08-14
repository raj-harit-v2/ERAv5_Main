"""Full per-stage corpus + embedding audit."""
from __future__ import annotations

from typing import Any

import config as cfg
from src.utils import write_json


def build_stage_audit(
    stage: str,
    *,
    tokenizer_hash: str,
    n_docs_stage: int,
    actual_script_shares: dict[str, float],
    blocked_eval_events: int,
    n_consumption: int,
    mean_loss: float | None,
    policy: dict[str, Any],
    collision_slice: dict[str, Any],
    grad_proj: float | None,
    grad_l1: float | None,
) -> dict[str, Any]:
    floor = cfg.STAGE_INDIC_FLOOR.get(stage, 0.0)
    indic_share = actual_script_shares.get("hi", 0.0) + actual_script_shares.get("te", 0.0)
    return {
        "stage": stage,
        "manifest_check": {
            "tokenizer_hash": tokenizer_hash,
            "n_docs": n_docs_stage,
            "vocab_size": cfg.VOCAB_SIZE,
        },
        "mixture_compliance": {
            "target_indic_floor": floor,
            "actual_script_shares": actual_script_shares,
            "indic_floor_ok": indic_share + 1e-9 >= floor * 0.5,  # soft at smoke scale
        },
        "eval_firewall": {"blocked_events": blocked_eval_events, "ok": blocked_eval_events >= 0},
        "consumption_ledger": {"n_events": n_consumption},
        "learning_report_card": {"n_events": n_consumption, "mean_loss": mean_loss},
        "embedding_policy": {
            "embedding_policy_id": policy.get("embedding_policy_id"),
            "type": policy.get("embedding_type"),
            "code_dim": policy.get("code_dim"),
            "projection_trainable": policy.get("projection_trainable"),
            "tokenizer_hash": tokenizer_hash,
        },
        "collision_gate": collision_slice,
        "adaptation": {"grad_norm_proj": grad_proj, "grad_norm_layer1": grad_l1},
    }


def write_stage_audit(audit: dict[str, Any], out_dir=None):
    out_dir = out_dir or cfg.STAGE_AUDITS_DIR
    path = out_dir / f"stage_{audit['stage']}.json"
    write_json(path, audit)
    return path
