"""Full per-stage corpus audit helper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config as cfg
from src.utils import write_json


def build_stage_audit(
    stage: str,
    *,
    manifests: list[dict[str, Any]],
    firewall_blocked: list[dict[str, Any]],
    mixture_info: dict[str, Any],
    opus_decisions: list[dict[str, Any]],
    consumption_events: list[dict[str, Any]],
    learning_events: list[dict[str, Any]],
    ppl_records: list[dict[str, Any]],
    checkpoint_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_cons = [e for e in consumption_events if e.get("curriculum_stage") == stage]
    stage_learn = [e for e in learning_events if e.get("curriculum_stage") == stage]
    stage_opus = [e for e in opus_decisions if e.get("curriculum_stage") == stage]
    stage_ppl = [e for e in ppl_records if e.get("curriculum_stage") == stage]

    tok_hashes = {m.get("tokenizer_hash") for m in manifests}
    audit = {
        "stage": stage,
        "manifest_check": {
            "n_manifests": len(manifests),
            "tokenizer_hash_unique": list(tok_hashes),
            "all_clean_eval_overlap": all(
                m.get("eval_test_overlap_status") == "clean" for m in manifests
            ),
        },
        "eval_firewall": {
            "blocked_events": len(firewall_blocked),
            "ok": True,
        },
        "mixture_compliance": mixture_info,
        "opus_board": {
            "n_decisions": len(stage_opus),
            "statuses": sorted({d.get("status") for d in stage_opus}),
        },
        "consumption_ledger": {
            "n_events": len(stage_cons),
            "batch_ids": [e.get("batch_id") for e in stage_cons],
        },
        "token_perplexity": {
            "n_records": len(stage_ppl),
            "avg_ppl": (
                sum(r["avg_ppl"] for r in stage_ppl) / len(stage_ppl) if stage_ppl else None
            ),
            "mastered": [r["shard_id"] for r in stage_ppl if r.get("mastered")],
        },
        "learning_report_card": {
            "n_events": len(stage_learn),
            "classifications": sorted({e.get("classification") for e in stage_learn}),
        },
        "stage_exit_checkpoint": {
            "present": checkpoint_meta is not None,
            "ledger_offset": (checkpoint_meta or {}).get("ledger_offset"),
        },
    }
    return audit


def write_stage_audit(audit: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or cfg.STAGE_AUDITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stage_{audit['stage']}.json"
    write_json(path, audit)
    return path
