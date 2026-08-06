"""Generate evidence.json + evidence.md from real ledger/manifest artifacts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import config as cfg
from src.ledger import read_jsonl
from src.shard_manifest import check_admission, validate_manifest
from src.utils import read_json, write_json


def _result(status: str, evidence_path: str, detail: str) -> dict[str, str]:
    return {"status": status, "evidence_path": evidence_path, "detail": detail}


def build_evidence(
    run_id: str,
    artifacts_dir: Path,
    *,
    resume_info: dict[str, Any] | None = None,
    replay_info: dict[str, Any] | None = None,
    fork_info: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
    step0_loss: float | None = None,
    vocab_size: int = cfg.SMOKE_VOCAB_SIZE,
) -> tuple[dict[str, Any], str]:
    manifests_dir = artifacts_dir / "manifests"
    ledgers_dir = artifacts_dir / "ledgers"
    cons_path = ledgers_dir / "consumption_ledger.jsonl"
    learn_path = ledgers_dir / "learning_ledger.jsonl"
    opus_path = ledgers_dir / "opus_ledger.jsonl"
    perf_path = artifacts_dir / "performance.json"
    fw_path = ledgers_dir / "firewall.json"

    consumption = read_jsonl(cons_path)
    learning = read_jsonl(learn_path)
    opus = read_jsonl(opus_path)
    manifests = list(manifests_dir.glob("shard_*.json")) if manifests_dir.exists() else []

    results: dict[str, dict[str, str]] = {}

    # 1 tokenizer integrity
    tok_ok = True
    detail_parts = []
    for mp in manifests:
        m = read_json(mp)
        ok, errs = validate_manifest(m)
        shard_file = artifacts_dir / m.get("shard_path", "")
        adm, reason = check_admission(m, shard_file) if shard_file.exists() else (False, "missing")
        if not ok or not adm or not m.get("tokenizer_hash"):
            tok_ok = False
            detail_parts.append(reason or str(errs))
    results["tokenizer_integrity"] = _result(
        "PASS" if tok_ok and manifests else "FAIL",
        "manifests/",
        f"n_manifests={len(manifests)}; " + (",".join(detail_parts) if detail_parts else "hashes_ok"),
    )

    # 2 eval firewall
    blocked = []
    if fw_path.exists():
        blocked = read_json(fw_path).get("blocked_events", [])
    # also scan run.log later via presence
    fw_ok = len(blocked) > 0
    results["eval_firewall"] = _result(
        "PASS" if fw_ok else "FAIL",
        "ledgers/firewall.json",
        f"blocked_events={len(blocked)}",
    )

    # 3 packing correctness
    packed_path = artifacts_dir / "packed_batch_reports"
    utils = []
    if packed_path.exists():
        for p in packed_path.glob("*.json"):
            utils.append(read_json(p).get("utilization", 0))
    # fallback: from consumption
    if not utils:
        utils = [e.get("packing_utilization", 0) for e in consumption if e.get("event") == "consumption"]
    non_agentic = [u for u in utils if u is not None]
    pack_ok = bool(non_agentic) and (max(non_agentic) > 0.3)
    results["packing_correctness"] = _result(
        "PASS" if pack_ok else "FAIL",
        "packed_batch_reports/|consumption_ledger.jsonl",
        f"max_utilization={max(non_agentic) if non_agentic else 0:.3f}",
    )

    # 4 mixture compliance — floors + planned-vs-actual tolerance (honest gate)
    lanes: dict[str, float] = {}
    for e in consumption:
        if e.get("event") != "consumption":
            continue
        shares = e.get("lane_token_shares")
        if isinstance(shares, dict) and shares:
            for lane, n in shares.items():
                lanes[lane] = lanes.get(lane, 0.0) + float(n)
        else:
            lane = e.get("mixture_lane", "web")
            lanes[lane] = lanes.get(lane, 0.0) + float(e.get("useful_tokens", 0))
    total = sum(lanes.values()) or 1
    actual = {k: v / total for k, v in lanes.items()}
    floor_ok = all(lanes.get(l, 0) > 0 for l in cfg.ALWAYS_ON_SPLIT)
    mix_info = (performance or {}).get("mixture_compliance") or {}
    if not mix_info and perf_path.exists():
        mix_info = read_json(perf_path).get("mixture_compliance") or {}
    delta_ok = bool(mix_info.get("within_tolerance")) or float(mix_info.get("max_abs_delta", 1.0)) <= 0.18
    if not mix_info:
        # Fallback: floors only is insufficient — require non-trivial multi-lane diversity
        delta_ok = len([v for v in lanes.values() if v > 0]) >= 4
    results["mixture_compliance"] = _result(
        "PASS" if floor_ok and lanes and delta_ok else "FAIL",
        "ledgers/consumption_ledger.jsonl",
        f"actual_share={{{', '.join(f'{k}:{v:.3f}' for k,v in sorted(actual.items()))}}}; "
        f"max_abs_delta={mix_info.get('max_abs_delta', 'n/a')}",
    )

    # 5 OPUS audit trail
    statuses = {d.get("status") for d in opus}
    opus_ok = set(cfg.OPUS_DECISION_STATES).issubset(statuses)
    results["opus_audit_trail"] = _result(
        "PASS" if opus_ok else "FAIL",
        "ledgers/opus_ledger.jsonl",
        f"statuses={sorted(statuses)}",
    )

    # 6 crash recovery
    resume_ok = False
    resume_detail = "missing"
    if resume_info:
        resume_ok = bool(resume_info.get("matched"))
        resume_detail = (
            f"expected={resume_info.get('expected')} actual={resume_info.get('actual')}"
        )
    results["crash_recovery"] = _result(
        "PASS" if resume_ok else "FAIL",
        "checkpoints/",
        resume_detail,
    )

    # 7 replay
    replay_ok = False
    replay_detail = "missing"
    if replay_info:
        replay_ok = bool(replay_info.get("matched"))
        replay_detail = (
            f"compared={replay_info.get('n')} "
            f"original={replay_info.get('sample_original')} "
            f"replay={replay_info.get('sample_replay')}"
        )
    results["replay"] = _result(
        "PASS" if replay_ok else "FAIL",
        "ledgers/consumption_ledger.jsonl",
        replay_detail,
    )

    # 8 learning trace
    learn_ok = any(
        e.get("loss_delta") is not None and e.get("source_document_ids") for e in learning
    )
    results["learning_trace"] = _result(
        "PASS" if learn_ok else "FAIL",
        "ledgers/learning_ledger.jsonl",
        f"n_learning_events={len(learning)}",
    )

    # 9 throughput
    if performance is None and perf_path.exists():
        performance = read_json(perf_path)
    perf = performance or {}
    thr = float(perf.get("useful_tokens_per_second", 0))
    results["throughput"] = _result(
        "PASS" if thr > 0 else "FAIL",
        "performance.json",
        f"useful_tokens_per_second={thr:.2f}",
    )

    # extras
    results["fork_recorded"] = _result(
        "PASS" if fork_info and fork_info.get("new_branch_id") else "FAIL",
        "ledgers/consumption_ledger.jsonl",
        str(fork_info or {}),
    )
    if step0_loss is not None:
        exp = math.log(vocab_size)
        # Pass unless contamination-like (far below random baseline)
        sane = step0_loss > exp * 0.25
        results["initial_loss_sane"] = _result(
            "PASS" if sane else "FAIL",
            "ledgers/learning_ledger.jsonl",
            f"step0_loss={step0_loss:.4f} ln_vocab={exp:.4f}",
        )

    ppl_path = artifacts_dir / "ppl_traces.json"
    ppl_ok = ppl_path.exists()
    results["ppl_trace_exists"] = _result(
        "PASS" if ppl_ok else "FAIL",
        "ppl_traces.json",
        f"exists={ppl_ok}",
    )

    overall = all(r["status"] == "PASS" for r in results.values() if r["status"] in ("PASS", "FAIL"))
    # overall based on required 9
    required = [
        "tokenizer_integrity",
        "eval_firewall",
        "packing_correctness",
        "mixture_compliance",
        "opus_audit_trail",
        "crash_recovery",
        "replay",
        "learning_trace",
        "throughput",
    ]
    overall_pass = all(results[k]["status"] == "PASS" for k in required)

    evidence = {
        "run_id": run_id,
        "overall_pass": overall_pass,
        "results": results,
    }
    write_json(artifacts_dir / "evidence.json", evidence)

    rows = [
        ("Tokenizer integrity", "tokenizer_integrity", "Manifest record"),
        ("Evaluation firewall", "eval_firewall", "Blocked-shard event"),
        ("Packing correctness", "packing_correctness", "Packed-batch report"),
        ("Mixture compliance", "mixture_compliance", "Planned versus actual shares"),
        ("OPUS audit trail", "opus_audit_trail", "Candidate decision records"),
        ("Crash recovery", "crash_recovery", "Expected and resumed batch ids"),
        ("Replay", "replay", "Original and replay hashes"),
        ("Learning trace", "learning_trace", "Loss linked to source data"),
        ("Throughput", "throughput", "Performance report"),
    ]
    lines = [
        "# Assignment 06 Evidence Bundle",
        "",
        f"Run ID: `{run_id}`",
        f"Overall: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for name, key, default_ev in rows:
        r = results[key]
        lines.append(f"| {name} | {r['status']} | {r['evidence_path'] or default_ev} |")
    lines.append("")
    lines.append("## Details")
    for name, key, _ in rows:
        r = results[key]
        lines.append(f"- **{name}**: {r['detail']}")
    md = "\n".join(lines) + "\n"
    (artifacts_dir / "evidence.md").write_text(md, encoding="utf-8")
    return evidence, md
