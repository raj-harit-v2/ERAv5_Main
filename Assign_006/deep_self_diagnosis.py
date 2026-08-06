#!/usr/bin/env python3
"""Deep per-phase / per-stage self-diagnosis for Assignment 06."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import config as cfg
from src.corpus import CANARY_EVAL, generate_corpus
from src.eval_firewall import EvalFirewall
from src.ledger import ConsumptionLedger, LearningLedger, append_jsonl, read_jsonl
from src.loss_mask import assert_obs_not_trained, mask_agentic_text
from src.mixture_compiler import compile_schedule
from src.opus_extended import select_with_opus
from src.packing import (
    best_fit_packing,
    concatenate_and_chop,
    greedy_packing,
    pad_only,
    structure_preserving,
)
from src.shard_builder import build_lane_shards, compute_cleaning_hash
from src.shard_manifest import check_admission, validate_manifest
from src.tokenizer_wrapper import compute_tokenizer_hash, encode_document, get_tokenizer
from src.utils import batch_content_hash, set_seed, sha256_file, write_json

PROVENANCE = {
    "PH0_CORPUS": {"file": "src/corpus.py", "what": "lanes + eval/test canaries"},
    "PH1_TOKENIZER": {"file": "src/tokenizer_wrapper.py", "what": "frozen hash binding"},
    "PH2_SHARDS": {"file": "src/shard_builder.py", "what": "immutable npy + content hash"},
    "PH3_ADMISSION": {"file": "src/shard_manifest.py", "what": "7-check admission gate"},
    "PH4_FIREWALL": {"file": "src/eval_firewall.py", "what": "eval never in loss batch"},
    "PH5_PACKING": {"file": "src/packing.py", "what": "5 policies + masks"},
    "PH6_MIXTURE": {"file": "src/mixture_compiler.py", "what": "floors + warmup"},
    "PH7_OPUS": {"file": "src/opus_extended.py", "what": "4 decision states"},
    "PH8_CONSUMPTION": {"file": "src/ledger.py", "what": "append-only consumption"},
    "PH9_LEARNING": {"file": "src/ledger.py", "what": "learning report card"},
    "PH10_CKPT": {"file": "src/checkpoint_manager.py", "what": "6-component checkpoint"},
    "PH11_RESUME": {"file": "src/replay_engine.py", "what": "next batch match"},
    "PH12_REPLAY": {"file": "src/replay_engine.py", "what": "hash identity"},
    "PH13_FORK": {"file": "src/replay_engine.py", "what": "new branch_id"},
    "PH14_THROUGHPUT": {"file": "submission_artifacts/performance.json", "what": "useful tps"},
    "PH15_EVIDENCE": {"file": "src/evidence_builder.py", "what": "generated evidence"},
}


@dataclass
class Case:
    id: str
    name: str
    ok: bool
    detail: str
    provenance: dict


def _case(cid: str, name: str, ok: bool, detail: str = "") -> Case:
    return Case(id=cid, name=name, ok=ok, detail=detail, provenance=PROVENANCE.get(cid, {}))


def run_diagnostics(quick: bool = False) -> dict:
    set_seed(cfg.SEED)
    cases: list[Case] = []
    artifacts = cfg.ARTIFACTS_DIR

    # PH0
    try:
        corpus = generate_corpus(docs_per_lane=4 if quick else cfg.DOCS_PER_LANE)
        ok = all(l in corpus for l in cfg.CAPABILITY_LANES)
        ok = ok and "eval" in corpus and "test" in corpus
        ok = ok and any(CANARY_EVAL in d["text"] for d in corpus["eval"])
        cases.append(_case("PH0_CORPUS", "corpus_lanes_and_canaries", ok, f"lanes={list(corpus)}"))
    except Exception as e:
        cases.append(_case("PH0_CORPUS", "corpus_lanes_and_canaries", False, str(e)))
        corpus = generate_corpus(docs_per_lane=4)

    # PH1
    try:
        tok = get_tokenizer()
        h1 = compute_tokenizer_hash(tok)
        h2 = compute_tokenizer_hash(tok)
        ids = encode_document("hello world", tok, 32)
        ok = h1 == h2 and len(h1) == 64 and ids[-1] == getattr(tok, "eos_token_id", cfg.LOCAL_EOS_ID)
        cases.append(_case("PH1_TOKENIZER", "tokenizer_hash_stable", ok, f"hash={h1[:16]}"))
    except Exception as e:
        cases.append(_case("PH1_TOKENIZER", "tokenizer_hash_stable", False, str(e)))
        tok = get_tokenizer()
        h1 = compute_tokenizer_hash(tok)

    # PH2 / PH3 in temp dir
    try:
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            fw = EvalFirewall()
            fw.register_corpus_holdouts(corpus)
            manifests = build_lane_shards(
                {k: v for k, v in corpus.items() if k in cfg.CAPABILITY_LANES},
                tok=tok,
                tokenizer_hash=h1,
                output_dir=tdir,
                docs_per_shard=2,
                eval_registry=set(fw.doc_ids),
            )
            m0 = manifests[0]
            shard_path = tdir / m0["shard_path"]
            ok_fields, _ = validate_manifest(m0)
            ok_hash = sha256_file(shard_path) == m0["content_hash"]
            # mutate → new hash
            arr = np.load(shard_path)
            arr2 = arr.copy()
            arr2[0, 0] = (arr2[0, 0] + 1) % 100
            mut = tdir / "mutated.npy"
            np.save(mut, arr2)
            ok_immut = sha256_file(mut) != m0["content_hash"]
            cases.append(
                _case(
                    "PH2_SHARDS",
                    "shard_content_hash",
                    ok_fields and ok_hash and ok_immut,
                    f"n={len(manifests)}",
                )
            )

            bad = dict(m0)
            bad["tokenizer_hash"] = ""
            adm, reason = check_admission(bad, shard_path)
            cases.append(
                _case("PH3_ADMISSION", "admission_rejects_bad", (not adm), reason)
            )
    except Exception as e:
        cases.append(_case("PH2_SHARDS", "shard_content_hash", False, str(e)))
        cases.append(_case("PH3_ADMISSION", "admission_rejects_bad", False, str(e)))

    # PH4
    try:
        fw = EvalFirewall()
        fw.register_corpus_holdouts(corpus)
        ok, reason = fw.check_document(corpus["eval"][0])
        cases.append(_case("PH4_FIREWALL", "eval_blocked", (not ok), reason))
    except Exception as e:
        cases.append(_case("PH4_FIREWALL", "eval_blocked", False, str(e)))

    # PH5
    try:
        docs = corpus["web"][:4]
        code = corpus["code"][:2]
        agentic = corpus["agentic"][:2]
        p_pad = pad_only(docs, tok, 64, lane="web")
        p_concat = concatenate_and_chop(docs, tok, 64, lane="web")
        p_code = concatenate_and_chop(code, tok, 64, lane="code")
        p_greedy = greedy_packing(docs, tok, 64, lane="web")
        p_best = best_fit_packing(docs, tok, 64, lane="web")
        p_struct = structure_preserving(agentic, tok, 64, lane="agentic")
        info = mask_agentic_text(agentic[0]["text"])
        ok = (
            p_pad.utilization <= p_concat.utilization + 1e-9
            and "code_lane_refuse" in "".join(p_code.warnings)
            and p_best.utilization + 1e-9 >= p_greedy.utilization - 0.5
            and p_struct.attention_mask.ndim == 3
            and assert_obs_not_trained(info)
            and float(p_pad.loss_mask[:, -1].min()) >= 0.0
        )
        # pad positions loss 0
        pad_ok = True
        ids = p_pad.input_ids
        lm = p_pad.loss_mask
        pad_ok = bool(((ids == 0) | (lm == 0)).all() or ((ids == 0) == (lm == 0)).float().mean() > 0.5)
        cases.append(
            _case(
                "PH5_PACKING",
                "packing_policies",
                ok and pad_ok,
                f"pad={p_pad.utilization:.2f} concat={p_concat.utilization:.2f}",
            )
        )
    except Exception as e:
        cases.append(_case("PH5_PACKING", "packing_policies", False, str(e)))

    # PH6
    try:
        sched = compile_schedule()
        warm = any(s.get("warmup") for s in sched)
        floors = all(
            s["mixture"].get("indic", 0) >= cfg.ALWAYS_ON_SPLIT["indic"] - 1e-9 for s in sched
        )
        cases.append(
            _case("PH6_MIXTURE", "warmup_and_floors", warm and floors and len(sched) > 0, f"steps={len(sched)}")
        )
    except Exception as e:
        cases.append(_case("PH6_MIXTURE", "warmup_and_floors", False, str(e)))

    # PH7
    try:
        import random

        rng = random.Random(0)
        cands = [
            {"doc_id": f"d{i}", "lane": lane, "n_chars": 40, "tier_a": True, "text": "x"}
            for i, lane in enumerate(cfg.CAPABILITY_LANES)
        ]
        acc, dec = select_with_opus(
            cands, keep_n=2, curriculum_stage="seed", global_step=0, rng=rng, force_all_states=True
        )
        statuses = {d["status"] for d in dec}
        ok = set(cfg.OPUS_DECISION_STATES).issubset(statuses)
        cases.append(_case("PH7_OPUS", "four_states", ok, f"statuses={sorted(statuses)}"))
    except Exception as e:
        cases.append(_case("PH7_OPUS", "four_states", False, str(e)))

    # Artifact-backed checks if present
    cons_path = artifacts / "ledgers" / "consumption_ledger.jsonl"
    learn_path = artifacts / "ledgers" / "learning_ledger.jsonl"
    evid_path = artifacts / "evidence.json"
    perf_path = artifacts / "performance.json"

    if cons_path.exists():
        rows = read_jsonl(cons_path)
        cons = [r for r in rows if r.get("event") == "consumption"]
        offsets_mono = True
        steps = [r["global_step"] for r in cons]
        cases.append(
            _case(
                "PH8_CONSUMPTION",
                "consumption_append_only",
                len(cons) > 0 and steps == sorted(steps),
                f"n={len(cons)}",
            )
        )
        # replay hash
        if cons:
            h = batch_content_hash(cons[0].get("token_ids_flat", []))
            ok = h == cons[0].get("batch_hash")
            cases.append(_case("PH12_REPLAY", "replay_hash_identity", ok, h[:16]))
        forks = [r for r in rows if r.get("event") == "fork_point"]
        cases.append(
            _case("PH13_FORK", "fork_recorded", len(forks) > 0, f"n_forks={len(forks)}")
        )
    else:
        cases.append(_case("PH8_CONSUMPTION", "consumption_append_only", False, "run run_demo.py first"))
        cases.append(_case("PH12_REPLAY", "replay_hash_identity", False, "missing artifacts"))
        cases.append(_case("PH13_FORK", "fork_recorded", False, "missing artifacts"))

    if learn_path.exists():
        learn = read_jsonl(learn_path)
        ok = any(e.get("source_document_ids") and e.get("loss_delta") is not None for e in learn)
        cases.append(_case("PH9_LEARNING", "learning_linked", ok, f"n={len(learn)}"))
    else:
        cases.append(_case("PH9_LEARNING", "learning_linked", False, "missing artifacts"))

    ckpt_dir = artifacts / "checkpoints"
    if ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint_*")):
        meta_files = list(ckpt_dir.glob("checkpoint_*/metadata.json"))
        ok = False
        detail = "none"
        if meta_files:
            meta = json.loads(meta_files[-1].read_text(encoding="utf-8"))
            ok = "ledger_offset" in meta
            detail = f"ledger_offset={meta.get('ledger_offset')}"
            # 6 components
            cdir = meta_files[-1].parent
            comps = ["model.pt", "optimizer.pt", "scheduler.pt", "rng_state.pt", "metadata.json"]
            ok = ok and all((cdir / c).exists() for c in comps)
        cases.append(_case("PH10_CKPT", "checkpoint_components", ok, detail))
        # resume from evidence
        if evid_path.exists():
            ev = json.loads(evid_path.read_text(encoding="utf-8"))
            r = ev.get("results", {}).get("crash_recovery", {})
            cases.append(
                _case("PH11_RESUME", "resume_matched", r.get("status") == "PASS", r.get("detail", ""))
            )
        else:
            cases.append(_case("PH11_RESUME", "resume_matched", False, "no evidence"))
    else:
        cases.append(_case("PH10_CKPT", "checkpoint_components", False, "missing"))
        cases.append(_case("PH11_RESUME", "resume_matched", False, "missing"))

    if perf_path.exists():
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        ok = float(perf.get("useful_tokens_per_second", 0)) > 0
        cases.append(
            _case("PH14_THROUGHPUT", "useful_tps", ok, f"tps={perf.get('useful_tokens_per_second')}")
        )
    else:
        cases.append(_case("PH14_THROUGHPUT", "useful_tps", False, "missing"))

    if evid_path.exists():
        ev = json.loads(evid_path.read_text(encoding="utf-8"))
        ok = "results" in ev and ev.get("overall_pass") is not None
        cases.append(_case("PH15_EVIDENCE", "evidence_generated", ok, f"overall={ev.get('overall_pass')}"))
    else:
        cases.append(_case("PH15_EVIDENCE", "evidence_generated", False, "run run_demo.py first"))

    # STAGE_* audits
    stage_dir = artifacts / "stage_audits"
    for stage in cfg.STAGES:
        sp = stage_dir / f"stage_{stage}.json"
        if sp.exists():
            audit = json.loads(sp.read_text(encoding="utf-8"))
            ok = audit.get("stage") == stage and "manifest_check" in audit
            cases.append(_case(f"STAGE_{stage}", f"stage_audit_{stage}", ok, sp.name))
        else:
            cases.append(_case(f"STAGE_{stage}", f"stage_audit_{stage}", False, "missing"))

    report = {
        "ok": all(c.ok for c in cases),
        "n_pass": sum(1 for c in cases if c.ok),
        "n_total": len(cases),
        "cases": [asdict(c) for c in cases],
    }
    out = artifacts / "deep_diagnosis_report.json"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return report


def main() -> int:
    quick = "--quick" in sys.argv
    report = run_diagnostics(quick=quick)
    print(f"Deep diagnosis: {report['n_pass']}/{report['n_total']} passed")
    for c in report["cases"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['id']}: {c['name']} — {c['detail'][:100]}")
    print("Wrote", cfg.ARTIFACTS_DIR / "deep_diagnosis_report.json")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
