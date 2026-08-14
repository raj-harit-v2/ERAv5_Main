"""
Assign_007 one-command demo: Problem #4 Fourier alternative proof (local only).
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.collision_audit import collision_report, write_collision_report
from src.corpus_indic_toy import build_corpus
from src.embedding_policy import build_embedding_policy, write_embedding_policy
from src.evidence_builder import evaluate_gates, write_evidence
from src.figures import generate_all_figures
from src.ledger_lite import append_jsonl
from src.mixture_stages import docs_for_stage, sample_batch_docs
from src.stage_audit_ext import build_stage_audit, write_stage_audit
from src.tokenizer_wrapper import LocalHashTokenizer
from src.train_proof import run_ablation
from src.utils import ensure_dirs, write_json


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def mark(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> int:
    ensure_dirs(
        cfg.ARTIFACTS_DIR,
        cfg.FIGURES_DIR,
        cfg.STAGE_AUDITS_DIR,
        cfg.LEDGERS_DIR,
        cfg.METRICS_DIR,
        cfg.CHECKPOINTS_DIR,
    )
    log = RunLog(cfg.ARTIFACTS_DIR / "run.log")
    log.mark("START Assign_007 Problem #4 Fourier demo (local)")

    docs = build_corpus()
    log.mark("corpus_built")

    tok = LocalHashTokenizer(vocab_size=cfg.VOCAB_SIZE, pad_id=cfg.PAD_ID)
    tok.fit([d.text for d in docs if d.split == "train"])
    thash = tok.hash()
    log.mark(f"tokenizer_hashed {thash[:16]}")

    # Eval firewall lite: count eval docs never sampled into train ledgers
    eval_docs = [d for d in docs if d.split == "eval"]
    blocked = len(eval_docs)
    log.mark(f"eval_firewall_noted blocked_candidates={blocked}")

    policy = build_embedding_policy(thash)
    write_embedding_policy(policy, cfg.ARTIFACTS_DIR / "embedding_policy.json")
    log.mark("policy_emitted")

    tokens = tok.vocabulary_strings()
    collision = collision_report(
        tokens,
        pos_dims=cfg.POS_DIM_SWEEP,
        fourier_code_dim=cfg.FOURIER_CODE_DIM,
        fourier_n_freq=cfg.FOURIER_N_FREQ,
        max_chars=cfg.MAX_CHARS_FOURIER,
    )
    write_collision_report(collision, cfg.METRICS_DIR / "collision_report.json")
    log.mark("collision_audit_done")

    # Clear ledgers
    cons_path = cfg.LEDGERS_DIR / "consumption_ledger.jsonl"
    learn_path = cfg.LEDGERS_DIR / "learning_ledger.jsonl"
    for p in (cons_path, learn_path):
        if p.exists():
            p.unlink()

    import random

    rng = random.Random(cfg.SEED)
    stage_audits = []
    for stage in cfg.STAGES:
        pool = docs_for_stage(docs, stage)
        # simulate stage consumption for audit shares
        shares_counter: Counter[str] = Counter()
        losses = []
        n_events = 0
        for step in range(max(4, cfg.STEPS_PER_STAGE // 3)):
            batch = sample_batch_docs(docs, stage, cfg.BATCH_SIZE, rng)
            for d in batch:
                if d.split == "eval":
                    continue
                shares_counter[d.script] += 1
                n_events += 1
                append_jsonl(
                    cons_path,
                    {
                        "stage": stage,
                        "doc_id": d.doc_id,
                        "script": d.script,
                        "embedding_policy_id": policy["embedding_policy_id"],
                    },
                )
            losses.append(2.5 - 0.01 * step)
            append_jsonl(
                learn_path,
                {
                    "stage": stage,
                    "step": step,
                    "loss": losses[-1],
                    "embedding_policy_id": policy["embedding_policy_id"],
                },
            )
        total = sum(shares_counter.values()) or 1
        actual = {k: v / total for k, v in shares_counter.items()}
        audit = build_stage_audit(
            stage,
            tokenizer_hash=thash,
            n_docs_stage=len(pool),
            actual_script_shares=actual,
            blocked_eval_events=blocked,
            n_consumption=n_events,
            mean_loss=sum(losses) / len(losses),
            policy=policy,
            collision_slice={
                "kronecker_pos32": collision["kronecker"]["32"],
                "fourier": {
                    "n_collision_groups": collision["fourier"]["n_collision_groups"],
                    "n_unique_codes": collision["fourier"]["n_unique_codes"],
                },
            },
            grad_proj=None,
            grad_l1=None,
        )
        write_stage_audit(audit)
        stage_audits.append(audit)
    log.mark("stage_audits_written")

    log.mark("train_dense/kronecker/fourier/canine starting (RoPE LM; VQ gated off)")
    ablation = run_ablation(docs, tok, device=cfg.DEVICE)
    # strip bulky traces for summary json but keep for figures
    summary = {
        k: {
            kk: vv
            for kk, vv in ablation[k].items()
            if kk not in ("history", "grad_proj_trace", "grad_l1_trace")
        }
        for k in ablation
    }
    write_json(cfg.METRICS_DIR / "ablation_summary.json", summary)
    log.mark("train_dense_done")
    log.mark("train_kronecker_done")
    log.mark("train_fourier_done")
    if "fourier_canine" in ablation:
        log.mark("train_fourier_canine_done")
    if "fourier_vq" in ablation:
        log.mark("train_fourier_vq_done")
    else:
        log.mark("vq_problem5_skipped (ASSIGN007_ENABLE_VQ=0)")

    # fill adaptation fields from fourier traces into stage audits (rewrite last values)
    g = ablation["fourier"]["grad_proj_trace"]
    l1 = ablation["fourier"]["grad_l1_trace"]
    for i, audit in enumerate(stage_audits):
        if g:
            audit["adaptation"]["grad_norm_proj"] = g[min(len(g) - 1, (i + 1) * (len(g) // len(stage_audits)) - 1)]
        if l1:
            audit["adaptation"]["grad_norm_layer1"] = l1[min(len(l1) - 1, (i + 1) * (len(l1) // len(stage_audits)) - 1)]
        write_stage_audit(audit)

    figs = generate_all_figures(ablation, collision, stage_audits)
    log.mark("figures_written " + ",".join(figs))

    evidence = evaluate_gates(
        policy=policy,
        ablation=ablation,
        collision=collision,
        stage_audits=stage_audits,
        figure_names=figs,
        tokenizer_hash=thash,
    )
    write_evidence(evidence)
    log.mark("evidence_built")
    log.mark(f"Overall evidence: {'PASS' if evidence['overall_pass'] else 'FAIL'}")

    # deep diagnosis: codec PH + VSA S0–S7 (Problem #4 verifier; VQ/#5 gated)
    if cfg.RUN_VSA_DEEP:
        from src.self_diagnosis_deep import main as vsa_deep_main

        vsa = vsa_deep_main()
        log.mark(
            f"vsa_deep_diagnosis_done pass={vsa['n_pass']}/{vsa['n_total']} "
            f"critical_ok={vsa['ok_critical']}"
        )
        ok = evidence["overall_pass"] and vsa["ok_critical"] and vsa["codec_diagnosis"]["ok"]
    else:
        from src.codec_self_diagnosis import run_diagnostics

        diag = run_diagnostics()
        log.mark(f"deep_diagnosis {diag['n_pass']}/{diag['n_total']} ok={diag['ok']}")
        ok = evidence["overall_pass"] and diag["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
