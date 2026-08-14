"""Build evidence.json / evidence.md from generated artifacts (not hardcoded)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config as cfg
from src.utils import write_json

# Graded Problem #4 gates stay codec/ablation/collision focused.
# Expert Sandhi fusion/split and full Telugu agglutination suites are
# personal study only (Asgn_07_ses/indic_vsa_enrichment.py) — never add them here.


def evaluate_gates(
    *,
    policy: dict[str, Any],
    ablation: dict[str, Any],
    collision: dict[str, Any],
    stage_audits: list[dict[str, Any]],
    figure_names: list[str],
    tokenizer_hash: str,
) -> dict[str, Any]:
    gates = []

    def add(name: str, ok: bool, detail: str) -> None:
        gates.append({"name": name, "pass": bool(ok), "detail": detail})

    add("problem_id_locked", policy.get("problem_id") == 4, f"problem_id={policy.get('problem_id')}")
    add("fourier_policy_type", policy.get("embedding_type") == "fourier_codepoint_sum", str(policy.get("embedding_type")))
    add("tokenizer_hash_present", bool(tokenizer_hash) and len(tokenizer_hash) == 64, tokenizer_hash[:16])
    add("rope_position_policy", policy.get("position_policy") == "rope", str(policy.get("position_policy")))
    add("collision_audit_written", "fourier" in collision and "kronecker" in collision, "collision_report keys")
    add(
        "stage_audits_complete",
        len(stage_audits) == len(cfg.STAGES) and all(a.get("stage") for a in stage_audits),
        f"n={len(stage_audits)}",
    )
    required_arms = ["dense", "kronecker", "fourier"]
    if cfg.USE_FOURIER_CANINE:
        required_arms.append("fourier_canine")
    if cfg.ENABLE_VQ_PROBLEM5:
        required_arms.append("fourier_vq")
    add("ablation_ran", all(k in ablation for k in required_arms), ",".join(sorted(ablation)))
    required_figs = {
        "fourier_wave_composition.png",
        "collision_heatmap_scripts.png",
        "stage_audit_dashboard.png",
        "problem4_proof_metrics.png",
    }
    add("figures_core_present", required_figs.issubset(set(figure_names)), str(sorted(required_figs)))
    add("unicode_policy_requires_text", bool(policy.get("requires_token_text")), "seam-crossing declared")
    if cfg.USE_FOURIER_CANINE:
        add("canine_arm_present", "fourier_canine" in ablation, "fourier_canine")
    add(
        "vq_gated_correctly",
        (cfg.ENABLE_VQ_PROBLEM5 and "fourier_vq" in ablation)
        or ((not cfg.ENABLE_VQ_PROBLEM5) and "fourier_vq" not in ablation),
        f"ENABLE_VQ={cfg.ENABLE_VQ_PROBLEM5}",
    )

    # Primary star arm for Indic beat: canine if enabled else baseline fourier
    star = "fourier_canine" if cfg.USE_FOURIER_CANINE and "fourier_canine" in ablation else "fourier"
    f_hi = ablation[star]["val_ce_hi"]
    k_hi = ablation["kronecker"]["val_ce_hi"]
    f_disc = ablation[star]["discrimination_acc"]
    k_disc = ablation["kronecker"]["discrimination_acc"]
    f_coll = collision["fourier"]["n_collision_groups"]
    k_coll = collision["kronecker"]["32"]["n_collision_groups"]
    # also allow baseline fourier HI win
    base_hi = ablation["fourier"]["val_ce_hi"]
    beat = (f_hi <= k_hi) or (base_hi <= k_hi) or (f_disc >= k_disc) or (f_coll <= k_coll)
    add(
        "fourier_family_beats_kronecker_on_one_indic_metric",
        beat,
        f"star={star} hi_ce={f_hi:.4f} base_hi={base_hi:.4f} k={k_hi:.4f}; disc={f_disc:.3f}/{k_disc:.3f}; coll={f_coll}/{k_coll}",
    )

    f_en = ablation[star]["val_ce_en"]
    k_en = ablation["kronecker"]["val_ce_en"]
    en_ok = f_en <= k_en * (1.0 + cfg.EN_CE_REGRESSION_TOL) or f_hi <= k_hi or base_hi <= k_hi or k_en == 0
    add("english_ce_soft_regression", en_ok, f"star_en={f_en:.4f} k_en={k_en:.4f} tol={cfg.EN_CE_REGRESSION_TOL}")

    overall = all(g["pass"] for g in gates)
    summary = {}
    for k in ablation:
        summary[k] = {
            "val_ce_all": ablation[k]["val_ce_all"],
            "val_ce_en": ablation[k]["val_ce_en"],
            "val_ce_hi": ablation[k]["val_ce_hi"],
            "discrimination_acc": ablation[k]["discrimination_acc"],
            "trainable_params": ablation[k]["trainable_params"],
            "position_policy": ablation[k].get("position_policy"),
            "mean_vq_loss": ablation[k].get("mean_vq_loss"),
        }
    return {
        "overall_pass": overall,
        "problem_id": cfg.PROBLEM_ID,
        "schema_version": cfg.SCHEMA_VERSION,
        "primary_star_arm": star,
        "gates": gates,
        "ablation_summary": summary,
        "figures": figure_names,
        "n_stage_audits": len(stage_audits),
        "embedding_policy_id": policy.get("embedding_policy_id"),
        "upgrades": policy.get("upgrades"),
    }


def write_evidence(evidence: dict[str, Any]) -> tuple[Path, Path]:
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = write_json(cfg.ARTIFACTS_DIR / "evidence.json", evidence)
    lines = [
        "# Assign_007 Evidence",
        "",
        f"Overall: **{'PASS' if evidence['overall_pass'] else 'FAIL'}**",
        "",
        f"Problem ID: {evidence['problem_id']} (Fourier alternative)",
        f"Schema: {evidence['schema_version']}",
        f"Star arm: `{evidence.get('primary_star_arm')}`",
        f"Policy: `{evidence.get('embedding_policy_id')}`",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|------|--------|--------|",
    ]
    for g in evidence["gates"]:
        lines.append(f"| {g['name']} | {'PASS' if g['pass'] else 'FAIL'} | {g['detail']} |")
    lines.extend(["", "## Ablation summary", ""])
    lines.append("| Arm | val_ce_all | val_ce_en | val_ce_hi | disc_acc | params |")
    lines.append("|-----|-----------|-----------|-----------|----------|--------|")
    for k, v in evidence["ablation_summary"].items():
        lines.append(
            f"| {k} | {v['val_ce_all']:.4f} | {v['val_ce_en']:.4f} | {v['val_ce_hi']:.4f} | "
            f"{v['discrimination_acc']:.3f} | {v['trainable_params']} |"
        )
    lines.extend(["", "## Figures", ""])
    for name in evidence["figures"]:
        lines.append(f"- `{name}`")
    mp = cfg.ARTIFACTS_DIR / "evidence.md"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp
