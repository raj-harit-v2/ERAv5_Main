"""Codec / artifact self-diagnosis for Assign_007 (local hermetic checks)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.embeddings import KroneckerByteEmbedding, build_embedding
from src.embeddings.canine_vq import FourierCanineEmbedding
from src.embeddings.fourier_baseline import FourierCodepointEmbedding, znormalize
from src.rope import apply_rope, build_rope_cache
from src.utils import read_json, write_json


def _case(phase: str, name: str, ok: bool, detail: str) -> dict:
    return {"phase": phase, "name": name, "pass": bool(ok), "detail": detail}


def run_diagnostics() -> dict:
    cases: list[dict] = []

    # PH0 env / local
    cases.append(_case("PH0", "problem_id_is_4", cfg.PROBLEM_ID == 4, str(cfg.PROBLEM_ID)))
    cases.append(_case("PH0", "use_hf_default_off", cfg.USE_HF is False, str(cfg.USE_HF)))
    cases.append(_case("PH0", "artifacts_dir_local", cfg.ARTIFACTS_DIR.is_relative_to(cfg.ROOT) if hasattr(Path, "is_relative_to") else str(cfg.ARTIFACTS_DIR).startswith(str(cfg.ROOT)), str(cfg.ARTIFACTS_DIR)))

    cases.append(_case("PH0", "rope_default", cfg.POSITION_POLICY == "rope", cfg.POSITION_POLICY))
    cases.append(_case("PH0", "vq_gated_off_by_default", cfg.ENABLE_VQ_PROBLEM5 is False, str(cfg.ENABLE_VQ_PROBLEM5)))
    cases.append(_case("PH0", "canine_enabled", cfg.USE_FOURIER_CANINE is True, str(cfg.USE_FOURIER_CANINE)))

    # PH1 codec determinism
    f = FourierCodepointEmbedding(d_model=8, code_dim=cfg.FOURIER_CODE_DIM, n_freq=cfg.FOURIER_N_FREQ)
    a = f.encode_string("भारत")
    b = f.encode_string("भारत")
    cases.append(_case("PH1", "fourier_determinism", bool((a - b).abs().max() < 1e-6), "allclose"))

    # PH1 anagram
    c1 = f.encode_string("अब")
    c2 = f.encode_string("बा")
    cases.append(_case("PH1", "fourier_anagram_differs", bool((c1 - c2).abs().sum() > 1e-4), "phase sensitivity"))

    # PH2 kronecker / fourier param independence from V
    e1 = build_embedding(
        kind="fourier",
        vocab_size=128,
        d_model=cfg.D_MODEL,
        pad_id=0,
        pos_dim=32,
        fourier_code_dim=cfg.FOURIER_CODE_DIM,
        fourier_n_freq=cfg.FOURIER_N_FREQ,
        max_chars=cfg.MAX_CHARS_FOURIER,
    )
    e2 = build_embedding(
        kind="fourier",
        vocab_size=1024,
        d_model=cfg.D_MODEL,
        pad_id=0,
        pos_dim=32,
        fourier_code_dim=cfg.FOURIER_CODE_DIM,
        fourier_n_freq=cfg.FOURIER_N_FREQ,
        max_chars=cfg.MAX_CHARS_FOURIER,
    )
    n1 = sum(p.numel() for p in e1.parameters())
    n2 = sum(p.numel() for p in e2.parameters())
    cases.append(_case("PH2", "fourier_params_independent_of_V", n1 == n2, f"{n1} vs {n2}"))

    k = KroneckerByteEmbedding(d_model=cfg.D_MODEL, pos_dim=cfg.POS_DIM_KRON)
    nk = sum(p.numel() for p in k.parameters())
    expected = (256 * cfg.POS_DIM_KRON) * cfg.D_MODEL + cfg.D_MODEL  # Linear weight+bias
    cases.append(_case("PH2", "kronecker_proj_shape", nk == expected, f"{nk}=={expected}"))

    # PH3 composition structure
    waves = [f.wave(ord(ch), i) for i, ch in enumerate("राम")]
    import torch

    summed = torch.stack(waves).sum(0) / math.sqrt(len(waves))
    composed = znormalize(summed)
    direct = f.encode_string("राम")
    cases.append(
        _case(
            "PH3",
            "fourier_composition_matches_sum",
            bool((composed - direct).abs().max() < 1e-5),
            "sum+znorm",
        )
    )

    # PH3b RoPE + CANINE smoke
    cos, sin = build_rope_cache(8, 16, torch.device("cpu"))
    q = torch.randn(1, 2, 8, 16)
    q2 = apply_rope(q, cos, sin)
    cases.append(_case("PH3b", "rope_apply_shape", q2.shape == q.shape, str(tuple(q2.shape))))
    canine = FourierCanineEmbedding(d_model=32, code_dim=64, n_freq=8, max_chars=16, stride=2)
    c_a = canine.encode_string("भारत")
    c_b = canine.encode_string("भारत")
    cases.append(_case("PH3b", "canine_determinism", bool((c_a - c_b).abs().max() < 1e-5), "allclose"))
    cases.append(
        _case(
            "PH3b",
            "canine_anagram_differs",
            bool((canine.encode_string("अब") - canine.encode_string("बा")).abs().sum() > 1e-4),
            "phase+conv",
        )
    )

    # PH4 artifacts
    art = cfg.ARTIFACTS_DIR
    for name in (
        "evidence.json",
        "evidence.md",
        "run.log",
        "embedding_policy.json",
        "metrics/collision_report.json",
        "metrics/ablation_summary.json",
    ):
        p = art / name
        cases.append(_case("PH4", f"artifact_{name.replace('/', '_')}", p.exists(), str(p)))

    for stage in cfg.STAGES:
        p = art / "stage_audits" / f"stage_{stage}.json"
        cases.append(_case("PH5", f"stage_audit_{stage}", p.exists(), p.name))

    for fig in (
        "fourier_wave_composition.png",
        "collision_heatmap_scripts.png",
        "stage_audit_dashboard.png",
        "problem4_proof_metrics.png",
        "param_memory_budget.png",
        "v5_param_budget_reference.png",
        "train_val_loss_curves.png",
        "frozen_shift_grad_spike.png",
        "pos_dim_truncation_bars.png",
        "kronecker_byte_grid_example.png",
    ):
        p = art / "figures" / fig
        cases.append(_case("PH6", f"fig_{fig}", p.exists() and p.stat().st_size > 100, fig))

    # PH7 evidence honesty
    if (art / "evidence.json").exists():
        ev = read_json(art / "evidence.json")
        cases.append(_case("PH7", "evidence_has_gates", isinstance(ev.get("gates"), list) and len(ev["gates"]) > 0, str(len(ev.get("gates", [])))))
        cases.append(_case("PH7", "evidence_problem_4", ev.get("problem_id") == 4, str(ev.get("problem_id"))))
        # not a trivial always-true hardcoded overall without ablation
        cases.append(
            _case(
                "PH7",
                "evidence_has_ablation_summary",
                "fourier" in (ev.get("ablation_summary") or {}),
                "ablation_summary",
            )
        )

    if (art / "embedding_policy.json").exists():
        pol = read_json(art / "embedding_policy.json")
        cases.append(_case("PH8", "policy_schema_problem_id", pol.get("problem_id") == 4, str(pol.get("problem_id"))))
        cases.append(_case("PH8", "policy_has_tokenizer_hash", isinstance(pol.get("tokenizer_hash"), str) and len(pol["tokenizer_hash"]) == 64, "hash"))

    if (art / "metrics/ablation_summary.json").exists() or (art / "metrics" / "ablation_summary.json").exists():
        ab = read_json(art / "metrics" / "ablation_summary.json")
        cases.append(_case("PH9", "ablation_has_fourier", "fourier" in ab, ",".join(sorted(ab))))
        if cfg.USE_FOURIER_CANINE:
            cases.append(_case("PH9", "ablation_has_canine", "fourier_canine" in ab, ",".join(sorted(ab))))
        cases.append(_case("PH9", "vq_absent_when_gated", ("fourier_vq" in ab) == cfg.ENABLE_VQ_PROBLEM5, str(cfg.ENABLE_VQ_PROBLEM5)))

    n_pass = sum(1 for c in cases if c["pass"])
    report = {
        "n_pass": n_pass,
        "n_total": len(cases),
        "ok": n_pass == len(cases),
        "cases": cases,
    }
    write_json(cfg.ARTIFACTS_DIR / "deep_diagnosis_report.json", report)
    print(f"Deep diagnosis: {n_pass}/{len(cases)} passed ok={report['ok']}")
    return report


if __name__ == "__main__":
    r = run_diagnostics()
    raise SystemExit(0 if r["ok"] else 1)
