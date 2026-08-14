"""
Deep VSA/Fourier diagnosis for Assign_007 Problem #4.

Runs staged HRR+Fourier suites (S0–S7), codec smoke checks, writes JSON + MD analysis + PNGs.
Problem ID stays 4. VQ/#5 comparison gated off by default.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config as cfg
from src.codec_self_diagnosis import run_diagnostics as run_codec_diagnostics
from src.fourier_vsa_bridge import FourierVSABridge
from src.fourier_vsa_tests import run_all_stages
from src.hrr import bind, cosine, dropout_noise, unbind
from src.utils import ensure_dirs, write_json
from src.vsa_symbols import DIGITS, OPS, ROLES


def _fig_capacity(meta: dict) -> str:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curves = meta.get("curves", {})
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, ys in curves.items():
        ax.plot(range(1, len(ys) + 1), ys, label=f"D={label}")
    ax.axhline(cfg.VSA_CAPACITY_TARGET_COS, color="#dc2626", linestyle="--", label="target cos")
    ax.set_xlabel("bundled pairs k")
    ax.set_ylabel("retrieval cosine")
    ax.set_title("VSA capacity / interference curve")
    ax.legend()
    path = cfg.FIGURES_DIR / "vsa_capacity_curve.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_math_heatmap() -> str:
    bridge = FourierVSABridge()
    exprs = {
        "8+8": ("ADD", "16"),
        "8*8": ("MUL", "64"),
        "8/8": ("DIV", "1"),
    }
    results = ["16", "64", "1"]
    mat = []
    for name, (op, res) in exprs.items():
        row_role = bridge.from_spec(ROLES["OP"], prefer_fourier_text=False)
        res_role = bridge.from_spec(ROLES["RESULT"], prefer_fourier_text=False)
        mem = (
            bind(bridge.from_spec(ROLES["LEFT"], prefer_fourier_text=False), bridge.from_spec(DIGITS["8"], prefer_fourier_text=False))
            + bind(bridge.from_spec(ROLES["RIGHT"], prefer_fourier_text=False), bridge.from_spec(DIGITS["8"], prefer_fourier_text=False))
            + bind(row_role, bridge.from_spec(OPS[op], prefer_fourier_text=False))
            + bind(res_role, bridge.from_spec(DIGITS[res], prefer_fourier_text=False))
        )
        ret = unbind(mem, res_role)
        mat.append([cosine(ret, bridge.from_spec(DIGITS[r], prefer_fourier_text=False)) for r in results])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results)
    ax.set_yticks(range(len(exprs)))
    ax.set_yticklabels(list(exprs.keys()))
    ax.set_xlabel("candidate RESULT digit")
    ax.set_ylabel("expression")
    ax.set_title("VSA math role heatmap (RESULT unbind)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = cfg.FIGURES_DIR / "vsa_math_role_heatmap.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _fig_noise() -> str:
    bridge = FourierVSABridge()
    d = bridge.hrr_dim
    from src.hrr import random_hv

    a, b = random_hv(d, seed=9), random_hv(d, seed=10)
    mem = bind(a, b)
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ys = []
    for f in fracs:
        noisy = dropout_noise(mem, f, seed=1)
        ys.append(cosine(unbind(noisy, a), b))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(fracs, ys, marker="o", color="#0891b2")
    ax.set_xlabel("dropout fraction")
    ax.set_ylabel("retrieval cosine")
    ax.set_title("VSA holographic noise robustness")
    path = cfg.FIGURES_DIR / "vsa_noise_robustness.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _write_analysis_md(report: dict, fig_names: list[str], codec_report: dict) -> Path:
    by_stage: dict[str, list] = defaultdict(list)
    for c in report["cases"]:
        by_stage[c["stage"]].append(c)

    lines = [
        "# Deep VSA / Fourier Diagnosis Analysis (Problem #4)",
        "",
        f"**Overall critical OK:** {report['ok_critical']}",
        f"**Cases:** {report['n_pass']}/{report['n_total']} passed "
        f"(critical fails={report['n_critical_fail']}, soft fails={report['n_soft_fail']})",
        f"**HRR_DIM:** {report['hrr_dim']}",
        f"**VQ/#5 comparison:** {report['vq_comparison']}",
        f"**High bit-rate media:** {report['high_bitrate_tests']}",
        "",
        "## Honesty note",
        "",
        report.get("math_note", ""),
        "",
        "This layer verifies FFT circular convolution (HRR) **and** that Problem #4 Fourier "
        "codes lift into that algebra. It does **not** claim Problem #1 (numeric closure in embeddings) "
        "or Problem #5 (invertible LM head).",
        "",
        "## Codec diagnosis (existing)",
        "",
        f"- Codec `src.codec_self_diagnosis`: {codec_report['n_pass']}/{codec_report['n_total']} "
        f"ok={codec_report['ok']}",
        "",
        "## Stage results",
        "",
    ]
    for stage in sorted(by_stage.keys()):
        rows = by_stage[stage]
        n_ok = sum(1 for r in rows if r["pass"])
        lines.append(f"### {stage} ({n_ok}/{len(rows)})")
        lines.append("")
        lines.append("| Case | Pass | Neg | Metric | Detail |")
        lines.append("|------|------|-----|--------|--------|")
        for r in rows:
            m = "" if r.get("metric") is None else f"{r['metric']:.4f}"
            lines.append(
                f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | "
                f"{'Y' if r.get('negative') else ''} | {m} | {r['detail']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Negative controls — what they catch",
            "",
            "- **Zero bind:** NaN/collapse guards on empty frequency content.",
            "- **Wrong role / wrong op:** ensures retrieval is not always returning a popular vector.",
            "- **Anagram / Abugida truncate:** codec sensitivity before HRR.",
            "- **Kronecker@32 diagnostic:** records Indic byte-window risk (baseline, not Fourier).",
            "- **Dropout 30%:** holographic robustness of bound memory.",
            "- **Capacity sweep:** interference limit vs dimension.",
            "",
            "## Figures",
            "",
        ]
    )
    for n in fig_names:
        lines.append(f"- `{n}`")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- `src/hrr.py`, `src/fourier_vsa_tests.py`, `src/codec_self_diagnosis.py`",
            "- Session / NotebookLM notes kept outside the graded tree (personal archive).",
            "",
        ]
    )
    path = cfg.ARTIFACTS_DIR / "deep_diagnosis_vsa_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> dict:
    ensure_dirs(cfg.ARTIFACTS_DIR, cfg.FIGURES_DIR, cfg.METRICS_DIR)
    print("Running codec diagnosis...")
    codec_report = run_codec_diagnostics()
    print("Running VSA stages S0-S7...")
    report = run_all_stages()
    figs = [
        _fig_capacity(report.get("capacity_meta") or {}),
        _fig_math_heatmap(),
        _fig_noise(),
    ]
    report["figures"] = figs
    report["codec_diagnosis"] = {
        "n_pass": codec_report["n_pass"],
        "n_total": codec_report["n_total"],
        "ok": codec_report["ok"],
    }
    write_json(cfg.ARTIFACTS_DIR / "deep_diagnosis_vsa_report.json", report)
    _write_analysis_md(report, figs, codec_report)
    print(
        f"VSA deep diagnosis: {report['n_pass']}/{report['n_total']} "
        f"critical_ok={report['ok_critical']} figures={figs}"
    )
    print("Wrote", cfg.ARTIFACTS_DIR / "deep_diagnosis_vsa_report.json")
    print("Wrote", cfg.ARTIFACTS_DIR / "deep_diagnosis_vsa_analysis.md")
    return report


if __name__ == "__main__":
    r = main()
    # Fail only on critical VSA failures (codec already printed separately)
    raise SystemExit(0 if r["ok_critical"] else 1)
