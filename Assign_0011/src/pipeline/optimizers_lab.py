"""Orchestrate Session 11 assignment artifacts into reports/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pipeline.adam_bias_ablation import write_bias_ablation_artifacts
from src.pipeline.lr_width_sweep import run_lr_width_sweep
from src.pipeline.schedule_compare import run_schedule_compare
from src.pipeline.warmup_ratio_lab import run_warmup_ratio_lab
from src.utils.adam_hand import write_adam_hand_artifacts


def run_all(reports_dir: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    reports = reports_dir or (root / "reports")
    reports.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    results["task1_adam_hand"] = write_adam_hand_artifacts(reports)
    results["task2_bias"] = write_bias_ablation_artifacts(reports)
    results["task3_ratio"] = run_warmup_ratio_lab(reports)
    results["task4_schedules"] = run_schedule_compare(reports)
    results["task5_lr_sweep"] = run_lr_width_sweep(reports)

    summary_path = reports / "Assgn011_Summary.md"
    _write_summary(summary_path, results)
    results["summary_path"] = str(summary_path)

    manifest = reports / "assgn011_manifest.json"
    portable = _relativize_paths(results, root)
    manifest.write_text(json.dumps(portable, indent=2, default=str), encoding="utf-8")
    results["manifest_path"] = str(manifest)

    results["all_ok"] = _verification_gate(results, root)
    return results


def _relativize_paths(obj: Any, root: Path) -> Any:
    if isinstance(obj, dict):
        return {k: _relativize_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relativize_paths(v, root) for v in obj]
    if isinstance(obj, Path):
        try:
            return obj.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return str(obj)
    if isinstance(obj, str):
        try:
            p = Path(obj)
            if p.is_absolute():
                return p.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
        return obj
    return obj


def _resolve(root: Path, p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _verification_gate(results: dict[str, Any], root: Path) -> bool:
    checks = [
        results["task1_adam_hand"].get("ok"),
        results["task2_bias"].get("ok"),
        results["task3_ratio"].get("ok"),
        results["task4_schedules"].get("ok"),
        results["task5_lr_sweep"].get("ok"),
    ]
    primary = [
        _resolve(root, results["task2_bias"].get("png_path")),
        _resolve(root, results["task5_lr_sweep"].get("png_path")),
    ]
    plots_ok = all(p is not None and p.is_file() and p.stat().st_size > 0 for p in primary)
    return all(bool(c) for c in checks) and plots_ok


def _write_summary(path: Path, results: dict[str, Any]) -> None:
    t1 = results["task1_adam_hand"]
    t2 = results["task2_bias"]
    t3 = results["task3_ratio"]
    t4 = results["task4_schedules"]
    t5 = results["task5_lr_sweep"]

    lines = [
        "# Assgn011 Summary — Session 11 Optimizers and Schedules",
        "",
        "## 1. Adam bias correction at t=1",
        "",
        "Without bias correction, a constant-sign first gradient takes a step of about "
        f"{t2.get('measured_ratio_t1', 3.16):.3f} × η (theory "
        f"{t2.get('theory_ratio_t1', 3.162):.3f}). With correction the same step is ≈ 1.00 × η. "
        "That early overshoot is why warmup exists.",
        "",
        "## 2. Primary plots",
        "",
        f"- Bias ablation (20 steps): `{Path(t2.get('png_path', '')).name}`",
        f"- LR width sweep (three minima): `{Path(t5.get('png_path', '')).name}`",
        "",
        "## 3. Schedule keep-decision (plan 300, checkpoint 200)",
        "",
        f"Keep **{t4.get('keep')}**. Loss@200 — cosine: {t4.get('cosine_loss_200')}, "
        f"WSD: {t4.get('wsd_loss_200')}. Planned horizon is 300 steps; the keep-decision "
        "is the early-stop checkpoint at 200 (FULL §10).",
        "",
        "## 4. η at width 4096 + confidence",
        "",
        f"- SP extrapolation: η_4096 ≈ {t5.get('eta_4096_sp')} — confidence **LOW** "
        "(FULL Section 12: up to 16× overstatement risk under SP when transferring small→large width).",
        "",
        "## 5. Tune both sides",
        "",
        "Tasks 4 and 5 used matched step budgets, seeds, peak LRs, and warmup lengths across arms.",
        "",
        "## 6. Long-context / batch invariant",
        "",
        "Inherited from Session 10: when scaling context length, keep **global batch token count "
        "fixed per phase**. Longer sequences reduce micro-batch rows and raise accumulation steps; "
        "do not silently retune η mid-curriculum.",
        "",
        "## 7. Gate snapshot",
        "",
        "| Task | OK |",
        "| :--- | :---: |",
        f"| 1 Hand Adam | {t1.get('ok')} |",
        f"| 2 Bias plot | {t2.get('ok')} |",
        f"| 3 Ratio / warmup | {t3.get('ok')} |",
        f"| 4 Cosine vs WSD | {t4.get('ok')} |",
        f"| 5 LR sweep | {t5.get('ok')} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
