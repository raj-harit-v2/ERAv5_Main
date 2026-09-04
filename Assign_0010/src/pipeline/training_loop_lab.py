"""Orchestrate Session 10 assignment artifacts into reports/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pipeline.accumulation_bug import plot_accumulation_bug
from src.pipeline.grad_norm_track import run_grad_norm_track
from src.pipeline.mfu_report import measure_mfu, write_mfu_report
from src.pipeline.quantize_compare import run_quantize_compare
from src.utils.gradient_check import verify_two_weight_chain
from src.utils.precision_bits import write_precision_report
from src.utils.shape_hooks import run_tensor_shapes


def run_all(reports_dir: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    reports = reports_dir or (root / "reports")
    reports.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    results["task1_shapes"] = run_tensor_shapes(reports / "assgn010_tensor_shapes.log")
    results["task2_gradient"] = verify_two_weight_chain(reports / "assgn010_gradient_check.txt")
    results["task3_accumulation"] = plot_accumulation_bug(reports / "assgn010_accumulation_bug.png")
    results["task4_grad_norm"] = run_grad_norm_track(
        reports / "assgn010_step_metrics.csv",
        reports / "assgn010_gradnorm_vs_loss.png",
    )
    mfu_stats = measure_mfu()
    results["task5_mfu"] = write_mfu_report(reports / "assgn010_mfu_report.md", mfu_stats)
    results["task6_precision"] = write_precision_report(reports / "assgn010_precision_bits.md")
    results["task6_fp_tables"] = run_quantize_compare(reports)

    summary_path = reports / "Assgn010_Summary.md"
    _write_summary(summary_path, results)
    results["summary_path"] = str(summary_path.relative_to(root).as_posix())

    manifest = reports / "assgn010_manifest.json"
    portable = _relativize_paths(results, root)
    manifest.write_text(json.dumps(portable, indent=2, default=str), encoding="utf-8")
    results["manifest_path"] = str(manifest.relative_to(root).as_posix())

    results["all_ok"] = _verification_gate(results, root)
    return results


def _relativize_paths(obj: Any, root: Path) -> Any:
    """Rewrite absolute paths under *root* to posix-relative strings for the manifest."""
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


def _verification_gate(results: dict[str, Any], root: Path) -> bool:
    checks = [
        results["task1_shapes"].get("ok"),
        results["task2_gradient"].get("passed"),
        results["task3_accumulation"].get("ok"),
        results["task4_grad_norm"].get("ok"),
        results["task5_mfu"].get("ok"),
        bool(results["task6_precision"].get("path")),
        results["task6_fp_tables"].get("ok"),
    ]

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (root / path)

    plots = [
        _resolve(results["task3_accumulation"]["path"]),
        _resolve(results["task4_grad_norm"]["png_path"]),
    ]
    plots_ok = all(p.is_file() and p.stat().st_size > 0 for p in plots)
    return all(checks) and plots_ok


def _write_summary(path: Path, results: dict[str, Any]) -> None:
    t3 = results["task3_accumulation"]
    t5 = results["task5_mfu"]
    lines = [
        "# Assgn010 Summary — Session 10 Training Loop",
        "",
        "## What silent bugs did we prove?",
        "",
        "We reproduced the **15.4% gradient-accumulation averaging bug** (token-weighted 2.6000 vs "
        "average-of-averages 3.0000) and demonstrated that **grad norm can move before loss** after "
        "injecting a poison step — matching the Session 10 thesis that serious bugs stay silent while "
        "loss still looks plausible. MFU reporting shows how identical loss curves can hide massive "
        "hardware under-utilization.",
        "",
        "## Primary plots",
        "",
        f"- [Accumulation bug](assgn010_accumulation_bug.png) — discrepancy {t3['discrepancy_pct']:.2f}%",
        f"- [Grad norm vs loss](assgn010_gradnorm_vs_loss.png) — flagged step {results['task4_grad_norm']['flagged_step']}",
        "",
        "## MFU table",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| MFU % | {t5['mfu_pct']:.2f}% |",
        f"| Achieved TFLOP/s | {t5['achieved_tflops']:.4f} |",
        f"| Peak TFLOP/s | {t5['peak_tflops']:.1f} |",
        f"| Gap to 40% target | {t5['gap_to_target']:.2f} pp |",
        "",
        "## BF16 recommendation",
        "",
        "Train in **BF16** with FP32 master weights (16 bytes/param Adam state). See "
        "[assgn010_precision_bits.md](assgn010_precision_bits.md) and the ten-format table in "
        "[assgn010_fp_format_table.md](assgn010_fp_format_table.md). MXFP4 outlier rescue: "
        "[assgn010_mbs_oas_table.md](assgn010_mbs_oas_table.md).",
        "",
        "## Long-context note",
        "",
        "When curriculum moves 4K → 8K → … → 1M, **do not change global batch "
        "token semantics mid-phase** (ERA V4 mistake). Longer context increases activation memory and "
        "accumulation steps — retune MFU and checkpointing per phase.",
        "",
        "## Artifacts",
        "",
        "- [assgn010_tensor_shapes.log](assgn010_tensor_shapes.log)",
        "- [assgn010_gradient_check.txt](assgn010_gradient_check.txt)",
        "- [assgn010_step_metrics.csv](assgn010_step_metrics.csv)",
        "- [assgn010_mfu_report.md](assgn010_mfu_report.md)",
        "- [assgn010_precision_bits.md](assgn010_precision_bits.md)",
        "- [assgn010_fp_format_table.md](assgn010_fp_format_table.md)",
        "- [assgn010_mbs_oas_table.md](assgn010_mbs_oas_table.md)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
