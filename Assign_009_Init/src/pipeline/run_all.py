"""CLI: run Part 1 harness → Part 2 MTP → Part 3 Curriculum plots → widget export."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from src.pipeline.export_widget import run_export
from src.pipeline.harness import run_harness
from src.pipeline.plot_stats import run_plots
from src.pipeline.train_smoke import run_mtp_smoke

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "evaluation"
WRITEUP = EVAL / "session09_writeup.txt"
PART2_JSON = EVAL / "part2_losses.json"


def _load_part2_from_disk() -> dict[str, Any] | None:
    """Reuse existing Part 2 artifact (e.g. Colab) when MTP training is skipped."""
    if not PART2_JSON.is_file():
        return None
    data = json.loads(PART2_JSON.read_text(encoding="utf-8"))
    if "L1_final" not in data or "L2_final" not in data or "sum_final" not in data:
        return None
    return data


def _argv_for_argparse(argv: list[str] | None) -> list[str]:
    """Return argv safe for argparse (strip Jupyter/ipykernel ``-f kernel.json``)."""
    if argv is not None:
        return list(argv)
    raw = sys.argv[1:]
    out: list[str] = []
    i = 0
    while i < len(raw):
        # Jupyter injects: -f C:\\...\\kernel-....json
        if raw[i] == "-f" and i + 1 < len(raw) and raw[i + 1].endswith(".json"):
            i += 2
            continue
        out.append(raw[i])
        i += 1
    return out


def _run_build_dist() -> None:
    """Load tests/build_dist.py by path (avoid fragile ``import tests`` on Colab)."""
    path = ROOT / "tests" / "build_dist.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. On Colab, zip must include tests/build_dist.py"
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("s9_build_dist", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load build_dist from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Session 9 full graded pipeline")
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-widget", action="store_true")
    parser.add_argument("--skip-dist", action="store_true")
    parser.add_argument("--mtp-steps", type=int, default=40)
    args = parser.parse_args(_argv_for_argparse(argv))

    print("=" * 60)
    print("PART 1 — Observable harness")
    print("=" * 60)
    seven = run_harness()

    part2: dict[str, Any] | None = None
    if not args.skip_mtp:
        print("=" * 60)
        print("PART 2 — MTP k=2 smoke")
        print("=" * 60)
        part2 = run_mtp_smoke(steps=args.mtp_steps)
    else:
        part2 = _load_part2_from_disk()
        if part2 is not None:
            print(
                f"PART 2 — skipped training; using {PART2_JSON.relative_to(ROOT)} "
                f"(L1={part2['L1_final']:.4f} L2={part2['L2_final']:.4f})"
            )
        else:
            print(f"PART 2 — skipped; no usable {PART2_JSON.name} on disk")

    plots = None
    if not args.skip_plots:
        print("=" * 60)
        print("PART 3 — Curriculum_Required_stats")
        print("=" * 60)
        plots = run_plots()

    sum_before = float(seven.get("boundary_loss_sum_before", 0.0))
    sum_after = float(seven.get("boundary_loss_sum_after", 0.0))
    sum_delta = sum_after - sum_before
    lines = [
        "## Session 9 numbers",
        "Part1:",
        f"  shapes_ok: {seven.get('shapes_ok')}",
        f"  shift_strings_ok: {seven.get('shift_strings_ok')}",
        f"  pad_count_before/after: {seven.get('pad_count_before')}/{seven.get('pad_count_after')}",
        f"  boundary contrib: {seven.get('boundary_contrib_before')}->{seven.get('boundary_contrib_after')}  "
        f"sum: {sum_before:.3f}->{sum_after:.3f} (d={sum_delta:.3f})",
        "  boundary_note: masked cross-doc next-token; sum drops, mean stays ~ln(V) at init",
        f"  ppl0 / loss0: {seven.get('ppl0'):.2f} / {seven.get('loss0'):.4f}",
        f"  tied_params / untied_params: {seven.get('tied_params')} / {seven.get('untied_params')}",
        f"  peak_full / peak_chunked / ratio: {seven.get('peak_full_mib'):.4f} / "
        f"{seven.get('peak_chunked_mib'):.4f} / {seven.get('ratio_full_over_chunked'):.4f}",
    ]
    if part2:
        note = part2.get("explanation") or (
            "L2 (t+2) typically stays higher / falls slower than L1 (t+1) "
            "from the same trunk state."
        )
        lines += [
            "Part2:",
            f"  L1 / L2 / sum: {part2['L1_final']:.4f} / {part2['L2_final']:.4f} / {part2['sum_final']:.4f}",
            f"  L2_vs_L1_note: {note}",
        ]
    if plots:
        lines += [
            "Part3:",
            "  Part 3 = Curriculum_Required_stats long-context demonstration.",
            f"  peak_vram: {plots['peak_vram']}",
            f"  chunk_frontier: {plots['chunk_frontier']}",
            f"  swiglu: {plots['swiglu']}",
        ]

    if not args.skip_widget:
        print("=" * 60)
        print("WIDGET — Export browser weights")
        print("=" * 60)
        run_export()
        lines.append("Widget: data/evaluation/widget_weights.json")

    if not args.skip_dist and not args.skip_widget:
        print("=" * 60)
        print("DIST — Build static site")
        print("=" * 60)
        _run_build_dist()
        lines.append("Dist: dist/index.html")

    lines.append(
        "Long_context takeaway: full CE scales with T; chunked flattens peak VRAM so 64K+ is feasible."
    )
    text = "\n".join(lines) + "\n"
    WRITEUP.parent.mkdir(parents=True, exist_ok=True)
    WRITEUP.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Wrote {WRITEUP}")


if __name__ == "__main__":
    main()
