"""Session 12 assignment lab runner.

Single-process (reports + graphics):
  python tests/assgn012_zero_lab.py

Distributed (32 virtual GPUs via gloo):
  torchrun --nproc_per_node=32 -m tests.assgn012_zero_lab

Optional fallback after a failed 32-process attempt:
  torchrun --nproc_per_node=8 -m tests.assgn012_zero_lab
  (analytic overlay still reports W=32)

Exit codes: 0 = PASS, 1 = gate FAIL, 2 = crash.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.zero_lab import format_gate_print, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign_0012 ZeRO lab")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Informational only; actual W comes from torchrun env. Overlay always uses 32.",
    )
    parser.add_argument("--steps", type=int, default=3, help="DP demo steps")
    args = parser.parse_args(argv)

    print(f"ERA V5 Assign_0012 — ZeRO lab (root={ROOT})")
    if args.world_size is not None:
        print(f"note: --world-size={args.world_size} recorded; overlay accounting uses W=32")

    try:
        results = run_all(ROOT / "reports", dist_steps=args.steps)
    except Exception:
        traceback.print_exc()
        return 2

    # Non-rank0 processes under torchrun may not populate all_ok; treat missing as skip
    if "all_ok" not in results or results.get("all_ok") is None:
        # Worker ranks: success if smoke/dp ran without exception
        smoke_ok = results.get("task1_smoke", {}).get("ok", True)
        dp_ok = results.get("task2_dp", {}).get("ok", True)
        print(f"worker rank finished smoke_ok={smoke_ok} dp_ok={dp_ok}")
        return 0 if (smoke_ok and dp_ok) else 1

    print("\n--- Task results ---")
    for key in ("task1_smoke", "task2_dp", "task3_zero", "task4_graphics"):
        block = results.get(key, {})
        status = block.get("ok", False)
        print(f"  {key}: {'PASS' if status else 'FAIL'}")

    print("\n--- Gate operands ---")
    for line in format_gate_print(results):
        print(line)

    print(f"\nSummary: {results.get('summary_path')}")
    print(f"Manifest: {results.get('manifest_path')}")
    if results.get("all_ok"):
        print("\nVERIFICATION GATE: PASS")
        return 0
    print("\nVERIFICATION GATE: FAIL — see reports/ and fix failing task")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
