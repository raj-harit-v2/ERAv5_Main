"""Session 10 assignment lab runner.

Run from project root:
  python tests/assgn010_training_loop_lab.py

Exit codes: 0 = all checks pass, 1 = verification failed, 2 = crash.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.training_loop_lab import run_all


def main() -> int:
    print(f"ERA V5 Assign_010 — training loop lab (root={ROOT})")
    try:
        results = run_all(ROOT / "reports")
    except Exception:
        traceback.print_exc()
        return 2

    print("\n--- Task results ---")
    for key in (
        "task1_shapes",
        "task2_gradient",
        "task3_accumulation",
        "task4_grad_norm",
        "task5_mfu",
        "task6_precision",
        "task6_fp_tables",
    ):
        block = results.get(key, {})
        status = block.get("ok", block.get("passed", True))
        print(f"  {key}: {'PASS' if status else 'FAIL'} — {block.get('path', block.get('png_path', block.get('csv_path', '')))}")

    tables = results.get("task6_fp_tables") or {}
    if tables.get("ascii_table1"):
        print("\n--- Table 1: FP formats (0.1 at index 0) ---")
        print(tables["ascii_table1"])
    if tables.get("ascii_table2"):
        print("\n--- Table 2: MXFP4 + MBS + OAS ---")
        print(tables["ascii_table2"])

    print(f"\nSummary: {results.get('summary_path')}")
    print(f"Manifest: {results.get('manifest_path')}")
    if results.get("all_ok"):
        print("\nVERIFICATION GATE: PASS")
        return 0
    print("\nVERIFICATION GATE: FAIL — see reports/ and fix failing task")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
