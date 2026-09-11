"""Session 11 assignment lab runner.

Run from project root:
  python tests/assgn011_optimizers_lab.py

Exit codes: 0 = all checks pass, 1 = verification failed, 2 = crash.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.optimizers_lab import format_gate_print, run_all


def main() -> int:
    print(f"ERA V5 Assign_0011 — optimizers lab (root={ROOT})")
    try:
        results = run_all(ROOT / "reports")
    except Exception:
        traceback.print_exc()
        return 2

    print("\n--- Task results ---")
    for key in (
        "task1_adam_hand",
        "task2_bias",
        "task3_ratio",
        "task4_schedules",
        "task5_lr_sweep",
    ):
        block = results.get(key, {})
        status = block.get("ok", True)
        path = (
            block.get("png_path")
            or block.get("table_path")
            or block.get("csv_path")
            or block.get("report_path")
            or ""
        )
        print(f"  {key}: {'PASS' if status else 'FAIL'} — {path}")

    print("\n--- Gate operands ---")
    print(f"all_ok: {results.get('all_ok')}")
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
