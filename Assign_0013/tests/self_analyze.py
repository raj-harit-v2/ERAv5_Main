"""Print the four-run table and missing PRIMARY PNGs (operands, not vibes)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.rev_graphics import PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED
from src.utils.versioning import resolve_reports_for_eval

PRIMARY = (PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        type=str,
        default=None,
        help="Version dir or path; default = reports/LATEST.txt",
    )
    args = parser.parse_args(argv)
    reports = resolve_reports_for_eval(REPORTS_ROOT, reports_arg=args.reports)
    csv_path = reports / "assgn013_runs.csv"
    print(f"ROOT={ROOT}")
    print(f"reports={reports}")
    if not csv_path.is_file():
        print("MISSING assgn013_runs.csv — run python tests/assgn013_rev_lab.py --smoke")
        return 1
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    print("\n--- four-run table ---")
    for row in rows:
        n = row.get("n_params")
        print(
            f"variant={row.get('variant')}  batch={row.get('batch')}  "
            f"tokens_seen={row.get('tokens_seen')}  final_loss={row.get('final_loss')}  "
            f"tok/s={row.get('tokens_per_s')}  peak_mem_bytes={row.get('peak_mem_bytes')}  "
            f"usd={row.get('usd')}  usd_shadow={row.get('usd_shadow')}  "
            f"rate={row.get('usd_per_hour')}  N={n}"
        )
    print("\n--- PRIMARY PNGs ---")
    for name in PRIMARY:
        p = reports / name
        ok = p.is_file() and p.stat().st_size > 0
        size = p.stat().st_size if p.is_file() else 0
        print(f"{name}  exists={ok}  bytes={size}")
    manifest = reports / "assgn013_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print("\n--- manifest operands ---")
        for k in (
            "n_params",
            "B_fixed",
            "B_max",
            "winner",
            "smoke",
            "all_ok",
            "version",
            "usd_per_hour",
            "rate_source",
        ):
            print(f"{k}={data.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
