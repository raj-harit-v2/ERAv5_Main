"""Print dense/MoE table and missing PRIMARY PNGs (operands, not vibes)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.moe_graphics import PRIMARY_LOAD, PRIMARY_LOSS, PRIMARY_SPEED
from src.utils.versioning import resolve_reports_for_eval

REPORTS_ROOT = ROOT / "reports"
PRIMARY = (PRIMARY_LOSS, PRIMARY_LOAD, PRIMARY_SPEED)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=str, default=None)
    args = parser.parse_args(argv)
    reports = resolve_reports_for_eval(REPORTS_ROOT, reports_arg=args.reports)
    print(f"ASSGN014 self_analyze reports={reports}")

    csv_path = reports / "assgn014_runs.csv"
    if csv_path.is_file():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        print("\n--- runs ---")
        for r in rows:
            print(
                f"  {r.get('variant')}: N={r.get('n_params')} active={r.get('n_active')} "
                f"tokens={r.get('tokens_seen')} loss={r.get('final_loss')} "
                f"tok/s={r.get('tokens_per_s')} mem={r.get('peak_mem_bytes')} "
                f"max_vio={r.get('max_vio')} dead={r.get('dead_experts')}"
            )
    else:
        print("missing assgn014_runs.csv")

    manifest_path = reports / "assgn014_manifest.json"
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("\n--- manifest ---")
        for k in (
            "version",
            "n_dense",
            "n_moe_total",
            "n_moe_active",
            "loss_at_convert",
            "final_moe_loss",
            "max_vio",
            "dead_experts",
            "use_ce_aux",
            "router_dtype",
            "all_ok",
        ):
            print(f"  {k}={m.get(k)}")
    else:
        print("missing assgn014_manifest.json")

    print("\n--- PRIMARY PNGs ---")
    for name in PRIMARY:
        p = reports / name
        ok = p.is_file() and p.stat().st_size > 0
        print(f"  {name}: {'OK' if ok else 'MISSING'} ({p})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
