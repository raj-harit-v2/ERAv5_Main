"""Section 6 gates with operands (never a naked True)."""

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
from src.pipeline.train_rev import OPTIMIZER_NAME
from src.utils.metrics import FULL_TOKEN_BUDGET, N_PARAMS_MAX, N_PARAMS_MIN, params_in_range
from src.utils.versioning import resolve_reports_for_eval

PRIMARY = (PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED)


def evaluate(smoke_ok: bool, reports: Path | None = None) -> dict:
    reports = reports or resolve_reports_for_eval(REPORTS_ROOT)
    csv_path = reports / "assgn013_runs.csv"
    manifest_path = reports / "assgn013_manifest.json"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8"))) if csv_path.is_file() else []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    n = int(manifest.get("n_params") or (rows[0]["n_params"] if rows else 0) or 0)
    n_ok = params_in_range(n)
    variants = {r["variant"] for r in rows}
    b_fixed = int(manifest.get("B_fixed") or 0)
    b_max = int(manifest.get("B_max") or 0)
    has_euler = "euler" in variants
    has_mid = "midpoint" in variants
    has_std = "standard" in variants
    pngs = []
    for name in PRIMARY:
        p = reports / name
        pngs.append(p.is_file() and p.stat().st_size > 0)
    png_ok = all(pngs)
    dropouts = [float(r.get("dropout") or 0) for r in rows if r.get("variant") in ("euler", "midpoint")]
    drop_ok = all(d == 0.0 for d in dropouts) if dropouts else False
    opt_ok = True
    if rows:
        opt_ok = all(r.get("optimizer") == OPTIMIZER_NAME for r in rows)
    smoke = bool(manifest.get("smoke", True))
    token_ok = True
    if not smoke and not smoke_ok:
        token_ok = all(float(r["tokens_seen"]) >= FULL_TOKEN_BUDGET for r in rows) if rows else False
    b_ok = b_max > b_fixed > 0 if (b_max and b_fixed) else False
    all_ok = n_ok and has_euler and has_mid and has_std and png_ok and drop_ok and opt_ok and token_ok and b_ok

    return {
        "reports": str(reports),
        "version": manifest.get("version"),
        "n_params": n,
        "n_ok": n_ok,
        "has_euler": has_euler,
        "has_midpoint": has_mid,
        "has_standard": has_std,
        "B_fixed": b_fixed,
        "B_max": b_max,
        "b_ok": b_ok,
        "png_ok": png_ok,
        "primary_pngs": sum(1 for x in pngs if x),
        "drop_ok": drop_ok,
        "opt_ok": opt_ok,
        "token_ok": token_ok,
        "smoke": smoke,
        "usd_per_hour": manifest.get("usd_per_hour"),
        "rate_source": manifest.get("rate_source"),
        "all_ok": all_ok,
        "winner": manifest.get("winner"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Exempt 50M token gate")
    parser.add_argument(
        "--reports",
        type=str,
        default=None,
        help="Version dir (e.g. reports/v001_local_cpu_smoke); default = LATEST.txt",
    )
    args = parser.parse_args(argv)
    reports = resolve_reports_for_eval(REPORTS_ROOT, reports_arg=args.reports)
    gate = evaluate(smoke_ok=args.smoke, reports=reports)
    print("ASSGN013 self_evaluate operands:")
    for k, v in gate.items():
        print(f"  {k}={v}")
    print(f"all_ok={gate['all_ok']}  (PASS if True; FAIL if False)")
    print(f"N range [{N_PARAMS_MIN},{N_PARAMS_MAX}]  FULL_TOKEN_BUDGET={FULL_TOKEN_BUDGET}")
    return 0 if gate["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
