"""Assgn_014 gates with operands (never a naked True)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.moe_graphics import PRIMARY_LOAD, PRIMARY_LOSS, PRIMARY_SPEED
from src.pipeline.moe_lab import evaluate_gates
from src.utils.versioning import resolve_reports_for_eval

REPORTS_ROOT = ROOT / "reports"
PRIMARY = (PRIMARY_LOSS, PRIMARY_LOAD, PRIMARY_SPEED)


def evaluate(smoke_ok: bool = True, reports: Path | None = None) -> dict:
    reports = reports or resolve_reports_for_eval(REPORTS_ROOT)
    manifest_path = reports / "assgn014_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    smoke = bool(manifest.get("smoke", True))
    pngs = []
    for name in PRIMARY:
        p = reports / name
        pngs.append(p.is_file() and p.stat().st_size > 0)
    manifest["primary_pngs"] = sum(1 for x in pngs if x)
    gates = evaluate_gates(manifest, smoke=smoke)
    gates.update(
        {
            "reports": str(reports),
            "version": manifest.get("version"),
            "smoke": smoke,
            "primary_pngs": manifest["primary_pngs"],
            "usd_per_hour": manifest.get("usd_per_hour"),
            "rate_source": manifest.get("rate_source"),
            "n_experts": manifest.get("n_experts"),
            "top_k": manifest.get("top_k"),
            "use_ce_aux": manifest.get("use_ce_aux"),
            "router_dtype": manifest.get("router_dtype"),
        }
    )
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=str, default=None)
    args = parser.parse_args(argv)
    reports = resolve_reports_for_eval(REPORTS_ROOT, reports_arg=args.reports)
    gate = evaluate(smoke_ok=True, reports=reports)
    print("ASSGN014 self_evaluate operands:")
    for k, v in gate.items():
        print(f"  {k}={v}")
    print(f"all_ok={gate['all_ok']}  (PASS if True; FAIL if False)")
    return 0 if gate["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
