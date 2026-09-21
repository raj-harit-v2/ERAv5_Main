"""Session 13 assignment lab runner.

Smoke (CPU/CUDA, writes reports/vNNN_.../):
  python tests/assgn013_rev_lab.py --smoke --tokens 4096

Full 50M token budget (Colab GPU):
  python tests/assgn013_rev_lab.py --full --tokens 50000000

Resume after Colab disconnect:
  set ASSGN013_RUN_VERSION=v002_colab_t4_full
  python tests/assgn013_rev_lab.py --full --tokens 50000000 --version v002_colab_t4_full

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

from src.pipeline.rev_lab import format_gate_print, run_all
from src.pipeline.train_rev import dump_run_json, train_one
from src.utils.metrics import FULL_TOKEN_BUDGET, pick_device
from src.utils.versioning import read_latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign_0013 reversibility lab")
    parser.add_argument("--mode", choices=("standard", "euler", "midpoint", "all"), default="all")
    parser.add_argument("--tokens", type=int, default=None, help="Token budget this invocation")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Short run; exempts 50M token gate")
    parser.add_argument("--full", action="store_true", help="50M tokens per run (Colab)")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Reports root (default: ROOT/reports). Version folders are created under it.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Resume/write into this version id (e.g. v002_colab_t4_full)",
    )
    parser.add_argument(
        "--free-colab",
        action="store_true",
        help="Advertised USD=0; still stamps usd_shadow at T4 0.12",
    )
    args = parser.parse_args(argv)

    smoke = not args.full
    if args.smoke:
        smoke = True
    if args.full:
        smoke = False
    tokens = args.tokens
    if tokens is None and smoke:
        tokens = 4096
    if tokens is None and not smoke:
        tokens = FULL_TOKEN_BUDGET

    reports_root = Path(args.reports_dir) if args.reports_dir else (ROOT / "reports")
    if not reports_root.is_absolute():
        reports_root = (ROOT / reports_root).resolve()

    print(f"ERA V5 Assign_0013 - reversibility lab (root={ROOT})")
    print(f"device={pick_device()}  smoke={smoke}  tokens={tokens}  mode={args.mode}")
    print(f"reports_root={reports_root}")

    try:
        if args.mode == "all":
            results = run_all(
                reports_root,
                root=ROOT,
                tokens=tokens,
                seq_len=args.seq_len,
                batch=args.batch,
                smoke=smoke,
                version=args.version,
                free_colab=args.free_colab,
            )
        else:
            bsz = args.batch or 2
            seq = args.seq_len or (128 if smoke else 512)
            metrics, history = train_one(
                args.mode,
                batch=bsz,
                seq_len=seq,
                token_budget=tokens,
                smoke=smoke,
            )
            out = reports_root / "runs" / f"{args.mode}_B{bsz}.json"
            dump_run_json(out, metrics, history)
            print(metrics.as_dict())
            return 0
    except Exception:
        traceback.print_exc()
        return 2

    print("\n--- Gate operands ---")
    for line in format_gate_print(results):
        print(line)
    print(f"\nLATEST={read_latest(reports_root)}")
    print(f"Summary: {results.get('summary_path')}")
    if results.get("all_ok"):
        print("\nVERIFICATION GATE: PASS")
        return 0
    print("\nVERIFICATION GATE: FAIL — see reports/ and tests/self_correct.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
