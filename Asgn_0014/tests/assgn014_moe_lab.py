"""CLI runner for Assign_0014 MoE upcycling lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.moe_lab import format_gate_print, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assgn_014 dense->MoE lab")
    parser.add_argument("--smoke", action="store_true", help="short token budgets")
    parser.add_argument("--full", action="store_true", help="25M+25M token budgets")
    parser.add_argument("--tokens-dense", type=int, default=None)
    parser.add_argument("--tokens-moe", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--reports", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--free-colab", action="store_true")
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args(argv)

    smoke = not args.full
    if args.smoke:
        smoke = True
    if args.full:
        smoke = False

    reports = Path(args.reports) if args.reports else ROOT / "reports"
    results = run_all(
        reports,
        root=ROOT,
        tokens_dense=args.tokens_dense,
        tokens_moe=args.tokens_moe,
        seq_len=args.seq_len,
        batch=args.batch,
        smoke=smoke,
        version=args.version,
        free_colab=args.free_colab,
        n_experts=args.n_experts,
        top_k=args.top_k,
    )
    print("\n".join(format_gate_print(results)))
    if results.get("all_ok"):
        print("\nVERIFICATION GATE: PASS")
        return 0
    print("\nVERIFICATION GATE: FAIL — see reports/ and tests/self_correct.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
