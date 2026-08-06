"""CLI: reconstruct a consumption event by global step."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.ledger import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Audit query for consumption ledger")
    p.add_argument("--step", type=int, required=True)
    p.add_argument(
        "--ledger",
        type=Path,
        default=cfg.LEDGERS_DIR / "consumption_ledger.jsonl",
    )
    args = p.parse_args()
    rows = read_jsonl(args.ledger)
    hit = [r for r in rows if r.get("global_step") == args.step and r.get("event") == "consumption"]
    if not hit:
        print(json.dumps({"ok": False, "error": "not_found", "step": args.step}))
        sys.exit(1)
    print(json.dumps({"ok": True, "record": hit[0]}, indent=2))


if __name__ == "__main__":
    main()
