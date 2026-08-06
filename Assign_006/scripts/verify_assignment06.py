#!/usr/bin/env python3
"""Closed-loop Assignment 06 verifier: wipe → run_demo → evidence PASS → pytest."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    print("=== verify_assignment06: wipe artifacts ===")
    if cfg.ARTIFACTS_DIR.exists():
        shutil.rmtree(cfg.ARTIFACTS_DIR)

    rc = _run([sys.executable, "run_demo.py"])
    if rc != 0:
        print("[FAIL] run_demo.py exited", rc)
        return rc

    evidence_path = cfg.ARTIFACTS_DIR / "evidence.json"
    if not evidence_path.exists():
        print("[FAIL] missing evidence.json")
        return 1
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    results = evidence.get("results") or {}
    overall_pass = bool(evidence.get("overall_pass"))
    print(f"evidence overall_pass={overall_pass}")
    if not overall_pass:
        for k, v in results.items():
            if isinstance(v, dict) and v.get("status") != "PASS":
                print(f"  [FAIL] {k}: {v}")
        return 1

    rc = _run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if rc != 0:
        print("[FAIL] pytest")
        return rc

    print("[PASS] verify_assignment06 closed-loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
