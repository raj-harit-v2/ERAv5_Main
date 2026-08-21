"""Asgn_08_cleanup.py — remove caches and generated artifacts under Assign_008.

Safe defaults: never deletes .venv/, dist/, source demos, web/, README,
or submission_artifacts/evidence.* (unless --with-evidence).

Usage:
  uv run python tests/Asgn_08_cleanup.py                    # dry-run
  uv run python tests/Asgn_08_cleanup.py --apply            # delete caches/diags
  uv run python tests/Asgn_08_cleanup.py --apply --with-evidence
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

ROOT = config.ROOT

CACHE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})

GENERATED_TREES = (
    ROOT / "diagnostics_02",
    ROOT / "diagnostics_chrono",
)

ROOT_REPORT_GLOBS = (
    "Diagionistics_*.md",
    "Chrono_self_Diagnostics.md",
    "*_Diagnostics.md",
    "*_Diagionistics*.md",
)

CLEAR_DIRS_KEEP_GITKEEP = (
    ROOT / "demo_08" / "coach_demo" / "figures",
)

EVIDENCE_FILES = (
    ROOT / "submission_artifacts" / "evidence.md",
    ROOT / "submission_artifacts" / "evidence.json",
    ROOT / "submission_artifacts" / "attention_weights_heatmap.png",
)


def _under_venv(path: Path) -> bool:
    try:
        path.resolve().relative_to((ROOT / ".venv").resolve())
        return True
    except (ValueError, OSError):
        return False


def _rm(path: Path, *, apply: bool, removed: list[str]) -> None:
    if not path.exists():
        return
    removed.append(str(path.relative_to(ROOT)) + ("/" if path.is_dir() else ""))
    if not apply:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _clear_keep_gitkeep(path: Path, *, apply: bool, removed: list[str]) -> None:
    if not path.is_dir():
        return
    for item in path.iterdir():
        if item.name == ".gitkeep":
            continue
        _rm(item, apply=apply, removed=removed)


def collect_cache_dirs() -> list[Path]:
    found: list[Path] = []
    for name in CACHE_DIR_NAMES:
        for match in ROOT.rglob(name):
            if match.is_dir() and not _under_venv(match):
                found.append(match)
    found.sort(key=lambda p: len(p.parts), reverse=True)
    return found


def plan_removals(*, with_evidence: bool) -> list[Path]:
    targets: list[Path] = []
    targets.extend(collect_cache_dirs())
    for tree in GENERATED_TREES:
        if tree.exists():
            targets.append(tree)
    for pattern in ROOT_REPORT_GLOBS:
        targets.extend(p for p in ROOT.glob(pattern) if p.is_file())
    if with_evidence:
        targets.extend(p for p in EVIDENCE_FILES if p.exists())

    seen: set[Path] = set()
    unique: list[Path] = []
    for t in targets:
        key = t.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def run(*, apply: bool, with_evidence: bool) -> int:
    removed: list[str] = []
    print(f"Assign_008 cleanup @ {ROOT}")
    print(f"mode: {'APPLY' if apply else 'DRY-RUN'}")
    if with_evidence:
        print("evidence: included")
    print()

    for path in plan_removals(with_evidence=with_evidence):
        _rm(path, apply=apply, removed=removed)

    for path in CLEAR_DIRS_KEEP_GITKEEP:
        _clear_keep_gitkeep(path, apply=apply, removed=removed)

    if not removed:
        print("Nothing to clean.")
        return 0

    label = "Removed" if apply else "Would remove"
    print(f"{label} ({len(removed)}):")
    for rel in removed:
        print(f"  - {rel}")
    if not apply:
        print()
        print("Re-run with --apply to delete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Assign_008 caches and generated artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run).")
    parser.add_argument(
        "--with-evidence",
        action="store_true",
        help="Also delete submission_artifacts/evidence.* (regenerate via: uv run python run_demo.py).",
    )
    args = parser.parse_args()
    return run(apply=args.apply, with_evidence=args.with_evidence)


if __name__ == "__main__":
    raise SystemExit(main())
