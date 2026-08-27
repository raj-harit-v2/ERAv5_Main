"""Asgn_09_cleanup.py — move local clutter out of Assign_009_Init before zip/submit.

Moves (does not delete) backup and cache paths into the Session 9 referral folder:

  ERA_V5/Assign_009_Ses_O/Asgn_09_All_docs_xfr/Asgn_09_Moved/

Targets:
  - data/evaluation/*_v1.* and reports/*_v1.*
  - .pytest_cache/
  - notebooks/.ipynb_checkpoints/
  - project __pycache__/ trees (never under .venv/)

Never touches .venv/, src/, canonical evaluation JSON/PNG, widgets/, or web/.

Usage (from Assign_009_Init project root)::

    python tests/Asgn_09_cleanup.py
    python tests/Asgn_09_cleanup.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = (
    ROOT.parent / "Assign_009_Ses_O" / "Asgn_09_All_docs_xfr" / "Asgn_09_Moved"
)


def _unique_dest(dest_root: Path, relative: Path) -> Path:
    """Prefer preserving relative path under dest; collide with numeric suffix."""
    candidate = dest_root / relative
    if not candidate.exists():
        return candidate
    stem = candidate.name
    parent = candidate.parent
    n = 1
    while True:
        alt = parent / f"{stem}__{n}"
        if not alt.exists():
            return alt
        n += 1


def _collect_targets(root: Path) -> list[Path]:
    targets: list[Path] = []

    for folder in (root / "data" / "evaluation", root / "reports"):
        if folder.is_dir():
            for p in sorted(folder.glob("*_v1.*")):
                if p.is_file():
                    targets.append(p)

    pytest_cache = root / ".pytest_cache"
    if pytest_cache.exists():
        targets.append(pytest_cache)

    checkpoints = root / "notebooks" / ".ipynb_checkpoints"
    if checkpoints.exists():
        targets.append(checkpoints)

    for pycache in sorted(root.rglob("__pycache__")):
        if not pycache.is_dir():
            continue
        try:
            pycache.relative_to(root / ".venv")
            continue
        except ValueError:
            pass
        targets.append(pycache)

    # Deduplicate while preserving order (parent dirs before nested if any)
    seen: set[Path] = set()
    unique: list[Path] = []
    for t in targets:
        resolved = t.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(t)
    return unique


def _move_one(src: Path, dest_root: Path, *, dry_run: bool) -> Path:
    rel = src.relative_to(ROOT)
    dest = _unique_dest(dest_root, rel)
    if dry_run:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move Assign_009_Init clutter into Asgn_09_Moved (pre-submit)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned moves only; do not move anything.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    dest_root: Path = args.dest.resolve()
    targets = _collect_targets(ROOT)

    if not targets:
        print(f"Nothing to clean under {ROOT}")
        print(f"Destination would be: {dest_root}")
        return 0

    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    print(f"ROOT: {ROOT}")
    print(f"DEST: {dest_root}")
    print(f"MODE: {'dry-run' if args.dry_run else 'move'}")
    print(f"Items: {len(targets)}")
    print("---")

    moved: list[tuple[Path, Path]] = []
    for src in targets:
        dest = _move_one(src, dest_root, dry_run=args.dry_run)
        moved.append((src, dest))
        verb = "WOULD MOVE" if args.dry_run else "MOVED"
        print(f"{verb}: {src.relative_to(ROOT)} -> {dest}")

    print("---")
    print(f"Summary: {len(moved)} path(s) {'planned' if args.dry_run else 'moved'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
