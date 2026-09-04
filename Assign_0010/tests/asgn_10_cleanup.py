"""asgn_10_cleanup.py — move local clutter out of Assign_0010 before zip/submit.

Moves (does not delete) session dumps, IDE stubs, and caches into:

  ERA_V5/Assign_0010_Ses_O/Asgn_10_Moved/

Targets (if present under project root):
  - Untitled.canvas / Untitled*
  - Session 10*, *.mhtml, assignment_10_Chat.txt
  - Short_Future_AMBIGUITY*, Cursor IDE Prompt*, Generate README*
  - Assgn_010_*, Sess_10_*, Project_Scaffolding.md
  - .pytest_cache/, project __pycache__/ (never under .venv/)
  - .cursor/plans/ (plans only; rules stay until whole .cursor is ignored)

Never touches: .venv/, src/, tests/ (except this script), reports/assgn010_* evidence.

Usage (from Assign_0010 project root)::

    python tests/asgn_10_cleanup.py
    python tests/asgn_10_cleanup.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT.parent / "Assign_0010_Ses_O" / "Asgn_10_Moved"

# Exact root filenames / prefixes to move when present
ROOT_NAME_GLOBS: tuple[str, ...] = (
    "Untitled*",
    "Session 10*",
    "*.mhtml",
    "assignment_10_Chat.txt",
    "Short_Future_AMBIGUITY*",
    "Cursor IDE Prompt*",
    "Generate README*",
    "Assgn_010_*",
    "Sess_10_*",
    "Project_Scaffolding.md",
)


def _unique_dest(dest_root: Path, relative: Path) -> Path:
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

    for pattern in ROOT_NAME_GLOBS:
        for p in sorted(root.glob(pattern)):
            if p.name in {".gitignore", "README.md", "requirements.txt"}:
                continue
            if p.is_file() or p.is_dir():
                targets.append(p)

    pytest_cache = root / ".pytest_cache"
    if pytest_cache.exists():
        targets.append(pytest_cache)

    plans = root / ".cursor" / "plans"
    if plans.exists():
        targets.append(plans)

    for pycache in sorted(root.rglob("__pycache__")):
        if not pycache.is_dir():
            continue
        try:
            pycache.relative_to(root / ".venv")
            continue
        except ValueError:
            pass
        targets.append(pycache)

    seen: set[Path] = set()
    unique: list[Path] = []
    for t in targets:
        resolved = t.resolve()
        if resolved in seen:
            continue
        # Skip anything under src/ or tests/
        try:
            t.relative_to(root / "src")
            continue
        except ValueError:
            pass
        try:
            t.relative_to(root / "tests")
            continue
        except ValueError:
            pass
        # Never move canonical reports evidence
        try:
            rel = t.relative_to(root / "reports")
            if rel.name.startswith("assgn010") or rel.name.startswith("Assgn010"):
                continue
        except ValueError:
            pass
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
        description="Move Assign_0010 clutter into Asgn_10_Moved (pre-submit)."
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
