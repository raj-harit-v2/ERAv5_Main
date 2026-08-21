"""Build Netlify publish tree under dist/ from source demos + widgets."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

ROOT = config.ROOT
DIST = ROOT / "dist"
WEB = ROOT / "web" / "index.html"

SKIP_ASGN_NAMES = {"FROZEN.md", "CHECKPOINT_UI.md", "demo_Asgn_08_Explained.md", "freeze_hashes.json"}
CHRONO_SKIP_DIRS = {"__pycache__"}


def _copy_tree_filtered(src: Path, dst: Path, *, skip_names: set[str] | None = None, skip_dirs: set[str] | None = None) -> None:
    skip_names = skip_names or set()
    skip_dirs = skip_dirs or set()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        if item.name.startswith(".") or item.name in skip_dirs:
            continue
        if item.name == "__pycache__":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep"),
            )
        else:
            shutil.copy2(item, target)


def build() -> None:
    if not WEB.is_file():
        raise SystemExit(f"missing landing template: {WEB}")

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    shutil.copy2(WEB, DIST / "index.html")

    _copy_tree_filtered(
        ROOT / "demo_chrono",
        DIST / "demo_chrono",
        skip_dirs=CHRONO_SKIP_DIRS,
    )
    # Keep chrono_app.js for the Netlify timeline; drop Python export helpers only.
    scripts = DIST / "demo_chrono" / "scripts"
    if scripts.is_dir():
        for py in scripts.glob("*.py"):
            py.unlink()

    app_js = DIST / "demo_chrono" / "scripts" / "chrono_app.js"
    if not app_js.is_file():
        raise SystemExit(f"missing timeline app JS in dist: {app_js}")

    _copy_tree_filtered(
        ROOT / "demo_Asgn_08",
        DIST / "demo_Asgn_08",
        skip_names=SKIP_ASGN_NAMES,
    )

    widgets_src = ROOT / "demo_08" / "coach_demo" / "widgets"
    widgets_dst = DIST / "demo_08" / "coach_demo" / "widgets"
    if not widgets_src.is_dir():
        raise SystemExit(f"missing widgets: {widgets_src}")
    widgets_dst.parent.mkdir(parents=True, exist_ok=True)
    if widgets_dst.exists():
        shutil.rmtree(widgets_dst)
    shutil.copytree(
        widgets_src,
        widgets_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    print(f"Built {DIST}")
    print(f"  landing: {DIST / 'index.html'}")
    print(f"  chrono: {(DIST / 'demo_chrono').is_dir()}")
    print(f"  chrono_app: {app_js.is_file()}")
    print(f"  asgn08: {(DIST / 'demo_Asgn_08').is_dir()}")
    print(f"  widgets: {widgets_dst.is_dir()} ({len(list(widgets_dst.glob('*.html')))} html)")


if __name__ == "__main__":
    build()
