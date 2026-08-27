"""Build static dist/ site: landing + widgets + JSON + reports."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WEB = ROOT / "web" / "index.html"
WIDGETS = ROOT / "widgets"
EVAL = ROOT / "data" / "evaluation"
REPORTS = ROOT / "reports"

REQUIRED_JSON = [
    "seven_numbers.json",
    "part2_losses.json",
    "self_diagnostics_01.json",
    "widget_weights.json",
]

REQUIRED_PNG = [
    "Curriculum_Required_stats_01_peak_vram.png",
    "Curriculum_Required_stats_02_chunk_frontier.png",
    "Curriculum_Required_stats_03_swiglu.png",
]

REQUIRED_WIDGETS = [
    "s9_widget_0_loss_flow.html",
    "s9_widget_1_chunk_ce.html",
    "s9_widget_2_swiglu.html",
    "s9_widget_3_mtp.html",
    "s9_widget_4_stats_board.html",
    "s9_engine.js",
    "s9_boot.js",
    "s9_swiglu_diagram.js",
]


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")


def build() -> None:
    if not WEB.is_file():
        raise SystemExit(
            f"missing landing template: {WEB}\n"
            "DIST needs web/index.html (and widgets/). "
            "On Colab, re-zip Assign_009_Init including web/ and widgets/, then re-upload."
        )

    for name in REQUIRED_JSON:
        _require(EVAL / name, "evaluation JSON")
    for name in REQUIRED_PNG:
        _require(REPORTS / name, "report PNG")
    for name in REQUIRED_WIDGETS:
        _require(WIDGETS / name, "widget asset")

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    shutil.copy2(WEB, DIST / "index.html")

    dst_data = DIST / "data"
    dst_data.mkdir()
    for name in REQUIRED_JSON:
        shutil.copy2(EVAL / name, dst_data / name)

    dst_reports = DIST / "reports"
    dst_reports.mkdir()
    for name in REQUIRED_PNG:
        shutil.copy2(REPORTS / name, dst_reports / name)

    dst_widgets = DIST / "widgets"
    dst_widgets.mkdir()
    for name in REQUIRED_WIDGETS:
        shutil.copy2(WIDGETS / name, dst_widgets / name)

    netlify = ROOT / "netlify.toml"
    if netlify.is_file():
        shutil.copy2(netlify, DIST / "netlify.toml")

    print(f"Built {DIST}")
    print(f"  landing: {DIST / 'index.html'}")
    print(f"  widgets: {len(REQUIRED_WIDGETS)} files")
    print(f"  data json: {len(REQUIRED_JSON)} files")
    print(f"  reports png: {len(REQUIRED_PNG)} files")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    build()
