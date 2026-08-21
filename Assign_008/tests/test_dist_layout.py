"""Sanity checks for Netlify dist layout (source templates + build script)."""

from __future__ import annotations

import config

ROOT = config.ROOT


def test_landing_template_exists():
    assert (ROOT / "web" / "index.html").is_file()


def test_root_index_removed():
    assert not (ROOT / "index.html").exists()


def test_build_dist_script_exists():
    assert (ROOT / "tests" / "build_dist.py").is_file()


def test_netlify_toml_publish_dist():
    text = (ROOT / "netlify.toml").read_text(encoding="utf-8")
    assert 'publish = "dist"' in text


def test_widgets_source_present():
    widgets = ROOT / "demo_08" / "coach_demo" / "widgets"
    assert widgets.is_dir()
    assert any(widgets.glob("*.html"))


def test_chrono_app_js_present_in_source():
    assert (ROOT / "demo_chrono" / "scripts" / "chrono_app.js").is_file()


def test_build_dist_keeps_chrono_app_js():
    from build_dist import build

    build()
    assert (ROOT / "dist" / "demo_chrono" / "scripts" / "chrono_app.js").is_file()
    assert not list((ROOT / "dist" / "demo_chrono" / "scripts").glob("*.py"))
    html = (ROOT / "dist" / "demo_chrono" / "index.html").read_text(encoding="utf-8")
    assert "scripts/chrono_app.js" in html
    assert "chronology.embed.js" in html
