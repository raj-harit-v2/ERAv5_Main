"""Tests for demo_chrono Assignment §18 chronology wall."""

from __future__ import annotations

import json
from pathlib import Path

import config

CHRONO = config.ROOT / "demo_chrono"
VERIFY = "VERIFY_PRIMARY_SOURCE"


def _table_data_lines(md: str, header_prefix: str) -> list[str]:
    return [
        ln
        for ln in md.splitlines()
        if ln.startswith("|")
        and not ln.startswith("| :")
        and not ln.startswith("| ---")
        and not ln.startswith(header_prefix)
    ]


def _load_chrono() -> dict:
    return json.loads((CHRONO / "data" / "chronology.json").read_text(encoding="utf-8"))


def _load_required() -> list[str]:
    return json.loads((CHRONO / "data" / "required_mechanisms.json").read_text(encoding="utf-8"))


def test_chrono_files_exist():
    assert (CHRONO / "index.html").is_file()
    assert (CHRONO / "data" / "chronology.json").is_file()
    assert (CHRONO / "data" / "required_mechanisms.json").is_file()
    assert (CHRONO / "scripts" / "chrono_app.js").is_file()
    assert (CHRONO / "CHRONOLOGY_SOURCES_BY_DATE.md").is_file()
    assert (CHRONO / "CHRONOLOGY_Compare.md").is_file()
    assert (config.ROOT / "demo_Asgn_08" / "freeze_hashes.json").is_file()


def test_required_mechanism_ids():
    required = _load_required()
    data = _load_chrono()
    ids = {c["id"] for c in data["cards"]}
    assert len(required) >= 18
    assert ids == set(required)


def test_card_required_fields():
    data = _load_chrono()
    for c in data["cards"]:
        for key in (
            "id",
            "title",
            "year_display",
            "year_sort",
            "problem",
            "mechanism",
            "buy",
            "give_up",
            "when",
        ):
            assert c.get(key), f"{c.get('id')}: missing {key}"
        src = c["source"]
        assert src.get("label")
        assert "date_verified" in src
        if src.get("date_verified"):
            assert src.get("url"), c["id"]
            assert src.get("pdf_verified"), c["id"]
            assert src.get("pdf_v1"), c["id"]
            assert c["year_display"] != VERIFY, c["id"]
            assert isinstance(c["year_sort"], int) and c["year_sort"] >= 10000101


def test_verify_discipline():
    data = _load_chrono()
    for c in data["cards"]:
        if not c["source"].get("date_verified"):
            assert c["year_display"] == VERIFY, c["id"]


def test_html_timeline_hooks():
    html = (CHRONO / "index.html").read_text(encoding="utf-8")
    assert 'id="timeline"' in html
    assert "chrono_app.js" in html
    assert "chronology.embed.js" in html
    assert "demo_Asgn_08" in html
    assert "What the timeline shows" in html
    assert "CHRONOLOGY_SOURCES_BY_DATE.md" in html
    assert "CHRONOLOGY_SOURCES.md" not in html.split("footer")[1] if "footer" in html else True
    assert "CHRONOLOGY_SOURCES_All.md" not in html


def test_embed_js_matches_json():
    data = _load_chrono()
    embed = (CHRONO / "data" / "chronology.embed.js").read_text(encoding="utf-8")
    assert "window.CHRONOLOGY_DATA" in embed
    assert data["schema"] in embed
    for rid in _load_required():
        assert rid in embed


def test_sources_by_date_md():
    md = (CHRONO / "CHRONOLOGY_SOURCES_BY_DATE.md").read_text(encoding="utf-8")
    assert "| Sr_No | id |" in md
    assert "by launch date" in md.lower()
    assert "pdf_v1" in md
    lines = _table_data_lines(md, "| Sr_No | id |")
    assert len(lines) >= 18
    first_data = lines[0]
    parts = [p.strip() for p in first_data.split("|") if p.strip()]
    assert parts[0] == "1"
    assert parts[1] == "standard_attention"


def test_sources_compare_md():
    md = (CHRONO / "CHRONOLOGY_Compare.md").read_text(encoding="utf-8")
    assert "Table A" not in md
    assert "Chronology vs teaching order" in md
    assert "| Sr_No | Te_No | id |" in md
    assert "Sec_No" not in md
    assert "[url](http" in md
    assert "flow_teaching.svg" not in md
    assert md.count("#fde047") == 2
    data = _table_data_lines(md, "| Sr_No | Te_No | id |")
    assert len(data) == 20
    first = [p.strip() for p in data[0].split("|") if p.strip()]
    assert first[0] == "1"
    assert first[1] == "1"
    assert first[2] == "standard_attention"
    rope_row = next(ln for ln in data if "| rope |" in ln or "| rope " in ln)
    cols = [p.strip() for p in rope_row.split("|")]
    assert cols[1] == "8"
    assert cols[2] == "8"
    assert cols[3] == "rope"
    wall_rows = [ln for ln in data if "#fde047" not in ln]
    assert len(wall_rows) == 18
    extra_rows = [ln for ln in data if "#fde047" in ln]
    assert len(extra_rows) == 2
    for ln in extra_rows:
        cols = [p.strip() for p in ln.split("|")]
        assert cols[1] in ("10", "12")
        assert cols[2] == ""


def test_verified_sort_is_ymd():
    data = _load_chrono()
    order = data.get("provisional_order") or []
    rank = {cid: i for i, cid in enumerate(order)}
    cards = data["cards"]
    sorted_ids = [
        c["id"]
        for c in sorted(cards, key=lambda c: (int(c["year_sort"]), rank.get(c["id"], 9999)))
    ]
    assert sorted_ids[0] == "standard_attention"
    assert sorted_ids[-1] == "deepseek_compressed_sparse"
    assert all(c["source"].get("date_verified") for c in cards)
    assert all(c["source"].get("pdf_verified") for c in cards)
    assert all(c["source"].get("pdf_v1") for c in cards)
    js = (CHRONO / "scripts" / "chrono_app.js").read_text(encoding="utf-8")
    assert "sortCards" in js
    assert "provisional_order" in js
    assert "year_sort" in js
    assert "pdf_verified" in js
    assert "pdf_v1" in js
    assert ">pdf_v</a>" in js
    assert "99999999" in js
