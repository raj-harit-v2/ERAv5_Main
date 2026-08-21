"""Export CHRONOLOGY_SOURCES_BY_DATE.md + Compare from chronology.json.

Sr_No  = rank by year_sort ascending (launch date)
Te_No  = 1-based index in teaching_order (Full Doc + Transcript)
Assignment §18 cover list is NOT teaching order.
Compare table does not show Sec_No (kept in JSON / TEACHING_ORDER.md only).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chronology.json"
OUT_BY_DATE = ROOT / "CHRONOLOGY_SOURCES_BY_DATE.md"
OUT_COMPARE = ROOT / "CHRONOLOGY_Compare.md"
EMBED_JS = ROOT / "data" / "chronology.embed.js"

MISSING_MARK = '<span style="background:#fde047;color:#0c1018;font-weight:700">missing</span>'

EXTRA_MISSING_SOURCES: list[dict] = [
    {
        "id": "flash_attention **missing**",
        "title": "FlashAttention (IO-aware exact attention)",
        "year_display": "27-May-2022",
        "year_sort": 20220527,
        "sec_no": None,
        "source": {
            "label": "Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "url": "https://arxiv.org/abs/2205.14135",
            "pdf_verified": "arXiv:2205.14135",
            "pdf_v1": "https://arxiv.org/pdf/2205.14135",
            "date_verified": False,
        },
    },
    {
        "id": "position_interpolation **missing**",
        "title": "Position Interpolation (RoPE stretch)",
        "year_display": "27-Jun-2023",
        "year_sort": 20230627,
        "sec_no": None,
        "source": {
            "label": "Chen et al. — Extending Context Window of Large Language Models via Positional Interpolation",
            "url": "https://arxiv.org/abs/2306.15595",
            "pdf_verified": "arXiv:2306.15595",
            "pdf_v1": "https://arxiv.org/pdf/2306.15595",
            "date_verified": False,
        },
    },
]


def write_embed(data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    EMBED_JS.write_text(
        "window.CHRONOLOGY_DATA = " + payload + ";\n",
        encoding="utf-8",
    )


def _esc(s: str) -> str:
    return (s or "").replace("|", "\\|")


def _url_link(url: str) -> str:
    u = (url or "").strip()
    return f"[url]({u})" if u else "—"


def _id_with_missing_highlight(raw_id: str) -> str:
    if "**missing**" in raw_id:
        base = raw_id.replace(" **missing**", "").replace("**missing**", "").strip()
        return f"{base} {MISSING_MARK}"
    return raw_id


def _format_row(c: dict, sr_no: int, *, highlight_missing: bool = False) -> str:
    src = c.get("source") or {}
    verified = src.get("date_verified", False)
    raw_id = c.get("id", "")
    display_id = _id_with_missing_highlight(raw_id) if highlight_missing else raw_id
    return "| {sr} | {id} | {title} | {year} | {label} | {url} | {pdf} | {pdfv1} | {ver} |".format(
        sr=sr_no,
        id=display_id,
        title=_esc(c.get("title") or ""),
        year=c.get("year_display", ""),
        label=_esc(src.get("label") or ""),
        url=_esc(src.get("url") or "") or "—",
        pdf=_esc(src.get("pdf_verified") or src.get("date_note") or "") or "—",
        pdfv1=_esc(src.get("pdf_v1") or "") or "—",
        ver="yes" if verified else "no",
    )


def _format_slim_compare_row(
    c: dict,
    sr_no: int,
    te_no: int | None = None,
    *,
    highlight_missing: bool = False,
) -> str:
    src = c.get("source") or {}
    raw_id = c.get("id", "")
    display_id = _id_with_missing_highlight(raw_id) if highlight_missing else raw_id
    te_cell = "" if te_no is None else str(te_no)
    return "| {sr} | {te} | {id} | {year} | {label} | {url} |".format(
        sr=sr_no,
        te=te_cell,
        id=display_id,
        year=c.get("year_display", ""),
        label=_esc(src.get("label") or ""),
        url=_url_link(src.get("url") or ""),
    )


def _header_lines(title: str, subtitle: str) -> list[str]:
    return [
        title,
        "",
        subtitle,
        "",
        "| Sr_No | id | title | year_display | source.label | source.url | pdf_verified | pdf_v1 | date_verified |",
        "| ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]


def _sort_by_date(cards: list[dict], rank: dict[str, int]) -> list[dict]:
    return sorted(
        cards,
        key=lambda c: (int(c.get("year_sort") or 99999999), rank.get(c.get("id", ""), 9999)),
    )


def _card_by_id(cards: list[dict]) -> dict[str, dict]:
    return {c.get("id", ""): c for c in cards}


def _teaching_order(data: dict) -> list[str]:
    order = data.get("teaching_order")
    if order:
        return list(order)
    return list(data.get("assignment_cover_order") or data.get("provisional_order") or [])


def _cover_rank(data: dict) -> dict[str, int]:
    cover = data.get("assignment_cover_order") or data.get("provisional_order") or []
    return {cid: i for i, cid in enumerate(cover)}


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    write_embed(data)
    cards = list(data.get("cards") or [])
    teach = _teaching_order(data)
    te_by_id = {cid: i for i, cid in enumerate(teach, start=1)}
    cover_rank = _cover_rank(data)
    cards_by_date = _sort_by_date(cards, cover_rank)

    by_date_lines = _header_lines(
        "# Chronology sources (by launch date)",
        "Sorted ascending by `year_sort` (YYYYMMDD). Generated from `data/chronology.json`. "
        "**Sr_No** = launch-date rank only.",
    )
    for i, c in enumerate(cards_by_date, start=1):
        by_date_lines.append(_format_row(c, i))
    by_date_lines.append("")
    OUT_BY_DATE.write_text("\n".join(by_date_lines), encoding="utf-8")

    all_cards = _sort_by_date(cards_by_date + list(EXTRA_MISSING_SOURCES), cover_rank)
    date_sr_by_id = {c.get("id", ""): i for i, c in enumerate(all_cards, start=1)}

    compare_lines = [
        "# Chronology compare — launch date vs teaching order",
        "",
        "- **Sr_No** = launch-date rank (`year_sort`).",
        "- **Te_No** = teaching rank (`teaching_order`).",
        "",
        "See [`TEACHING_ORDER.md`](TEACHING_ORDER.md).",
        "",
        "## Chronology vs teaching order",
        "",
        "| Sr_No | Te_No | id | year_display | source.label | source.url |",
        "| ---: | ---: | :--- | :--- | :--- | :--- |",
    ]
    for c in all_cards:
        cid = c.get("id", "")
        base_id = cid.replace(" **missing**", "").replace("**missing**", "").strip()
        sr_no = date_sr_by_id.get(cid, "—")
        te_no = te_by_id.get(base_id)
        highlight = "**missing**" in cid
        compare_lines.append(
            _format_slim_compare_row(c, sr_no, te_no, highlight_missing=highlight)
        )
    compare_lines.append("")
    OUT_COMPARE.write_text("\n".join(compare_lines), encoding="utf-8")

    rope_te = te_by_id.get("rope")
    print(f"Wrote {OUT_BY_DATE} ({len(cards_by_date)} rows, date-sorted)")
    print(f"Wrote {OUT_COMPARE} (compare table, Sr-sorted, {len(teach)} wall + 2 extras)")
    print(f"Wrote {EMBED_JS}")
    print(f"VERIFY rope: Te_No={rope_te} Sr_No={date_sr_by_id.get('rope')}")


if __name__ == "__main__":
    main()
