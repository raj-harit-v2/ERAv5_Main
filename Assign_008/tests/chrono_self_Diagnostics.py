"""
chrono_self_Diagnostics.py — diagnostics for demo_chrono (Assignment §18 wall).

Separate from Self_Diagionistics_02.py (frozen demo_Asgn_08 teaching demo).
Emits JSON under diagnostics_chrono/json/ and Chrono_self_Diagnostics.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.utils import ensure_dir

ROOT = config.ROOT
CHRONO = ROOT / "demo_chrono"
DEMO = ROOT / "demo_Asgn_08"
OUT = ROOT / "diagnostics_chrono"
JSON_DIR = OUT / "json"
REPORT = ROOT / "diagnostics_02" / "Chrono_self_Diagnostics.md"

VERIFY = "VERIFY_PRIMARY_SOURCE"
REQUIRED_FIELDS = (
    "id",
    "title",
    "year_display",
    "year_sort",
    "problem",
    "mechanism",
    "buy",
    "give_up",
    "when",
    "source",
    "widget_iframe",
    "story_prev_id",
)
FORBIDDEN_GIVE_UP = ("no downside", "no meaningful downside", "none")
MIN_GIVE_UP_LEN = 12


@dataclass
class StepResult:
    step_id: str
    title: str
    ok: bool
    severity: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(step: StepResult) -> None:
    path = JSON_DIR / f"{step.step_id}.json"
    path.write_text(json.dumps({"generated_at": _utc(), **asdict(step)}, indent=2), encoding="utf-8")


def load_chrono() -> dict[str, Any]:
    return json.loads((CHRONO / "data" / "chronology.json").read_text(encoding="utf-8"))


def load_required() -> list[str]:
    return json.loads((CHRONO / "data" / "required_mechanisms.json").read_text(encoding="utf-8"))


def sort_cards_python(data: dict[str, Any]) -> list[dict[str, Any]]:
    order = data.get("provisional_order") or []
    rank = {cid: i for i, cid in enumerate(order)}
    cards = list(data.get("cards") or [])

    def key(c: dict[str, Any]) -> tuple[int, int]:
        ys = c.get("year_sort")
        if ys is None:
            ys = 99999999
        return (int(ys), rank.get(c.get("id", ""), 9999))

    return sorted(cards, key=key)


def step_c00_scaffold() -> StepResult:
    paths = {
        "index": CHRONO / "index.html",
        "chrono_json": CHRONO / "data" / "chronology.json",
        "required_json": CHRONO / "data" / "required_mechanisms.json",
        "embed_js": CHRONO / "data" / "chronology.embed.js",
        "app_js": CHRONO / "scripts" / "chrono_app.js",
        "freeze_hashes": DEMO / "freeze_hashes.json",
        "sources_by_date": CHRONO / "CHRONOLOGY_SOURCES_BY_DATE.md",
        "sources_compare": CHRONO / "CHRONOLOGY_Compare.md",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    ok = not missing
    return StepResult(
        "C00_scaffold",
        "Chronology app scaffold",
        ok,
        "pass" if ok else "fail",
        "All demo_chrono scaffold files present" if ok else f"missing: {missing}",
        {"paths": {k: str(v) for k, v in paths.items()}, "missing": missing},
        [],
    )


def step_c01_schema() -> StepResult:
    data = load_chrono()
    ok_schema = data.get("schema", "").startswith("a08.chronology")
    cards = data.get("cards") or []
    findings: list[str] = []
    for c in cards:
        cid = c.get("id", "?")
        for f in REQUIRED_FIELDS:
            if f not in c:
                findings.append(f"{cid}: missing field {f}")
        src = c.get("source")
        if not isinstance(src, dict):
            findings.append(f"{cid}: source must be object")
        elif not all(
            k in src
            for k in (
                "label",
                "url",
                "citation",
                "date_verified",
                "date_note",
                "pdf_verified",
                "pdf_v1",
            )
        ):
            findings.append(f"{cid}: incomplete source object")
    ok = ok_schema and not findings
    return StepResult(
        "C01_schema",
        "Chronology JSON schema",
        ok,
        "pass" if ok else "fail",
        f"{len(cards)} cards; schema ok={ok_schema}",
        {"schema": data.get("schema"), "n_cards": len(cards)},
        findings[:20],
    )


def step_c02_card_count() -> StepResult:
    required = load_required()
    data = load_chrono()
    ids = {c.get("id") for c in data.get("cards") or []}
    req_set = set(required)
    missing = sorted(req_set - ids)
    extra = sorted(ids - req_set)
    ok = not missing and not extra and len(ids) == len(required)
    return StepResult(
        "C02_card_count",
        "Required mechanism coverage",
        ok,
        "pass" if ok else "fail",
        f"{len(ids)}/{len(required)} required ids",
        {"required_n": len(required), "card_n": len(ids), "missing": missing, "extra": extra},
        missing + [f"+extra:{e}" for e in extra],
    )


def step_c03_trade_honesty() -> StepResult:
    data = load_chrono()
    findings: list[str] = []
    for c in data.get("cards") or []:
        cid = c.get("id", "?")
        for field_name in ("buy", "give_up", "when"):
            val = (c.get(field_name) or "").strip()
            if not val:
                findings.append(f"{cid}: empty {field_name}")
        give = (c.get("give_up") or "").lower()
        if len(give) < MIN_GIVE_UP_LEN:
            findings.append(f"{cid}: give_up too short")
        if any(p in give for p in FORBIDDEN_GIVE_UP):
            findings.append(f"{cid}: forbidden give_up phrase")
    ok = not findings
    return StepResult(
        "C03_trade_honesty",
        "Buy / give up / when honesty",
        ok,
        "pass" if ok else "fail",
        "All cards have honest trade-offs" if ok else f"{len(findings)} issues",
        {"issues": len(findings)},
        findings[:20],
    )


def step_c04_sources() -> StepResult:
    data = load_chrono()
    findings: list[str] = []
    for c in data.get("cards") or []:
        cid = c.get("id", "?")
        src = c.get("source") or {}
        if not (src.get("label") or "").strip():
            findings.append(f"{cid}: empty source.label")
        if "date_verified" not in src:
            findings.append(f"{cid}: missing date_verified")
        if not src.get("date_verified"):
            if c.get("year_display") != VERIFY:
                findings.append(f"{cid}: unverified but year_display != VERIFY")
        else:
            if c.get("year_display") == VERIFY:
                findings.append(f"{cid}: verified but year_display still VERIFY")
            if not (src.get("url") or "").strip():
                findings.append(f"{cid}: verified but empty source.url")
            if not (src.get("pdf_verified") or "").strip():
                findings.append(f"{cid}: verified but empty pdf_verified")
            if not (src.get("pdf_v1") or "").strip():
                findings.append(f"{cid}: verified but empty pdf_v1")
            ys = c.get("year_sort")
            if not isinstance(ys, int) or ys < 10000101 or ys > 99991231:
                findings.append(f"{cid}: year_sort should be YYYYMMDD, got {ys}")
    ok = not findings
    return StepResult(
        "C04_sources",
        "Source fields + verified stamp discipline",
        ok,
        "pass" if ok else "fail",
        "Source fields + verified stamp discipline",
        {"issues": len(findings)},
        findings[:20],
    )


def step_c05_story_fields() -> StepResult:
    data = load_chrono()
    ids = {c.get("id") for c in data.get("cards") or []}
    findings: list[str] = []
    for c in data.get("cards") or []:
        cid = c.get("id", "?")
        if not (c.get("problem") or "").strip():
            findings.append(f"{cid}: empty problem")
        if not (c.get("mechanism") or "").strip():
            findings.append(f"{cid}: empty mechanism")
        prev = c.get("story_prev_id")
        if prev is not None and prev not in ids:
            findings.append(f"{cid}: bad story_prev_id {prev}")
    ok = not findings
    return StepResult(
        "C05_story_fields",
        "Story spine fields",
        ok,
        "pass" if ok else "fail",
        "problem + mechanism present" if ok else f"{len(findings)} issues",
        {},
        findings[:20],
    )


def step_c06_widget_paths() -> StepResult:
    data = load_chrono()
    findings: list[str] = []
    checked = 0
    for c in data.get("cards") or []:
        w = c.get("widget_iframe")
        if not w:
            continue
        checked += 1
        path = (CHRONO / w).resolve()
        if not path.is_file():
            findings.append(f"{c.get('id')}: missing widget {w}")
    ok = not findings
    return StepResult(
        "C06_widget_paths",
        "Widget iframe paths",
        ok,
        "pass" if ok else "fail",
        f"{checked} iframes checked" if ok else f"{len(findings)} broken paths",
        {"iframes_checked": checked},
        findings,
    )


def step_c07_html_wiring() -> StepResult:
    html = (CHRONO / "index.html").read_text(encoding="utf-8")
    js = (CHRONO / "scripts" / "chrono_app.js").read_text(encoding="utf-8")
    ok = (
        'id="timeline"' in html
        and "chrono_app.js" in html
        and "chronology.embed.js" in html
        and "CHRONOLOGY_DATA" in js
        and "sortCards" in js
        and "demo_Asgn_08" in html
    )
    return StepResult(
        "C07_html_wiring",
        "HTML + JS wiring",
        ok,
        "pass" if ok else "fail",
        "timeline + chrono_app.js linked" if ok else "wiring incomplete",
        {"has_timeline": 'id="timeline"' in html},
        [],
    )


def step_c08_sort_order() -> StepResult:
    data = load_chrono()
    sorted_cards = sort_cards_python(data)
    ids = [c.get("id") for c in sorted_cards]
    js = (CHRONO / "scripts" / "chrono_app.js").read_text(encoding="utf-8")
    ok_js = "provisional_order" in js and "year_sort" in js
    ok = ok_js and len(ids) == len(data.get("cards") or [])
    return StepResult(
        "C08_sort_order",
        "Sort: YYYYMMDD year_sort then provisional_order",
        ok,
        "pass" if ok else "fail",
        f"sorted {len(ids)} cards; JS spec present={ok_js}",
        {"first_id": ids[0] if ids else None, "last_id": ids[-1] if ids else None},
        [],
    )


def step_c09_pytest() -> StepResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_chronology_cards.py", "-q", "--tb=no"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        ok = proc.returncode == 0
        return StepResult(
            "C09_pytest",
            "Chronology pytest",
            ok,
            "pass" if ok else "fail",
            "pytest tests/test_chronology_cards.py",
            {"returncode": proc.returncode, "tail": out[-400:]},
            out.splitlines()[-6:],
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult("C09_pytest", "Chronology pytest", False, "fail", str(exc), {}, [])


def step_c10_freeze_integrity() -> StepResult:
    freeze_path = DEMO / "freeze_hashes.json"
    if not freeze_path.is_file():
        return StepResult(
            "C10_freeze_integrity",
            "demo_Asgn_08 freeze hashes",
            False,
            "fail",
            "freeze_hashes.json missing",
            {},
            [],
        )
    expected = json.loads(freeze_path.read_text(encoding="utf-8")).get("files") or {}
    findings: list[str] = []
    for rel, want in expected.items():
        p = ROOT / rel.replace("/", "\\") if "\\" not in rel else ROOT / rel
        if not p.is_file():
            findings.append(f"missing {rel}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            findings.append(f"hash drift {rel}")
    ok = not findings
    return StepResult(
        "C10_freeze_integrity",
        "Frozen demo_Asgn_08 core files",
        ok,
        "pass" if ok else "fail",
        "hashes match freeze snapshot" if ok else f"{len(findings)} drift",
        {"files_checked": len(expected)},
        findings,
    )


def step_c11_scorecard(steps: list[StepResult]) -> StepResult:
    fails = [s.step_id for s in steps if not s.ok]
    ok = not fails
    return StepResult(
        "C11_scorecard",
        "Chronology readiness",
        ok,
        "pass" if ok else "fail",
        "READY" if ok else f"NOT_READY fails={fails}",
        {"fails": fails, "n": len(steps)},
        [],
    )


def write_report(steps: list[StepResult], summary: dict[str, Any]) -> None:
    lines = [
        "# Chrono_self_Diagnostics — demo_chrono",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        "```text",
        "C00 scaffold → C01 schema → C02 card count → C03 trade honesty",
        "            → C04 sources → C05 story → C06 widgets → C07 wiring",
        "            → C08 sort → C09 pytest → C10 freeze → C11 scorecard",
        "```",
        "",
        "Independent of Diag_02 (frozen `demo_Asgn_08` teaching demo).",
        "",
    ]
    for s in steps:
        lines += [
            f"## {s.step_id} — {s.title}",
            "",
            f"- **Result:** `{s.severity}` (ok={s.ok})",
            f"- **Summary:** {s.summary}",
            f"- **JSON:** `diagnostics_chrono/json/{s.step_id}.json`",
            "",
        ]
        if s.findings:
            lines.append("- **Findings:**")
            for f in s.findings[:10]:
                lines.append(f"  - {f}")
            lines.append("")
    ensure_dir(REPORT.parent)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(JSON_DIR)
    print("=== chrono_self_Diagnostics — demo_chrono ===")
    steps: list[StepResult] = []
    for fn in (
        step_c00_scaffold,
        step_c01_schema,
        step_c02_card_count,
        step_c03_trade_honesty,
        step_c04_sources,
        step_c05_story_fields,
        step_c06_widget_paths,
        step_c07_html_wiring,
        step_c08_sort_order,
        step_c09_pytest,
        step_c10_freeze_integrity,
    ):
        s = fn()
        steps.append(s)
        write_json(s)
        tag = "PASS" if s.ok else "FAIL"
        print(f"[{tag}] {s.step_id}: {s.summary}")

    score = step_c11_scorecard(steps)
    steps.append(score)
    write_json(score)

    summary = {"generated_at": _utc(), "verdict": score.summary}
    write_report(steps, summary)
    print(f"Wrote {REPORT.name}")
    print(f"Verdict: {score.summary}")
    return 0 if score.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
