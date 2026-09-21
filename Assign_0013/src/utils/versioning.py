"""Versioned report directories under reports/ (PRIMARY filenames stay stable).

Layout:
  reports/
    LATEST.txt
    index.md
    v001_local_cpu_smoke/
      assgn013_*.png, csv, json, runs/, checkpoints/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

_VERSION_RE = re.compile(r"^v(\d{3})_")


def next_version_number(reports_root: Path) -> int:
    reports_root.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in reports_root.iterdir():
        if not p.is_dir():
            continue
        m = _VERSION_RE.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def make_version_id(
    *,
    host: str,
    gpu_slug: str,
    kind: str,
    n: Optional[int] = None,
    reports_root: Optional[Path] = None,
) -> str:
    kind = "full" if kind == "full" else "smoke"
    host_s = "".join(ch if ch.isalnum() else "_" for ch in (host or "local").lower())
    gpu_s = "".join(ch if ch.isalnum() else "_" for ch in (gpu_slug or "cpu").lower())
    if n is None:
        if reports_root is None:
            raise ValueError("reports_root required when n is None")
        n = next_version_number(reports_root)
    return f"v{n:03d}_{host_s}_{gpu_s}_{kind}"


def resolve_reports_dir(
    reports_root: Path,
    *,
    host: str,
    gpu_slug: str,
    smoke: bool,
    override_version: Optional[str] = None,
) -> Tuple[Path, str]:
    """Return (version_dir, version_id). Override resumes an existing folder."""
    reports_root.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if smoke else "full"
    if override_version:
        vid = override_version.strip()
        vdir = reports_root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir, vid
    vid = make_version_id(host=host, gpu_slug=gpu_slug, kind=kind, reports_root=reports_root)
    vdir = reports_root / vid
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir, vid


def write_latest(reports_root: Path, version_id: str) -> Path:
    reports_root.mkdir(parents=True, exist_ok=True)
    path = reports_root / "LATEST.txt"
    path.write_text(version_id.strip() + "\n", encoding="utf-8")
    return path


def read_latest(reports_root: Path) -> Optional[str]:
    path = reports_root / "LATEST.txt"
    if not path.is_file():
        return None
    line = path.read_text(encoding="utf-8").strip().splitlines()
    return line[0].strip() if line else None


def list_versions(reports_root: Path) -> List[str]:
    if not reports_root.is_dir():
        return []
    out = []
    for p in sorted(reports_root.iterdir()):
        if p.is_dir() and _VERSION_RE.match(p.name):
            out.append(p.name)
    return out


def update_index(reports_root: Path) -> Path:
    versions = list_versions(reports_root)
    latest = read_latest(reports_root) or ""
    lines = [
        "# Assgn013 report versions",
        "",
        f"LATEST = `{latest}`",
        "",
        "| Version | Path |",
        "| :--- | :--- |",
    ]
    for v in versions:
        marker = " (LATEST)" if v == latest else ""
        lines.append(f"| `{v}`{marker} | `reports/{v}/` |")
    lines.append("")
    path = reports_root / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def resolve_reports_for_eval(
    reports_root: Path,
    *,
    reports_arg: Optional[str] = None,
) -> Path:
    """CLI/eval: --reports path, else LATEST.txt, else reports_root itself."""
    if reports_arg:
        p = Path(reports_arg)
        candidates = [
            p if p.is_absolute() else Path.cwd() / p,
            reports_root / reports_arg,
            reports_root.parent / reports_arg,
        ]
        for c in candidates:
            c = c.resolve()
            if c.is_dir():
                return c
        # fall through: create/use as given relative to reports_root
        return (reports_root / reports_arg).resolve()
    latest = read_latest(reports_root)
    if latest:
        cand = reports_root / latest
        if cand.is_dir():
            return cand
    return reports_root
