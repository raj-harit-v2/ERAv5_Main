"""Versioned report directories under reports/ (PRIMARY filenames stay stable)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

_VERSION_RE = re.compile(r"^v(\d{3})_")
_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


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


def _slug_part(text: str, fallback: str = "x") -> str:
    s = _SLUG_SAFE.sub("_", (text or "").lower()).strip("_")
    return s or fallback


def budget_slug_from_tokens(tokens_dense: int, tokens_moe: int) -> str:
    """Short budget label from planned phase tokens (for folder names)."""
    total = max(0, int(tokens_dense)) + max(0, int(tokens_moe))
    # Named ladder points (50/50 splits used in Colab notebook).
    named = {
        8192: "4k",
        2_000_000: "2m",
        5_000_000: "5m",
        10_000_000: "10m",
        50_000_000: "50m",
    }
    for target, label in named.items():
        if abs(total - target) <= max(512, int(target * 0.02)):
            return label
    if total < 1_000_000:
        k = max(1, int(round(total / 1000.0)))
        return f"{k}k"
    m = total / 1_000_000.0
    if abs(m - round(m)) < 0.05:
        return f"{int(round(m))}m"
    return f"{m:.1f}m".replace(".", "p")


def make_version_id(
    *,
    host: str,
    gpu_slug: str,
    kind: str,
    n: Optional[int] = None,
    reports_root: Optional[Path] = None,
    budget_slug: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    kind = "full" if kind == "full" else "smoke"
    host_s = _slug_part(host, "local")
    gpu_s = _slug_part(gpu_slug, "cpu")
    if n is None:
        if reports_root is None:
            raise ValueError("reports_root required when n is None")
        n = next_version_number(reports_root)
    parts = [f"v{n:03d}", host_s, gpu_s]
    if budget_slug:
        parts.append(_slug_part(budget_slug, "x"))
    parts.append(kind)
    if tag:
        parts.append(_slug_part(tag, "tag"))
    return "_".join(parts)


def resolve_reports_dir(
    reports_root: Path,
    *,
    host: str,
    gpu_slug: str,
    smoke: bool,
    override_version: Optional[str] = None,
    budget_slug: Optional[str] = None,
    tag: Optional[str] = None,
) -> Tuple[Path, str]:
    reports_root.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if smoke else "full"
    if override_version:
        vid = override_version.strip()
        vdir = reports_root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        return vdir, vid
    vid = make_version_id(
        host=host,
        gpu_slug=gpu_slug,
        kind=kind,
        reports_root=reports_root,
        budget_slug=budget_slug,
        tag=tag,
    )
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
        "# Assgn014 report versions",
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
        return (reports_root / reports_arg).resolve()
    latest = read_latest(reports_root)
    if latest:
        cand = reports_root / latest
        if cand.is_dir():
            return cand
    return reports_root
