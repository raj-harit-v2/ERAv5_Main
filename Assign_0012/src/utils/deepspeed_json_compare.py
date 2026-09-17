"""Map DeepSpeed-style ZeRO JSON configs to Session 12 accounting and diff vs lab CSV.

JSON selects ``zero_optimization.stage`` only; GiB comes from our formulas
(``zero_accounting``), compared to ``reports/assgn012_zero_table.csv``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils.zero_accounting import (
    FULL_LADDER_GIB,
    N_30B,
    stage_row,
)

STAGE_NAME = {1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
GIB_TOL = 0.2
BPP_TOL = 1e-2


def as_repo_rel(path: Path | str, root: Path) -> str:
    """Return a portable repo-relative path (forward slashes) for manifests/reports."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.name


@dataclass
class CompareRow:
    stage: int
    stage_name: str
    json_path: str
    implied_bpp: float
    implied_gib: float
    lab_gib: Optional[float]
    delta_gib: Optional[float]
    comm: str
    fit: str
    lab_fit: Optional[str]
    lab_comm: Optional[str]
    catalog_gib: Optional[float]
    verdict: str
    notes: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_from_ds_config(cfg: Dict[str, Any]) -> int:
    zo = cfg.get("zero_optimization") or {}
    stage = zo.get("stage")
    if stage is None:
        raise ValueError("missing zero_optimization.stage")
    stage_i = int(stage)
    if stage_i not in (1, 2, 3):
        raise ValueError(f"unsupported ZeRO stage: {stage_i}")
    return stage_i


def implied_accounting(stage: int, world_size: int = 32, n_params: float = N_30B) -> Dict[str, Any]:
    name = STAGE_NAME[stage]
    row = stage_row(name, world_size, n_params=n_params)
    return {
        "stage_name": name,
        "bytes_per_param": row.bytes_per_param,
        "gib_per_gpu": row.gib_per_gpu,
        "comm": row.comm,
        "fit": row.fit,
    }


def load_lab_w32_rows(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return stage_name -> row dict for world_size==32."""
    if not csv_path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["world_size"])) != 32:
                continue
            out[row["stage"]] = row
    return out


def compare_one(
    json_path: Path,
    lab_rows: Dict[str, Dict[str, Any]],
    world_size: int = 32,
    n_params: float = N_30B,
    catalog_expected: Optional[Dict[str, Any]] = None,
) -> CompareRow:
    cfg = load_json(json_path)
    stage = stage_from_ds_config(cfg)
    implied = implied_accounting(stage, world_size=world_size, n_params=n_params)
    name = implied["stage_name"]
    lab = lab_rows.get(name)
    lab_gib = float(lab["gib_per_gpu"]) if lab else None
    lab_fit = lab.get("fit") if lab else None
    lab_comm = lab.get("comm") if lab else None
    delta = abs(implied["gib_per_gpu"] - lab_gib) if lab_gib is not None else None

    catalog_gib = None
    if catalog_expected and "gib_per_gpu" in catalog_expected:
        catalog_gib = float(catalog_expected["gib_per_gpu"])

    problems: List[str] = []
    if lab is None:
        problems.append("lab CSV missing W=32 row")
    else:
        if delta is not None and delta > GIB_TOL:
            problems.append(f"GiB delta {delta:.3f} > {GIB_TOL}")
        if abs(float(lab["bytes_per_param"]) - implied["bytes_per_param"]) > BPP_TOL:
            problems.append("bytes/param mismatch vs lab")
        if lab_fit != implied["fit"]:
            problems.append(f"fit lab={lab_fit} vs implied={implied['fit']}")
        if lab_comm != implied["comm"]:
            problems.append(f"comm lab={lab_comm} vs implied={implied['comm']}")

    if catalog_expected:
        exp_bpp = float(catalog_expected.get("bytes_per_param", implied["bytes_per_param"]))
        exp_gib = float(catalog_expected.get("gib_per_gpu", implied["gib_per_gpu"]))
        if abs(exp_bpp - implied["bytes_per_param"]) > BPP_TOL:
            problems.append("catalog bytes/param mismatch")
        if abs(exp_gib - implied["gib_per_gpu"]) > GIB_TOL:
            problems.append("catalog GiB mismatch")
        if catalog_expected.get("comm") and catalog_expected["comm"] != implied["comm"]:
            problems.append("catalog comm mismatch")
        if catalog_expected.get("fit") and catalog_expected["fit"] != implied["fit"]:
            problems.append("catalog fit mismatch")

    # Also check FULL published ladder cell
    full_target = FULL_LADDER_GIB.get(world_size, {}).get(name)
    if full_target is not None and abs(implied["gib_per_gpu"] - full_target) > GIB_TOL:
        problems.append(f"FULL ladder target {full_target} mismatch")

    verdict = "PASS" if not problems else "FAIL"
    notes = "; ".join(problems) if problems else "stage maps to lab W=32 row within tolerance"
    return CompareRow(
        stage=stage,
        stage_name=name,
        json_path=str(json_path).replace("\\", "/"),
        implied_bpp=implied["bytes_per_param"],
        implied_gib=implied["gib_per_gpu"],
        lab_gib=lab_gib,
        delta_gib=delta,
        comm=implied["comm"],
        fit=implied["fit"],
        lab_fit=lab_fit,
        lab_comm=lab_comm,
        catalog_gib=catalog_gib,
        verdict=verdict,
        notes=notes,
    )


def compare_catalog(
    root: Path,
    catalog_path: Optional[Path] = None,
    lab_csv: Optional[Path] = None,
    world_size: int = 32,
) -> Dict[str, Any]:
    root = Path(root)
    catalog_path = catalog_path or (root / "configs" / "deepspeed" / "zero_stages_catalog.json")
    lab_csv = lab_csv or (root / "reports" / "assgn012_zero_table.csv")
    catalog = load_json(catalog_path)
    n_params = float(catalog.get("n_params", N_30B))
    w = int(catalog.get("world_size", world_size))
    lab_rows = load_lab_w32_rows(lab_csv)
    rows: List[CompareRow] = []
    for entry in catalog.get("configs", []):
        rel = entry["path"]
        json_path = root / rel
        rows.append(
            compare_one(
                json_path,
                lab_rows,
                world_size=w,
                n_params=n_params,
                catalog_expected=entry.get("expected"),
            )
        )
    all_ok = all(r.verdict == "PASS" for r in rows)
    return {
        "ok": all_ok,
        "rows": rows,
        "catalog_path": as_repo_rel(catalog_path, root),
        "lab_csv": as_repo_rel(lab_csv, root),
        "world_size": w,
        "n_params": n_params,
    }


def write_compare_report(reports_dir: Path, result: Dict[str, Any]) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "assgn012_zero_json_compare.md"
    rows: Sequence[CompareRow] = result["rows"]
    lines = [
        "# Assgn012 DeepSpeed JSON vs Lab Accounting",
        "",
        "JSON configs select `zero_optimization.stage` only. GiB is **computed** by Session 12 formulas",
        "and compared to `assgn012_zero_table.csv` at W=32. This is not measured DeepSpeed VRAM.",
        "",
        f"- Catalog: `{result['catalog_path']}`",
        f"- Lab CSV: `{result['lab_csv']}`",
        f"- Overlay: N={result['n_params']:.0e}, W={result['world_size']}",
        f"- Overall: **{'PASS' if result['ok'] else 'FAIL'}** (ok={result['ok']})",
        "",
        "| Stage | JSON | Implied GiB | Lab GiB | Δ | Comm | Fit | Verdict |",
        "| :--- | :--- | ---: | ---: | ---: | :---: | :--- | :--- |",
    ]
    for r in rows:
        lab_s = f"{r.lab_gib:.2f}" if r.lab_gib is not None else "—"
        d_s = f"{r.delta_gib:.3f}" if r.delta_gib is not None else "—"
        rel = Path(r.json_path).name
        lines.append(
            f"| {r.stage_name} | `{rel}` | {r.implied_gib:.2f} | {lab_s} | {d_s} | "
            f"{r.comm} | {r.fit} | **{r.verdict}** |"
        )
    lines.extend(
        [
            "",
            "## Details",
            "",
        ]
    )
    for r in rows:
        lines.append(
            f"- **{r.stage_name}** (`stage={r.stage}`): bpp={r.implied_bpp:.4f}; "
            f"lab_fit={r.lab_fit}; lab_comm={r.lab_comm}; {r.notes}"
        )
    lines.extend(
        [
            "",
            "## Formulas",
            "",
            "```text",
            "ZeRO-1: 4 + 12/W",
            "ZeRO-2: 2 + 14/W",
            "ZeRO-3: 16/W",
            "```",
            "",
            "## Note on root GPU_example.json",
            "",
            "Root `GPU_example.json` is a ZeRO-3 + CPU offload sketch (HF-style `auto` fields).",
            "It is **not** the compare baseline; use `configs/deepspeed/ds_zero{1,2,3}.json`.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_compare(root: Path) -> Dict[str, Any]:
    root = Path(root)
    result = compare_catalog(root)
    report = write_compare_report(root / "reports", result)
    result["report_path"] = as_repo_rel(report, root)
    result["ok_value"] = result["ok"]
    return result
