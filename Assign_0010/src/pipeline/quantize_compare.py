"""Compare scalar and block-scaled low-precision formats, including MBS/OAS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.float_formats import (
    E2M1_MAX,
    SCALAR_SPECS,
    QuantResult,
    e8m0_pow2_scale,
    encode_8mant_scale,
    quantize_e2m1,
    quantize_e4m3,
    quantize_scalar,
)

TARGET = 0.1
OUTLIER = 5.0


def make_tensor(n: int) -> np.ndarray:
    x = np.full(n, TARGET, dtype=np.float64)
    x[0] = TARGET
    x[1] = OUTLIER
    return x


def _err_pct(stored: float, target: float = TARGET) -> float:
    if target == 0:
        return 0.0
    return abs(stored - target) / abs(target) * 100.0


def _case_from_stored(stored: float, used_scale: bool) -> str:
    if stored == 0.0:
        return "Underflow to zero"
    if used_scale:
        return "Clamped/Scaled"
    return "Normal"


def quantize_int8_tensor(x: np.ndarray) -> QuantResult:
    max_val = float(np.max(np.abs(x)))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127)
    recon = q * scale
    stored = float(recon[0])
    return QuantResult(
        format_name="int8",
        layout="8b + shared scale max/127",
        stored_value=stored,
        abs_error_pct=_err_pct(stored),
        case=_case_from_stored(stored, True),
        bit_string=f"scale={scale:.6g}",
    )


def _mxfp4_block(block: np.ndarray, *, isolate_outlier: bool) -> np.ndarray:
    """Dequantize one MXFP4 block (E2M1 + E8M0 power-of-two scale)."""
    abs_b = np.abs(block)
    if isolate_outlier and abs_b.size > 1:
        body_max = float(np.partition(abs_b, -2)[-2])
        max_b = body_max if body_max > 0 else float(np.max(abs_b))
    else:
        max_b = float(np.max(abs_b))
    scale = e8m0_pow2_scale(max_b / E2M1_MAX)
    recon = np.empty_like(block)
    for i, v in enumerate(block):
        recon[i] = quantize_e2m1(float(v) / scale) * scale
    return recon


def quantize_mxfp4(x: np.ndarray, *, isolate_outlier: bool = False) -> np.ndarray:
    out = np.empty_like(x)
    for start in range(0, len(x), 32):
        sl = slice(start, start + 32)
        out[sl] = _mxfp4_block(x[sl], isolate_outlier=isolate_outlier)
    return out


def _nvfp4_block(block: np.ndarray) -> np.ndarray:
    max_b = float(np.max(np.abs(block)))
    scale = quantize_e4m3(max_b / E2M1_MAX)
    if scale == 0.0:
        scale = min_nonzero_e4m3()
    recon = np.empty_like(block)
    for i, v in enumerate(block):
        recon[i] = quantize_e2m1(float(v) / scale) * scale
    return recon


def min_nonzero_e4m3() -> float:
    return quantize_e4m3(2.0 ** (1 - 7 - 3))


def quantize_nvfp4(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for start in range(0, len(x), 16):
        sl = slice(start, start + 16)
        out[sl] = _nvfp4_block(x[sl])
    return out


def apply_mbs(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Macro-block scale with 8 mantissa bits; land outlier near OAS sweet spot."""
    max_abs = float(np.max(np.abs(x)))
    target_peak = 3.25  # midpoint of OAS band [3.0, 3.5]
    raw = max_abs / target_peak if target_peak > 0 else max_abs
    macro = encode_8mant_scale(raw)
    if macro == 0.0:
        macro = 1.0
    return x / macro, macro


def quantize_mxfp4_mbs(x: np.ndarray) -> np.ndarray:
    pre, macro = apply_mbs(x)
    recon_pre = quantize_mxfp4(pre, isolate_outlier=True)
    return recon_pre * macro


def _oas_micro_block(block: np.ndarray) -> np.ndarray:
    """1x16 OAS: if max in [3.0, 3.5], double values to use full E2M1 range."""
    max_b = float(np.max(np.abs(block)))
    boost = 2.0 if 3.0 <= max_b <= 3.5 else 1.0
    boosted = block * boost
    abs_b = np.abs(boosted)
    if abs_b.size > 1:
        body_max = float(np.partition(abs_b, -2)[-2])
        max_use = body_max if body_max > 0 else float(np.max(abs_b))
    else:
        max_use = float(np.max(abs_b))
    scale = e8m0_pow2_scale(max_use / E2M1_MAX)
    recon = np.empty_like(block)
    for i, v in enumerate(boosted):
        recon[i] = quantize_e2m1(float(v) / scale) * scale
    return recon / boost


def quantize_mxfp4_mbs_oas(x: np.ndarray) -> np.ndarray:
    pre, macro = apply_mbs(x)
    out_pre = np.empty_like(pre)
    for start in range(0, len(pre), 16):
        sl = slice(start, start + 16)
        out_pre[sl] = _oas_micro_block(pre[sl])
    return out_pre * macro


def table1_rows() -> list[QuantResult]:
    x32 = make_tensor(32)
    rows: list[QuantResult] = []
    for spec in SCALAR_SPECS:
        q = quantize_scalar(TARGET, spec)
        rows.append(q)

    rows.append(quantize_int8_tensor(x32))

    mx = quantize_mxfp4(x32, isolate_outlier=False)
    stored = float(mx[0])
    rows.append(
        QuantResult(
            "MXFP4",
            "32-el E2M1 + E8M0",
            stored,
            _err_pct(stored),
            _case_from_stored(stored, True),
        )
    )

    nv = quantize_nvfp4(x32)
    stored_nv = float(nv[0])
    rows.append(
        QuantResult(
            "NVFP4",
            "16-el E2M1 + E4M3",
            stored_nv,
            _err_pct(stored_nv),
            _case_from_stored(stored_nv, True),
        )
    )

    # Merged Table 2 methods (fp32 / Standard MXFP4 already above); 128-el tensor for MBS/OAS.
    x128 = make_tensor(128)
    mbs_stored = float(quantize_mxfp4_mbs(x128)[0])
    rows.append(
        QuantResult(
            "MXFP4 + MBS",
            "128-el MBS + 32-el E2M1/E8M0",
            mbs_stored,
            _err_pct(mbs_stored),
            "Rescued!" if mbs_stored != 0.0 else "Underflow to zero",
        )
    )
    oas_stored = float(quantize_mxfp4_mbs_oas(x128)[0])
    rows.append(
        QuantResult(
            "MXFP4 + MBS + OAS",
            "MBS + 16-el OAS",
            oas_stored,
            _err_pct(oas_stored),
            "Rescued!" if oas_stored != 0.0 else "Underflow to zero",
        )
    )
    return rows


def table2_rows() -> list[dict[str, Any]]:
    x128 = make_tensor(128)
    fp32 = float(x128[0])
    mx = float(quantize_mxfp4(x128, isolate_outlier=False)[0])
    mbs = float(quantize_mxfp4_mbs(x128)[0])
    oas = float(quantize_mxfp4_mbs_oas(x128)[0])

    def status(name: str, stored: float) -> str:
        if name == "fp32":
            return "Normal"
        if stored == 0.0:
            return "Underflow to zero"
        if name.endswith("OAS") and stored != 0.0:
            return "Rescued!"
        if name.endswith("MBS") and stored != 0.0:
            return "Rescued!"
        return "Normal"

    methods = [
        ("fp32 (Reference)", fp32, False),
        ("Standard MXFP4", mx, False),
        ("MXFP4 + MBS", mbs, False),
        ("MXFP4 + MBS + OAS", oas, True),
    ]
    out = []
    for name, stored, _ in methods:
        out.append(
            {
                "method": name,
                "reads_back": stored,
                "error_pct": _err_pct(stored),
                "status": status(name, stored),
            }
        )
    return out


def format_ascii_table1(rows: list[QuantResult]) -> str:
    headers = ("Format Name", "Bit Layout", "Quantized Value", "Abs Error %", "Simulation Case")
    data = [
        (
            r.format_name,
            r.layout,
            f"{r.stored_value:.8g}",
            f"{r.abs_error_pct:.4f}",
            r.case,
        )
        for r in rows
    ]
    return _ascii_table(headers, data)


def format_ascii_table2(rows: list[dict[str, Any]]) -> str:
    headers = ("Quantization Method", "Reads Back (0.1)", "Error %", "Status")
    data = [
        (
            r["method"],
            f"{r['reads_back']:.8g}",
            f"{r['error_pct']:.4f}",
            r["status"],
        )
        for r in rows
    ]
    return _ascii_table(headers, data)


def _ascii_table(headers: tuple[str, ...], data: list[tuple[str, ...]]) -> str:
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep, fmt(headers), sep]
    for row in data:
        lines.append(fmt(row))
    lines.append(sep)
    return "\n".join(lines)


def format_markdown_table1(rows: list[QuantResult]) -> str:
    lines = [
        "# FP format comparison (0.1 at index 0)",
        "",
        "Merged view: scalar/block formats use a **32-element** tensor "
        "(index 0 = 0.1, index 1 = 5.0, rest = 0.1). "
        "**MXFP4 + MBS** and **MXFP4 + MBS + OAS** use the **128-element** tensor "
        "from the MBS/OAS rescue demo. fp32 and Standard MXFP4 already appear above.",
        "",
        "| Format Name | Bit Layout | Quantized Value | Abs Error % | Simulation Case |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.format_name} | `{r.layout}` | {r.stored_value:.8g} | {r.abs_error_pct:.4f} | {r.case} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_markdown_table2(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MXFP4 MBS / OAS rescue (0.1 at index 0)",
        "",
        "Tensor: 128 elements, index 0 = 0.1, index 1 = 5.0, rest = 0.1.",
        "",
        "| Quantization Method | Reads Back (0.1) | Error % | Status |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['reads_back']:.8g} | {r['error_pct']:.4f} | {r['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_quantize_compare(reports_dir: Path) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows1 = table1_rows()
    rows2 = table2_rows()
    ascii1 = format_ascii_table1(rows1)
    ascii2 = format_ascii_table2(rows2)

    p1 = reports_dir / "assgn010_fp_format_table.md"
    p2 = reports_dir / "assgn010_mbs_oas_table.md"
    p1.write_text(format_markdown_table1(rows1), encoding="utf-8")
    p2.write_text(format_markdown_table2(rows2), encoding="utf-8")

    names = [r.format_name for r in rows1]
    mx_row = next(r for r in rows1 if r.format_name == "MXFP4")
    oas_row = next(r for r in rows2 if "OAS" in r["method"])
    mbs_row = next(r for r in rows2 if r["method"] == "MXFP4 + MBS")

    return {
        "path": str(p1),
        "mbs_oas_path": str(p2),
        "ascii_table1": ascii1,
        "ascii_table2": ascii2,
        "format_names": names,
        "mxfp4_underflow": mx_row.stored_value == 0.0,
        "mbs_reads_back": mbs_row["reads_back"],
        "oas_reads_back": oas_row["reads_back"],
        "oas_status": oas_row["status"],
        "ok": len(names) == 12 and mx_row.stored_value == 0.0 and oas_row["reads_back"] != 0.0,
    }
