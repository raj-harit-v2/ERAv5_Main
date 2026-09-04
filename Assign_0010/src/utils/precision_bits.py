"""IEEE-style bit encoding for FP32, BF16, FP8 E4M3 (Session 10 Task 6)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from src.pipeline.quantize_compare import format_markdown_table1, table1_rows


@dataclass(frozen=True)
class FloatEncoding:
    format_name: str
    value_target: float
    sign_bit: str
    exponent_bits: str
    mantissa_bits: str
    full_bit_string: str
    stored_value: float
    truncation_error_pct: float


def _fp32_bits(value: float) -> FloatEncoding:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    stored = struct.unpack(">f", struct.pack(">I", bits))[0]
    err_pct = (stored - value) / value * 100.0 if value != 0 else 0.0
    return FloatEncoding(
        format_name="FP32",
        value_target=value,
        sign_bit=str(sign),
        exponent_bits=f"{exp:08b}",
        mantissa_bits=f"{frac:023b}",
        full_bit_string=f"{sign} {exp:08b} {frac:023b}",
        stored_value=stored,
        truncation_error_pct=err_pct,
    )


def _round_to_n_bits(frac: float, n_bits: int) -> int:
    """Round fractional part [0,1) to n_bits after implicit leading 1."""
    scale = 1 << n_bits
    return int(round(frac * scale)) & ((1 << n_bits) - 1)


def _decode_binary_float(sign: int, exp_unbiased: int, frac_int: int, frac_bits: int) -> float:
    if frac_int == 0 and exp_unbiased == -127:
        return 0.0
    mantissa = 1.0 + frac_int / (1 << frac_bits)
    return ((-1) ** sign) * (2.0**exp_unbiased) * mantissa


def _bf16_bits(value: float) -> FloatEncoding:
    fp32 = _fp32_bits(value)
    sign = int(fp32.sign_bit)
    exp = int(fp32.exponent_bits, 2)
    frac32 = int(fp32.mantissa_bits, 2)
    frac16 = frac32 >> 16
    stored = _decode_binary_float(sign, exp - 127, frac16, 7)
    err_pct = (stored - value) / value * 100.0 if value != 0 else 0.0
    return FloatEncoding(
        format_name="BF16",
        value_target=value,
        sign_bit=str(sign),
        exponent_bits=f"{exp:08b}",
        mantissa_bits=f"{frac16:07b}",
        full_bit_string=f"{sign} {exp:08b} {frac16:07b}",
        stored_value=stored,
        truncation_error_pct=err_pct,
    )


def _fp8_e4m3_bits(value: float) -> FloatEncoding:
    """FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa."""
    if value == 0.0:
        return FloatEncoding("FP8 E4M3", value, "0", "0000", "000", "0 0000 000", 0.0, 0.0)
    sign = 0 if value >= 0 else 1
    v = abs(value)
    import math

    exp_unbiased = math.floor(math.log2(v))
    mantissa = v / (2.0**exp_unbiased)
    frac = mantissa - 1.0
    exp_biased = exp_unbiased + 7
    if exp_biased < 0:
        exp_biased = 0
    if exp_biased > 15:
        exp_biased = 15
    frac_int = _round_to_n_bits(frac, 3)
    stored = _decode_binary_float(sign, exp_biased - 7, frac_int, 3)
    err_pct = (stored - value) / value * 100.0 if value != 0 else 0.0
    return FloatEncoding(
        format_name="FP8 E4M3",
        value_target=value,
        sign_bit=str(sign),
        exponent_bits=f"{exp_biased:04b}",
        mantissa_bits=f"{frac_int:03b}",
        full_bit_string=f"{sign} {exp_biased:04b} {frac_int:03b}",
        stored_value=stored,
        truncation_error_pct=err_pct,
    )


def encode_value(value: float) -> dict[str, FloatEncoding]:
    return {
        "fp32": _fp32_bits(value),
        "bf16": _bf16_bits(value),
        "fp8_e4m3": _fp8_e4m3_bits(value),
    }


def _format_block(enc: FloatEncoding) -> list[str]:
    return [
        f"### {enc.format_name} — target {enc.value_target}",
        f"- Sign: `{enc.sign_bit}`",
        f"- Exponent: `{enc.exponent_bits}`",
        f"- Mantissa: `{enc.mantissa_bits}`",
        f"- Full bits: `{enc.full_bit_string}`",
        f"- Stored decimal: `{enc.stored_value}`",
        f"- Truncation error: `{enc.truncation_error_pct:.6f}%`",
        "",
    ]


def write_precision_report(out_path: Path) -> dict:
    sections: list[str] = [
        "# Session 10 — Hand Bit Encoding (Task 6)",
        "",
        "## Part A — 0.1 (repeating binary fraction)",
        "",
    ]
    enc_01 = encode_value(0.1)
    for key in ("fp32", "bf16", "fp8_e4m3"):
        sections.extend(_format_block(enc_01[key]))

    sections.extend(["## Part B — 1.0 (terminating fraction)", ""])
    enc_10 = encode_value(1.0)
    for key in ("fp32", "bf16", "fp8_e4m3"):
        sections.extend(_format_block(enc_10[key]))

    sections.extend(
        [
            "## Training format recommendation",
            "",
            "Choose **BF16** for general LLM training because:",
            "- Same exponent range as FP32 (gradients survive to ~1e-38 scale).",
            "- No loss scaling required (unlike FP16).",
            "- FP8 E4M3 is viable for production with external scaling (production FP8 recipes, 2026) but shows",
            "  higher truncation on repeating fractions such as 0.1 (+1.56% in this table).",
            "",
        ]
    )

    sections.extend(
        [
            "## Part C — Full format comparison table (0.1 in a 32-element tensor)",
            "",
        ]
    )
    md_table = format_markdown_table1(table1_rows())
    # Drop the H1 from the standalone report; keep the table body.
    table_body = "\n".join(md_table.splitlines()[1:])
    sections.append(table_body.strip())
    sections.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")

    bf16 = enc_01["bf16"]
    return {
        "path": str(out_path),
        "bf16_bits_01": bf16.full_bit_string,
        "bf16_stored_01": bf16.stored_value,
        "bf16_error_pct": bf16.truncation_error_pct,
        "fp8_bits_01": enc_01["fp8_e4m3"].full_bit_string,
    }
