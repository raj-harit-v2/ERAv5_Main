"""Quantize/dequantize for IEEE-like formats used in Session 10 + MBS/OAS lab."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class FormatSpec:
    name: str
    layout: str
    exp_bits: int
    mant_bits: int
    bias: int


@dataclass
class QuantResult:
    format_name: str
    layout: str
    stored_value: float
    abs_error_pct: float
    case: str
    bit_string: str = ""


SCALAR_SPECS: tuple[FormatSpec, ...] = (
    FormatSpec("fp32", "1+8+23", 8, 23, 127),
    FormatSpec("fp16", "1+5+10", 5, 10, 15),
    FormatSpec("bf16", "1+8+7", 8, 7, 127),
    FormatSpec("fp8 E4M3", "1+4+3", 4, 3, 7),
    FormatSpec("fp8 E5M2", "1+5+2", 5, 2, 15),
    FormatSpec("fp4 E2M1", "1+2+1", 2, 1, 1),
    FormatSpec("Custom", "1+6+9", 6, 9, 31),
)

# OCP MXFP4 E2M1 finite max: exp=3, mant=1 → 2^(3-1) * 1.5 = 6.0
E2M1_MAX = 6.0


def _fp32_pack(value: float) -> tuple[int, int, int, float]:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    stored = struct.unpack(">f", struct.pack(">I", bits))[0]
    return sign, exp, frac, stored


def max_finite(spec: FormatSpec) -> float:
    """Largest finite value (treat all-ones exp as finite for these mini formats)."""
    max_exp = (1 << spec.exp_bits) - 1
    unbiased = max_exp - spec.bias
    frac = (1 << spec.mant_bits) - 1
    return (1.0 + frac / (1 << spec.mant_bits)) * (2.0**unbiased)


def min_subnormal(spec: FormatSpec) -> float:
    return 2.0 ** (1 - spec.bias - spec.mant_bits)


def quantize_scalar(value: float, spec: FormatSpec) -> QuantResult:
    """Round ``value`` to the format and read it back."""
    if spec.name == "fp32":
        sign, exp, frac, stored = _fp32_pack(value)
        bits = f"{sign} {exp:08b} {frac:023b}"
        err = abs(stored - value) / abs(value) * 100.0 if value != 0 else 0.0
        return QuantResult(spec.name, spec.layout, stored, err, "Normal", bits)

    if spec.name == "bf16":
        sign, exp, frac32, _ = _fp32_pack(value)
        frac16 = frac32 >> 16
        stored = math.copysign(
            (1.0 + frac16 / (1 << 7)) * (2.0 ** (exp - 127)) if exp not in (0, 255) else 0.0,
            value,
        )
        if exp == 0:
            stored = 0.0
        bits = f"{sign} {exp:08b} {frac16:07b}"
        err = abs(stored - value) / abs(value) * 100.0 if value != 0 else 0.0
        return QuantResult(spec.name, spec.layout, stored, err, "Normal", bits)

    if value == 0.0:
        bits = _bit_string(0, 0, 0, spec)
        return QuantResult(spec.name, spec.layout, 0.0, 0.0, "Normal", bits)

    sign = 0 if value >= 0 else 1
    v = abs(value)
    max_v = max_finite(spec)
    min_sub = min_subnormal(spec)

    if v > max_v:
        stored = math.copysign(max_v, value)
        err = abs(stored - value) / abs(value) * 100.0
        bits = _bit_string(sign, (1 << spec.exp_bits) - 1, (1 << spec.mant_bits) - 1, spec)
        return QuantResult(spec.name, spec.layout, stored, err, "Clamped/Scaled", bits)

    if v < min_sub / 2.0:
        err = 100.0
        bits = _bit_string(sign, 0, 0, spec)
        return QuantResult(spec.name, spec.layout, 0.0, err, "Underflow to zero", bits)

    exp_unbiased = int(math.floor(math.log2(v)))
    exp_biased = exp_unbiased + spec.bias

    if exp_biased <= 0:
        # Subnormal: exponent stored as 0; mantissa holds v / 2^(1-bias)
        scale = 2.0 ** (1 - spec.bias)
        frac_int = int(round(v / scale * (1 << spec.mant_bits)))
        max_frac = 1 << spec.mant_bits
        if frac_int >= max_frac:
            stored = math.copysign(2.0 ** (1 - spec.bias), value)
            bits = _bit_string(sign, 1, 0, spec)
            err = abs(stored - value) / abs(value) * 100.0
            return QuantResult(spec.name, spec.layout, stored, err, "Normal", bits)
        stored = math.copysign(frac_int / max_frac * scale, value)
        if stored == 0.0:
            err = 100.0
            bits = _bit_string(sign, 0, 0, spec)
            return QuantResult(spec.name, spec.layout, 0.0, err, "Underflow to zero", bits)
        bits = _bit_string(sign, 0, frac_int, spec)
        err = abs(stored - value) / abs(value) * 100.0
        return QuantResult(spec.name, spec.layout, stored, err, "Normal", bits)

    max_exp = (1 << spec.exp_bits) - 1
    if exp_biased > max_exp:
        stored = math.copysign(max_v, value)
        bits = _bit_string(sign, max_exp, (1 << spec.mant_bits) - 1, spec)
        err = abs(stored - value) / abs(value) * 100.0
        return QuantResult(spec.name, spec.layout, stored, err, "Clamped/Scaled", bits)

    mantissa = v / (2.0**exp_unbiased)
    frac = mantissa - 1.0
    frac_int = int(round(frac * (1 << spec.mant_bits)))
    if frac_int >= (1 << spec.mant_bits):
        exp_biased += 1
        frac_int = 0
        if exp_biased > max_exp:
            stored = math.copysign(max_v, value)
            bits = _bit_string(sign, max_exp, (1 << spec.mant_bits) - 1, spec)
            err = abs(stored - value) / abs(value) * 100.0
            return QuantResult(spec.name, spec.layout, stored, err, "Clamped/Scaled", bits)

    stored = math.copysign(
        (1.0 + frac_int / (1 << spec.mant_bits)) * (2.0 ** (exp_biased - spec.bias)),
        value,
    )
    bits = _bit_string(sign, exp_biased, frac_int, spec)
    err = abs(stored - value) / abs(value) * 100.0 if value != 0 else 0.0
    return QuantResult(spec.name, spec.layout, stored, err, "Normal", bits)


def _bit_string(sign: int, exp: int, frac: int, spec: FormatSpec) -> str:
    return f"{sign} {exp:0{spec.exp_bits}b} {frac:0{spec.mant_bits}b}"


def quantize_e2m1(value: float) -> float:
    return quantize_scalar(value, SCALAR_SPECS[5]).stored_value


def quantize_e4m3(value: float) -> float:
    return quantize_scalar(value, SCALAR_SPECS[3]).stored_value


def encode_8mant_scale(value: float) -> float:
    """High-precision unsigned scale: 8 exponent bits (bias 127) + 8 mantissa bits."""
    spec = FormatSpec("E8M8-scale", "0+8+8", 8, 8, 127)
    return abs(quantize_scalar(abs(value), spec).stored_value)


def e8m0_pow2_scale(ratio: float) -> float:
    """MXFP4 shared scale: 2^ceil(log2(ratio)), stored as E8M0-style power of two."""
    if ratio <= 0:
        return 2.0 ** -127
    exp = math.ceil(math.log2(ratio))
    exp = max(-127, min(127, exp))
    return 2.0**exp
