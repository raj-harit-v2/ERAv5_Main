"""Advertised Colab compute-unit rates (honest USD/hour)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SHADOW_T4_USD_PER_HOUR = 0.12
RATE_SOURCE_MCCORMICK_2026_03 = (
    "McCormick Colab GPU pricing 2026-03-10; 0.10 USD/unit"
)


@dataclass(frozen=True)
class RatePick:
    usd_per_hour: float
    rate_source: str
    gpu_slug: str
    shadow_usd_per_hour: float


def gpu_slug_from_name(gpu_name: str) -> str:
    g = (gpu_name or "cpu").lower()
    if "a100" in g and ("80" in g or "80gb" in g):
        return "a100_80"
    if "a100" in g:
        return "a100_40"
    if "l4" in g:
        return "l4"
    if "t4" in g:
        return "t4"
    if "h100" in g:
        return "h100"
    if "rtx" in g or "pro 6000" in g or "g4" in g:
        return "g4"
    if g in ("cpu", "mps"):
        return g
    slug = "".join(ch if ch.isalnum() else "_" for ch in g)
    slug = "_".join(p for p in slug.split("_") if p)[:24]
    return slug or "gpu"


def catalog_rate_for_gpu(gpu_name: str) -> tuple[float, str]:
    slug = gpu_slug_from_name(gpu_name)
    table = {
        "t4": (0.12, RATE_SOURCE_MCCORMICK_2026_03 + "; T4 1.19 u/h"),
        "l4": (0.17, RATE_SOURCE_MCCORMICK_2026_03 + "; L4 1.71 u/h"),
        "a100_40": (0.54, RATE_SOURCE_MCCORMICK_2026_03 + "; A100-40 5.40 u/h"),
        "a100_80": (0.75, RATE_SOURCE_MCCORMICK_2026_03 + "; A100-80 7.52 u/h"),
        "g4": (0.87, RATE_SOURCE_MCCORMICK_2026_03 + "; G4/RTX PRO 6000 8.71 u/h"),
        "h100": (0.0, "H100 Colab units/hr not measured; set ASSGN014_USD_PER_HOUR explicitly"),
        "cpu": (0.0, "local CPU; no cloud GPU billed"),
        "mps": (0.0, "local MPS; no cloud GPU billed"),
    }
    if slug in table:
        return table[slug]
    return (
        SHADOW_T4_USD_PER_HOUR,
        RATE_SOURCE_MCCORMICK_2026_03 + f"; unknown GPU={gpu_name!r} default T4 shadow",
    )


def resolve_usd_rate(
    *,
    host: str,
    gpu_name: str,
    env_usd: Optional[float] = None,
    free_colab: bool = False,
) -> RatePick:
    slug = gpu_slug_from_name(gpu_name)
    catalog_usd, catalog_src = catalog_rate_for_gpu(gpu_name)
    shadow = SHADOW_T4_USD_PER_HOUR

    if free_colab:
        return RatePick(
            usd_per_hour=0.0,
            rate_source="free Colab; advertised spend 0.00; usd_shadow uses T4 0.12",
            gpu_slug=slug,
            shadow_usd_per_hour=shadow,
        )

    if env_usd is not None and float(env_usd) > 0.0:
        return RatePick(
            usd_per_hour=float(env_usd),
            rate_source=f"ASSGN014_USD_PER_HOUR={env_usd}",
            gpu_slug=slug,
            shadow_usd_per_hour=shadow,
        )

    if host == "colab" and slug not in ("cpu", "mps"):
        return RatePick(
            usd_per_hour=catalog_usd if catalog_usd > 0 else shadow,
            rate_source=catalog_src,
            gpu_slug=slug,
            shadow_usd_per_hour=shadow,
        )

    if host in ("runpod", "cloud") and slug not in ("cpu", "mps"):
        return RatePick(
            usd_per_hour=catalog_usd if catalog_usd > 0 else shadow,
            rate_source=catalog_src + f"; host={host} (set ASSGN014_USD_PER_HOUR if pod rate differs)",
            gpu_slug=slug,
            shadow_usd_per_hour=shadow,
        )

    return RatePick(
        usd_per_hour=float(env_usd) if env_usd is not None else 0.0,
        rate_source=catalog_src if slug in ("cpu", "mps") else (
            f"ASSGN014_USD_PER_HOUR={env_usd}" if env_usd is not None else catalog_src
        ),
        gpu_slug=slug,
        shadow_usd_per_hour=shadow,
    )
