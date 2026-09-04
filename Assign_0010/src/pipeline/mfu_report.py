"""Model FLOPs Utilization report (Session 10 Task 5)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from src.llm.mini_gpt import MiniGPT, MiniGPTConfig

# Peak BF16 tensor-core FLOP/s (theoretical, per device) — cite in report.
PEAK_TFLOPS_BY_DEVICE: dict[str, float] = {
    "cpu": 0.5,
    "cuda:T4": 65.0,
    "cuda:A100": 312.0,
    "cuda:H100": 989.0,
    "cuda:default": 100.0,
}


def _detect_peak_tflops(device: torch.device) -> tuple[float, str]:
    if device.type == "cpu":
        return PEAK_TFLOPS_BY_DEVICE["cpu"], "Conservative CPU estimate (0.5 TFLOP/s)"
    name = torch.cuda.get_device_name(device).upper() if device.type == "cuda" else ""
    if "T4" in name:
        return PEAK_TFLOPS_BY_DEVICE["cuda:T4"], f"NVIDIA {name} BF16 peak ~65 TFLOP/s (datasheet)"
    if "A100" in name:
        return PEAK_TFLOPS_BY_DEVICE["cuda:A100"], f"NVIDIA {name} BF16 peak ~312 TFLOP/s (datasheet)"
    if "H100" in name:
        return PEAK_TFLOPS_BY_DEVICE["cuda:H100"], f"NVIDIA {name} BF16 peak ~989 TFLOP/s (datasheet)"
    return PEAK_TFLOPS_BY_DEVICE["cuda:default"], f"Generic CUDA GPU estimate for {name or 'unknown'}"


def measure_mfu(
    warmup_steps: int = 5,
    measure_steps: int = 25,
    batch_size: int = 8,
    block_size: int = 128,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MiniGPTConfig(block_size=block_size)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_params = model.n_params
    tokens_per_step = batch_size * (block_size - 1)

    for _ in range(warmup_steps):
        batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device)
        opt.zero_grad()
        loss = model.loss_on_batch(batch)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(measure_steps):
        batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size), device=device)
        opt.zero_grad()
        loss = model.loss_on_batch(batch)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = tokens_per_step * measure_steps
    tokens_per_sec = total_tokens / elapsed
    achieved_tflops = 6.0 * n_params * tokens_per_sec / 1e12
    peak_tflops, peak_source = _detect_peak_tflops(device)
    mfu_pct = achieved_tflops / peak_tflops * 100.0
    target_pct = 40.0
    gap = target_pct - mfu_pct

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "n_params": n_params,
        "tokens_per_sec": tokens_per_sec,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_tflops,
        "peak_source": peak_source,
        "mfu_pct": mfu_pct,
        "target_pct": target_pct,
        "industry_band": "35-50%",
        "gap_to_target": gap,
    }


def write_mfu_report(out_path: Path, stats: dict) -> dict:
    n = stats["n_params"]
    tps = stats["tokens_per_sec"]
    achieved = stats["achieved_tflops"]
    peak = stats["peak_tflops"]
    mfu = stats["mfu_pct"]

    lines = [
        "# Session 10 — MFU Report (Task 5)",
        "",
        "## Formula",
        "",
        "$$\\text{MFU} = \\frac{6 \\times N \\times \\text{tokens/sec}}{\\text{peak TFLOP/s}}$$",
        "",
        "## Substituted values",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Model parameters $N$ | {n:,} |",
        f"| Tokens per second | {tps:,.1f} |",
        f"| Achieved TFLOP/s | {achieved:.4f} |",
        f"| Peak TFLOP/s | {peak:.1f} |",
        f"| Peak source | {stats['peak_source']} |",
        f"| **MFU %** | **{mfu:.2f}%** |",
        f"| Assignment target | {stats['target_pct']:.0f}% |",
        f"| Industry healthy band | {stats['industry_band']} |",
        f"| Gap to 40% target | {stats['gap_to_target']:.2f} percentage points |",
        "",
        "## Bottleneck narrative",
        "",
        f"On **{stats['gpu_name']}** ({stats['device']}), this small MiniGPT run is primarily "
        "**memory- and kernel-launch-bound**: micro-batch is modest, the model is tiny compared "
        "to production LLMs, and CPU-side work limits throughput when CUDA is unavailable. "
        "ERA V4 explicitly traded MFU for RAM via reversibility — the same loss curve "
        "can hide 4× compute waste (e.g. ~8% vs ~45% MFU). Long-context phases "
        "increase activation memory; changing batch semantics mid-curriculum (V4 mistake) "
        "would further distort averages without showing up in loss alone.",
        "",
        "## Long-context note",
        "",
        "When scaling 4K → 8K → … → 1M context, keep **global batch token count fixed per phase**. "
        "Longer sequences reduce micro-batch rows per GPU and increase accumulation steps, "
        "which typically lowers MFU unless checkpointing and parallelism are retuned.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out_path), **stats, "ok": mfu >= 0}
