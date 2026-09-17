"""Assign_0012 ZeRO lab orchestrator: accounting, demo DP, graphics, reports."""

from __future__ import annotations

import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.llm.demo_model import DemoConfig, TinyDecoder, cross_entropy_loss, synthetic_batch
from src.pipeline.zero_graphics import generate_all_graphics
from src.pipeline.zero_shard_sim import plan_for_stage, simulate_zero_memory_table
from src.utils.deepspeed_json_compare import as_repo_rel, run_compare
from src.utils.dist_helpers import (
    barrier,
    destroy_distributed,
    init_distributed,
    max_abs_param_diff_vs_rank0,
    peak_rss_mb,
    smoke_all_reduce,
    all_reduce_mean_,
)
from src.utils.zero_accounting import (
    B200_COMPUTE_S,
    CARD_GIB,
    FULL_BYTES_W8,
    FULL_LADDER_GIB,
    H100_COMPUTE_S,
    IB_2P_S,
    N_30B,
    bytes_per_param,
    gib_from_bytes_per_param,
    ladder_as_dicts,
    memory_ladder,
    replicated_floor_gib,
)


STAGES = ("DP", "ZeRO-1", "ZeRO-2", "ZeRO-3")


def _is_rank0(ctx) -> bool:
    return (not ctx.is_distributed) or ctx.rank == 0


def write_rank_smoke(reports: Path, ctx, smoke_val: float, smoke_ok: bool) -> Path:
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "assgn012_rank_smoke.txt"
    lines = [
        f"hostname={ctx.hostname}",
        f"rank={ctx.rank}",
        f"world_size={ctx.world_size}",
        f"backend={ctx.backend}",
        f"is_distributed={ctx.is_distributed}",
        f"smoke_all_reduce_sum={smoke_val}",
        f"smoke_ok={smoke_ok}",
        f"overlay_world_size=32",
        f"note=Analytic overlay always uses W=32 even if process group is smaller.",
    ]
    # Rank 0 writes; under distributed each rank appends via gather-style file is tricky —
    # write per-rank sidecar then rank0 merges if needed.
    if ctx.is_distributed and ctx.world_size > 1:
        side = reports / f"assgn012_rank_smoke_r{ctx.rank}.txt"
        side.write_text("\n".join(lines) + "\n", encoding="utf-8")
        barrier()
        if ctx.rank == 0:
            merged = ["# Assgn012 rank smoke (merged)", ""]
            for r in range(ctx.world_size):
                sp = reports / f"assgn012_rank_smoke_r{r}.txt"
                if sp.is_file():
                    merged.append(f"--- rank {r} ---")
                    merged.append(sp.read_text(encoding="utf-8").rstrip())
                    merged.append("")
            merged.append(f"smoke_gate_ok={smoke_ok}")
            path.write_text("\n".join(merged) + "\n", encoding="utf-8")
        barrier()
    else:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_dp_baseline(reports: Path, ctx, steps: int = 3) -> Dict[str, Any]:
    cfg = DemoConfig()
    device = torch.device("cpu")
    model = TinyDecoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    n_params = model.count_parameters()

    # Broadcast initial weights from rank 0 so replicas start identical
    if ctx.is_distributed:
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

    losses: List[float] = []
    for step in range(steps):
        x, y = synthetic_batch(cfg, rank=ctx.rank, device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        # DP: average gradients
        if ctx.is_distributed:
            for p in model.parameters():
                if p.grad is not None:
                    all_reduce_mean_(p.grad)
        opt.step()
        losses.append(float(loss.detach().item()))

    max_diff = max_abs_param_diff_vs_rank0(model)
    rss = peak_rss_mb()
    bpp_dp = 16.0
    local_bytes = bpp_dp * n_params

    result = {
        "ok": (max_diff < 1e-5) if ctx.is_distributed else True,
        "n_params": n_params,
        "steps": steps,
        "losses": losses,
        "max_abs_param_diff_vs_rank0": max_diff,
        "peak_rss_mb": rss,
        "dp_local_bytes": local_bytes,
        "dp_local_mib": local_bytes / (1024**2),
        "world_size": ctx.world_size,
        "is_distributed": ctx.is_distributed,
    }

    if _is_rank0(ctx):
        reports.mkdir(parents=True, exist_ok=True)
        md = reports / "assgn012_dp_baseline.md"
        md.write_text(
            "\n".join(
                [
                    "# Assgn012 DP Baseline",
                    "",
                    f"- Demo parameters N_demo = **{n_params}**",
                    f"- Process world_size = **{ctx.world_size}** (distributed={ctx.is_distributed})",
                    f"- Steps = {steps}; losses = {', '.join(f'{x:.4f}' for x in losses)}",
                    f"- Max |w_i - w_0| after sync = **{max_diff:.3e}** (expect ~0 under DP)",
                    f"- Peak RSS (rank0) ≈ **{rss:.1f} MiB**",
                    f"- DP local training state ≈ 16 × N_demo = **{local_bytes / (1024**2):.4f} MiB**",
                    "",
                    "Averaging gradients keeps replicas identical (Session 12 Data Parallelism).",
                    "30B numbers are **not** loaded here — see overlay tables.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        result["report_path"] = as_repo_rel(md, reports.parent)
    return result


def write_zero_tables(reports: Path) -> Dict[str, Any]:
    reports.mkdir(parents=True, exist_ok=True)
    root = reports.parent
    rows = ladder_as_dicts(memory_ladder())
    csv_path = reports / "assgn012_zero_table.csv"
    tmp_csv = reports / "assgn012_zero_table.csv.tmp"
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        tmp_csv.replace(csv_path)
    except PermissionError:
        # Windows file lock (e.g. CSV open in Excel) — keep .tmp as readable fallback.
        alt = reports / "assgn012_zero_table_unlocked.csv"
        tmp_csv.replace(alt)
        csv_path = alt

    # Match checks vs Session 12 published targets
    mismatches: List[str] = []
    for wsize, targets in FULL_LADDER_GIB.items():
        for stage, target in targets.items():
            got = gib_from_bytes_per_param(N_30B, bytes_per_param(stage, wsize))
            if abs(got - target) > 0.2:
                mismatches.append(f"W={wsize} {stage}: got {got:.3f} vs target {target}")

    for stage, target in FULL_BYTES_W8.items():
        got = bytes_per_param(stage, 8)
        if abs(got - target) > 1e-2:
            mismatches.append(f"bytes W=8 {stage}: got {got:.4f} vs target {target}")

    md = reports / "assgn012_zero_table.md"
    lines = [
        "# Assgn012 ZeRO Memory Table (30B overlay)",
        "",
        f"Card capacity = **{CARD_GIB:.2f} GiB**; replicated floor (4 B/weight) = **{replicated_floor_gib():.2f} GiB**.",
        "",
        "| W | Stage | Bytes/param | GiB/GPU | Cluster GiB | Comm | Fit |",
        "| ---: | :--- | ---: | ---: | ---: | :---: | :--- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['world_size']} | {r['stage']} | {r['bytes_per_param']:.4f} | "
            f"{r['gib_per_gpu']:.2f} | {r['cluster_gib']:.1f} | {r['comm']} | {r['fit']} |"
        )
    lines.extend(
        [
            "",
            "## Session 12 ladder match gate",
            "",
            f"- Mismatches: **{len(mismatches)}**",
        ]
    )
    if mismatches:
        lines.append("")
        for m in mismatches:
            lines.append(f"- FAIL: {m}")
    else:
        lines.append("- PASS: all ladder cells within 0.2 GiB; W=8 bytes/param match.")
    lines.extend(
        [
            "",
            "## Formulas (Session 12)",
            "",
            "```text",
            "DP:     16",
            "ZeRO-1: 4 + 12/W",
            "ZeRO-2: 2 + 14/W",
            "ZeRO-3: 16/W",
            "```",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")

    # Demo-scale shard table
    model = TinyDecoder(DemoConfig())
    demo_rows = simulate_zero_memory_table(model, world_size=32)
    demo_md = reports / "assgn012_demo_shard_table.md"
    dlines = [
        "# Demo-scale shard plan (W=32, tiny model)",
        "",
        f"N_demo = {model.count_parameters()}",
        "",
        "| Stage | bytes/param | local model | local grad | local opt | est MiB | overlay GiB@30B |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in demo_rows:
        dlines.append(
            f"| {r['stage']} | {r['bytes_per_param_formula']} | {r['local_model_params']} | "
            f"{r['local_grad_params']} | {r['local_opt_params']} | {r['estimated_local_mib']} | "
            f"{r['overlay_gib_30b']} |"
        )
    dlines.append("")
    demo_md.write_text("\n".join(dlines), encoding="utf-8")

    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "csv_path": as_repo_rel(csv_path, root),
        "md_path": as_repo_rel(md, root),
        "demo_md_path": as_repo_rel(demo_md, root),
        "rows": rows,
        "floor_gib": replicated_floor_gib(),
    }


def write_summary(reports: Path, results: Dict[str, Any]) -> Path:
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "Assgn012_Summary.md"
    gfx = results.get("task4_graphics", {})
    zero = results.get("task3_zero", {})
    dp = results.get("task2_dp", {})
    lines = [
        "# Assgn012 Summary — Distributed Training I (Data Parallel & ZeRO)",
        "",
        "## 1. Continuity from Session 11",
        "",
        "Mixed-precision AdamW stores **16 bytes per trainable parameter**. At 30B that is **447.0 GiB** before activations.",
        "",
        "## 2. PRIMARY graphics",
        "",
        f"- Memory ladder: `reports/{Path(str(gfx.get('ladder', 'assgn012_memory_ladder.png'))).name}`",
        f"- W=32 dashboard: `reports/{Path(str(gfx.get('dashboard', 'assgn012_w32_dashboard.png'))).name}`",
        f"- Compute vs comm: `reports/{Path(str(gfx.get('compute_comm', 'assgn012_compute_comm.png'))).name}`",
        f"- Activation vs S (recommended): `reports/assgn012_activation_vs_seq.png`",
        "",
        "## 3. Match ZeRO (Session 12 ladder)",
        "",
        f"- Gate: **{'PASS' if zero.get('ok') else 'FAIL'}**",
        f"- Replicated floor: **{zero.get('floor_gib', replicated_floor_gib()):.2f} GiB** (DP/ZeRO-1 never fit)",
        f"- W=32 GiB: DP 447.0 / Z1 122.2 / Z2 68.1 / Z3 14.0",
        "",
        "## 4. Demo DP run",
        "",
        f"- N_demo = {dp.get('n_params')}; max |w_i-w_0| = {dp.get('max_abs_param_diff_vs_rank0')}",
        f"- Distributed = {dp.get('is_distributed')}; world_size = {dp.get('world_size')}",
        "",
        "## 5. Pros / cons per stage",
        "",
        "| Stage | Pros | Cons |",
        "| :--- | :--- | :--- |",
        "| DP | Simple; math-identical larger batch | Full 16 B/param replica; never fits 30B |",
        "| ZeRO-1 | Shards 12 B optimizer; same 2P | Still replicates params+grads; never fits 30B |",
        "| ZeRO-2 | Fits from 32 GPUs; still 2P | Thin headroom (~6.4 GiB) before activations |",
        "| ZeRO-3 | Fits from 8 GPUs; lowest static | Comm rises to 3P; more gather traffic |",
        "",
        "## 6. Long context",
        "",
        "Activations scale with batch × sequence length and sit **on top** of the ladder. ZeRO-2@32 leaves ~6.4 GiB on a 74.5 GiB card — long context forces checkpointing (~+30% compute) or ZeRO-3.",
        "",
        "## 7. Industry honesty",
        "",
        "Public: DeepSpeed ZeRO (2019), PyTorch FSDP / FSDP2.",
        "Undisclosed: Anthropic Claude and current OpenAI/Gemini production sharding — not invented here.",
        "",
        f"## 8. Comm/compute reminder",
        "",
        f"H100: {IB_2P_S}/{H100_COMPUTE_S} ≈ {100*IB_2P_S/H100_COMPUTE_S:.0f}%; "
        f"B200: {IB_2P_S}/{B200_COMPUTE_S} ≈ {100*IB_2P_S/B200_COMPUTE_S:.0f}% (Session 12).",
        "",
        "## 9. Overall gate (boolean + full operand values)",
        "",
        *format_gate_print(results),
        "",
        f"Overall gate: **{'PASS' if results.get('all_ok') else 'FAIL'}** (all_ok={results.get('all_ok')})",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compute_all_ok(results: Dict[str, Any]) -> bool:
    return bool(
        results.get("task1_smoke", {}).get("ok")
        and results.get("task2_dp", {}).get("ok")
        and results.get("task3_zero", {}).get("ok")
        and results.get("task4_graphics", {}).get("ok")
    )


def build_manifest(results: Dict[str, Any]) -> Dict[str, Any]:
    smoke = results.get("task1_smoke", {})
    dp = results.get("task2_dp", {})
    zero = results.get("task3_zero", {})
    gfx = results.get("task4_graphics", {})
    primary_paths = [
        p for p in (gfx.get("ladder"), gfx.get("dashboard"), gfx.get("compute_comm")) if p
    ]
    return {
        "all_ok": results.get("all_ok"),
        "task1_smoke": {
            "ok": smoke.get("ok"),
            "smoke_sum": smoke.get("value"),
            "world_size": smoke.get("world_size"),
            "backend": smoke.get("backend"),
            "is_distributed": smoke.get("is_distributed"),
        },
        "task2_dp": {
            "ok": dp.get("ok"),
            "n_params": dp.get("n_params"),
            "max_abs_param_diff_vs_rank0": dp.get("max_abs_param_diff_vs_rank0"),
            "world_size": dp.get("world_size"),
            "peak_rss_mb": dp.get("peak_rss_mb"),
        },
        "task3_zero": {
            "ok": zero.get("ok"),
            "mismatches": len(zero.get("mismatches") or []),
            "floor_gib": zero.get("floor_gib"),
            "w32_gib": "447.0/122.2/68.1/14.0",
        },
        "task4_graphics": {
            "ok": gfx.get("ok"),
            "primary_pngs": len(primary_paths),
        },
        "card_gib": CARD_GIB,
        "n_30b": N_30B,
        "overlay_world_size": 32,
    }


def run_all(reports_dir: Path | str, dist_steps: int = 3) -> Dict[str, Any]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"all_ok": False}

    ctx = init_distributed()
    try:
        barrier()
        smoke_val, smoke_ok = smoke_all_reduce()
        barrier()
        smoke_path = write_rank_smoke(reports, ctx, smoke_val, smoke_ok)
        results["task1_smoke"] = {
            "ok": smoke_ok,
            "value": smoke_val,
            "path": as_repo_rel(smoke_path, reports.parent),
            "world_size": ctx.world_size,
            "is_distributed": ctx.is_distributed,
            "backend": ctx.backend,
        }

        results["task2_dp"] = run_dp_baseline(reports, ctx, steps=dist_steps)

        if _is_rank0(ctx):
            results["task3_zero"] = write_zero_tables(reports)
            results["task4_graphics"] = generate_all_graphics(reports)
            # DeepSpeed JSON stage configs vs W=32 lab accounting (no DeepSpeed install).
            root = reports.parent
            cmp = run_compare(root)
            results["task_json_compare"] = {
                "ok": cmp["ok"],
                "report_path": cmp.get("report_path"),
                "n_pass": sum(1 for r in cmp["rows"] if r.verdict == "PASS"),
                "n_total": len(cmp["rows"]),
            }
            model = TinyDecoder(DemoConfig())
            plans = {s: plan_for_stage(model, s, 32, rank=0) for s in STAGES}
            results["task3_plans"] = {
                s: {
                    "bytes_per_param": plans[s].bytes_per_param,
                    "local_opt": plans[s].local_opt_params,
                    "local_grad": plans[s].local_grad_params,
                    "local_model": plans[s].local_model_params,
                }
                for s in STAGES
            }
            # Compute all_ok BEFORE writing Summary (footer must see the real gate).
            results["all_ok"] = compute_all_ok(results)
            summary = write_summary(reports, results)
            results["summary_path"] = as_repo_rel(summary, root)
            manifest = build_manifest(results)
            manifest["task_json_compare"] = results["task_json_compare"]
            man_path = reports / "assgn012_manifest.json"
            man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            results["manifest_path"] = as_repo_rel(man_path, root)
        barrier()
    finally:
        destroy_distributed()

    return results


def format_gate_print(results: Dict[str, Any]) -> List[str]:
    """Always print boolean AND companion numeric / identity values."""
    smoke = results.get("task1_smoke", {})
    dp = results.get("task2_dp", {})
    zero = results.get("task3_zero", {})
    gfx = results.get("task4_graphics", {})
    jc = results.get("task_json_compare", {})
    primary_n = sum(
        1
        for k in ("ladder", "dashboard", "compute_comm")
        if gfx.get(k) is not None
    )
    lines = [
        (
            f"task1_smoke.ok={smoke.get('ok')}  smoke_sum={smoke.get('value')}  "
            f"world_size={smoke.get('world_size')}  backend={smoke.get('backend')}"
        ),
        (
            f"task2_dp.ok={dp.get('ok')}  max_abs_diff={dp.get('max_abs_param_diff_vs_rank0')}  "
            f"N_demo={dp.get('n_params')}  peak_rss_mb={dp.get('peak_rss_mb')}"
        ),
        (
            f"task3_zero.ok={zero.get('ok')}  mismatches={len(zero.get('mismatches') or [])}  "
            f"W32_GiB=447.0/122.2/68.1/14.0  floor_gib={zero.get('floor_gib')}"
        ),
        f"task4_graphics.ok={gfx.get('ok')}  primary_pngs={primary_n}",
        (
            f"task_json_compare.ok={jc.get('ok')}  "
            f"stages_pass={jc.get('n_pass')}/{jc.get('n_total')}"
        ),
        f"all_ok={results.get('all_ok')}  (PASS if True; FAIL if False)",
    ]
    return lines
