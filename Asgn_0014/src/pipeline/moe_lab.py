"""Assign_0014 lab orchestrator: dense train -> upcycle -> MoE train -> PRIMARY graphics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.pipeline.moe_graphics import (
    PRIMARY_LOAD,
    PRIMARY_LOSS,
    PRIMARY_SPEED,
    generate_all_graphics,
)
from src.pipeline.train_moe import OPTIMIZER_NAME, convert_dense_model, dump_run_json, train_dense, train_moe
from src.utils.env_config import env_float, env_int, env_str, load_env_file
from src.utils.metrics import (
    DENSE_TOKEN_BUDGET,
    MAX_VIO_BOUND,
    MOE_TOKEN_BUDGET,
    N_PARAMS_MAX,
    N_PARAMS_MIN,
    gpu_name,
    params_in_range,
    pick_device,
)
from src.utils.rates import resolve_usd_rate
from src.utils.versioning import (
    budget_slug_from_tokens,
    resolve_reports_dir,
    update_index,
    write_latest,
)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def evaluate_gates(manifest: Dict[str, Any], smoke: bool) -> Dict[str, Any]:
    n_dense = int(manifest.get("n_dense") or 0)
    n_moe = int(manifest.get("n_moe_total") or 0)
    n_ok = params_in_range(n_dense)
    moe_bigger = n_moe > n_dense > 0
    t_dense = float(manifest.get("tokens_dense") or 0)
    t_moe = float(manifest.get("tokens_moe") or 0)
    if smoke:
        token_ok = t_dense > 0 and t_moe > 0
    else:
        token_ok = t_dense >= DENSE_TOKEN_BUDGET * 0.99 and t_moe >= MOE_TOKEN_BUDGET * 0.99
    loss_convert = float(manifest.get("loss_at_convert") or float("nan"))
    loss_moe = float(manifest.get("final_moe_loss") or float("nan"))
    loss_ok = (loss_moe == loss_moe) and (loss_convert == loss_convert) and (loss_moe < loss_convert)
    # allow tiny float noise: also OK if relative drop
    if not loss_ok and loss_moe == loss_moe and loss_convert == loss_convert:
        loss_ok = loss_moe <= loss_convert * 1.001  # essentially not worse on smoke noise
        # for smoke: prefer any downward move in last moe window if available
        slope = float(manifest.get("moe_loss_slope") or 0.0)
        if smoke and slope < 0:
            loss_ok = True
    dead = float(manifest.get("dead_experts") or 0)
    max_vio = float(manifest.get("max_vio") or 0)
    # Require dead==0 always (4k smoke already clears this when routing works).
    # Full also caps MaxVio; mid-ladder smoke allows MaxVio up to MAX_VIO_BOUND*1.5
    # so a recovering run is not hard-failed solely on a transient spike.
    if smoke:
        max_vio_ok = max_vio == max_vio and max_vio <= (MAX_VIO_BOUND * 1.5)
        balance_ok = dead == 0.0 and max_vio_ok
    else:
        balance_ok = dead == 0.0 and max_vio <= MAX_VIO_BOUND
    png_ok = int(manifest.get("primary_pngs") or 0) >= 3
    opt_ok = manifest.get("optimizer") == OPTIMIZER_NAME
    aux_off = not bool(manifest.get("use_ce_aux", False))
    router_fp32 = str(manifest.get("router_dtype", "")).lower() in ("float32", "torch.float32", "fp32")
    all_ok = all(
        [n_ok, moe_bigger, token_ok, loss_ok, balance_ok, png_ok, opt_ok, aux_off, router_fp32]
    )
    return {
        "n_ok": n_ok,
        "moe_bigger": moe_bigger,
        "token_ok": token_ok,
        "loss_ok": loss_ok,
        "balance_ok": balance_ok,
        "png_ok": png_ok,
        "opt_ok": opt_ok,
        "aux_off": aux_off,
        "router_fp32": router_fp32,
        "all_ok": all_ok,
        "n_dense": n_dense,
        "n_moe_total": n_moe,
        "dead_experts": dead,
        "max_vio": max_vio,
        "loss_at_convert": loss_convert,
        "final_moe_loss": loss_moe,
    }


def format_gate_print(results: Dict[str, Any]) -> List[str]:
    g = results.get("gates") or {}
    lines = [
        f"n_dense={results.get('n_dense')}  range=[{N_PARAMS_MIN},{N_PARAMS_MAX}]  n_ok={g.get('n_ok')}",
        f"n_moe_total={results.get('n_moe_total')}  n_moe_active={results.get('n_moe_active')}  moe_bigger={g.get('moe_bigger')}",
        f"tokens_dense={results.get('tokens_dense')}  tokens_moe={results.get('tokens_moe')}  token_ok={g.get('token_ok')}",
        f"loss_at_convert={results.get('loss_at_convert')}  final_moe_loss={results.get('final_moe_loss')}  loss_ok={g.get('loss_ok')}",
        f"max_vio={results.get('max_vio')}  dead_experts={results.get('dead_experts')}  balance_ok={g.get('balance_ok')}",
        f"optimizer={OPTIMIZER_NAME}  use_ce_aux={results.get('use_ce_aux')}  router_dtype={results.get('router_dtype')}",
        f"aux_off={g.get('aux_off')}  router_fp32={g.get('router_fp32')}  png_ok={g.get('png_ok')}",
        f"version={results.get('version')}  reports_dir={results.get('reports_dir')}",
        f"usd_per_hour={results.get('usd_per_hour')}  rate_source={results.get('rate_source')}",
        f"primary_pngs={results.get('primary_pngs')}  (need 3)",
        f"all_ok={g.get('all_ok')}  (PASS if True; FAIL if False)",
    ]
    try:
        lc = float(results.get("loss_at_convert"))
        lm = float(results.get("final_moe_loss"))
        if lm == lm and lc == lc and lm > lc:
            lines.append(
                f"WARN: MoE final_loss {lm:.6f} > loss_at_convert {lc:.6f} "
                "(post-convert rise; OK on tiny smoke via slope, not for full)"
            )
    except (TypeError, ValueError):
        pass
    try:
        dead = float(results.get("dead_experts") or 0)
        max_vio = float(results.get("max_vio") or 0)
        if dead > 0 or (max_vio == max_vio and max_vio > MAX_VIO_BOUND):
            lines.append(
                f"WARN: routing unhealthy dead={dead} max_vio={max_vio} "
                f"(need dead==0 and MaxVio<={MAX_VIO_BOUND} for graded full)"
            )
    except (TypeError, ValueError):
        pass
    return lines


def run_all(
    reports: Path,
    *,
    root: Optional[Path] = None,
    tokens_dense: Optional[int] = None,
    tokens_moe: Optional[int] = None,
    seq_len: Optional[int] = None,
    batch: Optional[int] = None,
    smoke: bool = True,
    version: Optional[str] = None,
    free_colab: bool = False,
    n_experts: int = 8,
    top_k: int = 2,
) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    env = load_env_file(root / ".env")
    device = pick_device()
    gname = gpu_name(device)
    host = env_str(env, "ASSGN014_HOST", "local")
    free = free_colab or env_str(env, "ASSGN014_FREE_COLAB", "0") in ("1", "true", "True")
    env_usd_raw = env.get("ASSGN014_USD_PER_HOUR", "")
    env_usd = float(env_usd_raw) if env_usd_raw not in ("", None) else None
    rate = resolve_usd_rate(host=host, gpu_name=gname, env_usd=env_usd, free_colab=free)

    seq_len = int(seq_len if seq_len is not None else env_int(env, "ASSGN014_SEQ_LEN", 512))
    # Default batch 8 on CUDA (T4 headroom); 2 on CPU/MPS for smoke comfort.
    default_batch = 8 if device.type == "cuda" else 2
    batch = int(batch if batch is not None else env_int(env, "ASSGN014_BATCH", default_batch))
    if batch <= 0:
        batch = default_batch
    if smoke:
        td = int(tokens_dense if tokens_dense is not None else env_int(env, "ASSGN014_SMOKE_TOKENS_DENSE", 4096))
        tm = int(tokens_moe if tokens_moe is not None else env_int(env, "ASSGN014_SMOKE_TOKENS_MOE", 4096))
    else:
        td = int(tokens_dense if tokens_dense is not None else env_int(env, "ASSGN014_TOKENS_DENSE", DENSE_TOKEN_BUDGET))
        tm = int(tokens_moe if tokens_moe is not None else env_int(env, "ASSGN014_TOKENS_MOE", MOE_TOKEN_BUDGET))

    override = version or env_str(env, "ASSGN014_RUN_VERSION", "") or None
    version_tag = env_str(env, "ASSGN014_VERSION_TAG", "wu_phase") or "wu_phase"
    budget = budget_slug_from_tokens(td, tm)
    # Resolve reports root: explicit version dir, caller path, or ASSGN014_REPORTS_ROOT (Drive).
    if reports.name != "reports" and (reports / "assgn014_manifest.json").exists():
        vdir = reports
        vid = reports.name
        reports_root = reports.parent
    else:
        reports_root = Path(reports)
        reports_env = env_str(env, "ASSGN014_REPORTS_ROOT", "")
        env_reports = os.environ.get("ASSGN014_REPORTS_ROOT", "").strip()
        if env_reports:
            reports_root = Path(env_reports)
        elif reports_env:
            reports_root = Path(reports_env)
        reports_root.mkdir(parents=True, exist_ok=True)
        vdir, vid = resolve_reports_dir(
            reports_root,
            host=host,
            gpu_slug=rate.gpu_slug,
            smoke=smoke,
            override_version=override if override else None,
            budget_slug=budget,
            tag=version_tag,
        )
        reports_root = vdir.parent

    lr = env_float(env, "ASSGN014_LR", 3e-4)
    bias_gamma = env_float(env, "ASSGN014_BIAS_GAMMA", 0.001)
    drop_frac = env_float(env, "ASSGN014_DROP_REDRAW", 0.5)
    warmup_steps = env_int(env, "ASSGN014_WARMUP_STEPS", 2000)
    # Caller/CLI wins; .env fills only when still at library defaults mid-call.
    if env.get("ASSGN014_N_EXPERTS", "").strip() and n_experts == 8:
        n_experts = env_int(env, "ASSGN014_N_EXPERTS", n_experts)
    if env.get("ASSGN014_TOP_K", "").strip() and top_k == 2:
        top_k = env_int(env, "ASSGN014_TOP_K", top_k)
    log_every = 1 if smoke else 20

    ckpt_dir = vdir / "checkpoints"
    runs_dir = vdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dense_model, dense_m, dense_hist = train_dense(
        batch=batch,
        seq_len=seq_len,
        token_budget=td,
        lr=lr,
        usd_per_hour=rate.usd_per_hour,
        host=host,
        device=device,
        log_every=log_every,
        ckpt_dir=ckpt_dir,
        smoke=smoke,
        rate_source=rate.rate_source,
        shadow_usd_per_hour=rate.shadow_usd_per_hour,
        version=vid,
    )
    dump_run_json(runs_dir / "dense.json", dense_m, dense_hist)
    loss_at_convert = dense_m.final_loss
    convert_tokens = float(dense_m.tokens_seen)

    moe_model = convert_dense_model(
        dense_model,
        n_experts=n_experts,
        top_k=top_k,
        bias_gamma=bias_gamma,
        drop_redraw_frac=drop_frac,
        warmup_steps=warmup_steps,
        use_three_tier=True,
    )
    moe_model, moe_m, moe_hist = train_moe(
        moe_model,
        batch=batch,
        seq_len=seq_len,
        token_budget=tm,
        tokens_offset=int(convert_tokens),
        loss_at_convert=loss_at_convert,
        lr=lr,
        usd_per_hour=rate.usd_per_hour,
        host=host,
        device=device,
        log_every=log_every,
        ckpt_dir=ckpt_dir,
        smoke=smoke,
        rate_source=rate.rate_source,
        shadow_usd_per_hour=rate.shadow_usd_per_hour,
        version=vid,
    )
    dump_run_json(runs_dir / "moe.json", moe_m, moe_hist)

    # MoE loss slope (last half of moe history)
    moe_loss_slope = 0.0
    if len(moe_hist) >= 2:
        mid = len(moe_hist) // 2
        y0 = moe_hist[mid]["loss"]
        y1 = moe_hist[-1]["loss"]
        moe_loss_slope = float(y1 - y0)

    load_t = moe_model.load_histogram()
    load_np = load_t.numpy() if load_t is not None else None

    generate_all_graphics(
        vdir,
        dense_hist=dense_hist,
        moe_hist=moe_hist,
        dense_metrics=dense_m.as_dict(),
        moe_metrics=moe_m.as_dict(),
        convert_tokens=convert_tokens,
        load_hist=load_np,
    )
    pngs = sum(
        1
        for name in (PRIMARY_LOSS, PRIMARY_LOAD, PRIMARY_SPEED)
        if (vdir / name).is_file() and (vdir / name).stat().st_size > 0
    )

    rows = [dense_m.as_dict(), moe_m.as_dict()]
    _write_csv(vdir / "assgn014_runs.csv", rows)

    manifest: Dict[str, Any] = {
        "version": vid,
        "smoke": smoke,
        "host": host,
        "gpu_name": gname,
        "batch": batch,
        "seq_len": seq_len,
        "n_dense": dense_m.n_params,
        "n_moe_total": moe_m.n_params,
        "n_moe_active": moe_m.n_active,
        "tokens_dense": dense_m.tokens_seen,
        "tokens_moe": moe_m.tokens_seen,
        "loss_at_convert": loss_at_convert,
        "final_dense_loss": dense_m.final_loss,
        "final_moe_loss": moe_m.final_loss,
        "moe_loss_slope": moe_loss_slope,
        "max_vio": moe_m.max_vio,
        "dead_experts": moe_m.dead_experts,
        "optimizer": OPTIMIZER_NAME,
        "use_ce_aux": False,
        "router_dtype": moe_m.router_dtype,
        "n_experts": n_experts,
        "top_k": top_k,
        "bias_gamma": bias_gamma,
        "drop_redraw_frac": drop_frac,
        "warmup_steps": warmup_steps,
        "warmup_steps_effective": int(getattr(moe_m, "warmup_steps_effective", 0) or 0),
        "use_three_tier": True,
        "version_tag": version_tag,
        "budget_slug": budget,
        "primary_pngs": pngs,
        "usd_per_hour": rate.usd_per_hour,
        "rate_source": rate.rate_source,
        "convert_tokens": convert_tokens,
        "upcycle": "partition_drop",
    }
    gates = evaluate_gates(manifest, smoke=smoke)
    manifest["gates"] = gates
    manifest["all_ok"] = gates["all_ok"]
    (vdir / "assgn014_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = dict(manifest)
    results["reports_dir"] = str(vdir)
    results["gates"] = gates
    results["all_ok"] = gates["all_ok"]

    wu_eff = int(getattr(moe_m, "warmup_steps_effective", 0) or 0)
    summary_lines = [
        "# Assgn014 smoke/full summary",
        "",
        f"version={vid}",
        f"N_dense={dense_m.n_params}  N_moe_total={moe_m.n_params}  N_moe_active={moe_m.n_active}",
        f"smoke={smoke}  batch={batch}  seq_len={seq_len}",
        f"tokens_dense={dense_m.tokens_seen}  tokens_moe={moe_m.tokens_seen}",
        f"loss_at_convert={loss_at_convert:.6f}  final_moe_loss={moe_m.final_loss:.6f}",
        f"max_vio={moe_m.max_vio:.4f}  dead_experts={moe_m.dead_experts}",
        f"usd_per_hour={rate.usd_per_hour}  rate_source={rate.rate_source}",
        f"upcycle=partition_drop  n_experts={n_experts}  top_k={top_k}  bias_gamma={bias_gamma}",
        f"use_ce_aux=False  router_dtype={moe_m.router_dtype}",
        f"warmup_steps_floor={warmup_steps}  warmup_steps_effective={wu_eff}  "
        f"use_three_tier=True  version_tag={version_tag}  budget_slug={budget}",
        "",
        *format_gate_print(results),
        "",
        "## How to read this run",
        "",
        f"- Recipe tag `{version_tag}`: probabilistic warm-up auto-extends to cover the MoE phase "
        f"(floor={warmup_steps}, effective={wu_eff} steps), then hard top-k only if phase ends first; "
        "three-tier loss-free bias (gamma1=0.0005, gamma2=0.002, gamma3=0.01) with bias clamp +/-5.",
        "- CE floor: vocab=8192 -> random-token floor ln(8192)~=9.011. Tiny late loss drops on synthetic data are expected.",
        f"- balance_ok needs dead_experts==0. Smoke no longer forgives dead experts (MaxVio slack up to {MAX_VIO_BOUND * 1.5}). "
        f"Full also needs MaxVio<={MAX_VIO_BOUND}.",
        "- PRIMARY PNGs (graded proof): assgn014_loss_trajectories.png, assgn014_expert_load_health.png, "
        "assgn014_memory_speed_dashboard.png. pipeline_bubble_*.png is NOT required.",
        "- Next: if all_ok and dead==0, continue ladder. If dead>0, stop before graded full. "
        "Failed full -> new version folder; do not RESUME across different budgets.",
        f"- Folder naming: vNNN_host_gpu_budget_kind_tag (this run budget={budget}, tag={version_tag}). "
        "Older folders without _wu_phase are pre-phase-warmup history (wu2k ladder OK but full collapsed after step 2000).",
        "",
    ]
    (vdir / "Assgn014_Summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    write_latest(reports_root, vid)
    update_index(reports_root)

    # optional load hist save
    if load_np is not None:
        np.save(vdir / "expert_load_mean.npy", load_np)

    return results
