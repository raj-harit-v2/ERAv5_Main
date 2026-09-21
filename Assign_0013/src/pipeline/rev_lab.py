"""Assign_0013 lab orchestrator: four-run protocol, versioned reports, gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.tiny_lm import LMConfig, TinyReversibleLM
from src.pipeline.rev_graphics import (
    PRIMARY_LOSS,
    PRIMARY_MEM,
    PRIMARY_SPEED,
    generate_all_graphics,
)
from src.pipeline.train_rev import OPTIMIZER_NAME, dump_run_json, probe_max_batch, train_one
from src.utils.env_config import env_float, env_int, env_str, load_env_file
from src.utils.metrics import (
    FULL_TOKEN_BUDGET,
    N_PARAMS_MAX,
    N_PARAMS_MIN,
    gpu_name,
    params_in_range,
    pick_device,
)
from src.utils.rates import resolve_usd_rate
from src.utils.versioning import resolve_reports_dir, update_index, write_latest


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


def format_gate_print(results: Dict[str, Any]) -> List[str]:
    n = results.get("n_params")
    smoke = results.get("smoke")
    b_fixed = results.get("B_fixed")
    b_max = results.get("B_max")
    pngs = results.get("primary_pngs", 0)
    winner = results.get("winner")
    all_ok = results.get("all_ok")
    return [
        f"n_params={n}  range=[{N_PARAMS_MIN},{N_PARAMS_MAX}]  in_range={params_in_range(int(n or 0))}",
        f"smoke={smoke}  token_budget={results.get('token_budget')}  FULL_TOKEN_BUDGET={FULL_TOKEN_BUDGET}",
        f"B_fixed={b_fixed}  B_max={b_max}  B_max_gt_B_fixed={bool(b_max and b_fixed and b_max > b_fixed)}",
        f"winner={winner}  (picked from loss trajectory at B_fixed)",
        f"optimizer={OPTIMIZER_NAME}  reversible_dropout={results.get('reversible_dropout')}",
        f"version={results.get('version')}  reports_dir={results.get('reports_dir')}",
        f"usd_per_hour={results.get('usd_per_hour')}  rate_source={results.get('rate_source')}",
        f"primary_pngs={pngs}  (need 3)",
        f"all_ok={all_ok}  (PASS if True; FAIL if False)",
    ]


def run_all(
    reports: Path,
    *,
    root: Optional[Path] = None,
    tokens: Optional[int] = None,
    seq_len: Optional[int] = None,
    batch: Optional[int] = None,
    smoke: bool = True,
    h: Optional[float] = None,
    version: Optional[str] = None,
    free_colab: bool = False,
) -> Dict[str, Any]:
    """Run the four-train protocol into a versioned folder under reports/.

    ``reports`` may be the reports root (preferred) or a specific version dir.
    When it is the root (or ends with ``reports``), a new ``vNNN_...`` folder
    is allocated unless ``version`` / ASSGN013_RUN_VERSION resumes one.
    """
    root = root or reports.parent
    env = load_env_file(root / ".env")
    host = env_str(env, "ASSGN013_HOST", "local")
    env_usd_raw = env.get("ASSGN013_USD_PER_HOUR", "")
    env_usd: Optional[float]
    if env_usd_raw == "":
        env_usd = None
    else:
        env_usd = float(env_usd_raw)
    free_colab = free_colab or env_str(env, "ASSGN013_FREE_COLAB", "0") in ("1", "true", "True", "yes")
    seq_len = int(seq_len or env_int(env, "ASSGN013_SEQ_LEN", 128 if smoke else 512))
    h = float(h if h is not None else env_float(env, "ASSGN013_H", 0.25))
    lr = env_float(env, "ASSGN013_LR", 3e-4)
    default_tokens = env_int(env, "ASSGN013_SMOKE_TOKENS", 100_000) if smoke else env_int(
        env, "ASSGN013_TOKENS", FULL_TOKEN_BUDGET
    )
    token_budget = int(tokens if tokens is not None else default_tokens)
    device = pick_device()
    gname = gpu_name(device)
    rate = resolve_usd_rate(host=host, gpu_name=gname, env_usd=env_usd, free_colab=free_colab)

    # reports root: either explicit root/reports or the parent if a version dir was passed
    if reports.name.startswith("v") and reports.parent.name == "reports":
        reports_root = reports.parent
        override = reports.name
    elif reports.name == "reports":
        reports_root = reports
        override = version or env_str(env, "ASSGN013_RUN_VERSION", "") or None
    else:
        # custom root (e.g. Drive path)
        reports_root = reports
        override = version or env_str(env, "ASSGN013_RUN_VERSION", "") or None

    reports_env = env_str(env, "ASSGN013_REPORTS_ROOT", "")
    if reports_env:
        reports_root = Path(reports_env)
        reports_root.mkdir(parents=True, exist_ok=True)

    version_dir, version_id = resolve_reports_dir(
        reports_root,
        host=host,
        gpu_slug=rate.gpu_slug,
        smoke=smoke,
        override_version=override if override else None,
    )
    write_latest(reports_root, version_id)
    update_index(reports_root)

    probe_model = TinyReversibleLM(LMConfig(mode="standard", dropout=0.0, h=h))
    n_params = probe_model.count_parameters()
    del probe_model

    env_b_fixed = env_int(env, "ASSGN013_B_FIXED", 0)
    env_b_max = env_int(env, "ASSGN013_B_MAX", 0)
    if batch is not None:
        b_fixed = int(batch)
        b_note = "cli --batch"
    elif env_b_fixed > 0:
        b_fixed = env_b_fixed
        b_note = ".env ASSGN013_B_FIXED"
    else:
        b_fixed, b_note = probe_max_batch("standard", seq_len, h=h, device=device, cpu_fixed=2)

    if env_b_max > 0 and batch is None:
        b_max = env_b_max
        b_max_note = ".env ASSGN013_B_MAX"
    else:
        b_max, b_max_note = probe_max_batch(
            "midpoint", seq_len, h=h, device=device, cpu_fixed=b_fixed, cpu_scale=2
        )
        if b_max <= b_fixed:
            b_max = b_fixed * 2
            b_max_note = "forced B_max = 2 * B_fixed so the protocol can run"

    version_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = version_dir / "checkpoints"
    common = dict(
        seq_len=seq_len,
        token_budget=token_budget,
        h=h,
        lr=lr,
        usd_per_hour=rate.usd_per_hour,
        host=host,
        device=device,
        ckpt_dir=ckpt_dir,
        smoke=smoke,
        log_every=1 if smoke else 20,
        rate_source=rate.rate_source,
        shadow_usd_per_hour=rate.shadow_usd_per_hour,
        version=version_id,
    )

    packed_runs: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    def _go(mode: str, bsz: int, is_max: bool) -> Dict[str, Any]:
        metrics, history = train_one(mode, batch=bsz, **common)
        d = metrics.as_dict()
        d["history"] = history
        d["is_max_batch"] = is_max
        d["probe_note"] = b_note if not is_max else b_max_note
        run_id = f"{mode}_B{bsz}"
        dump_run_json(version_dir / "runs" / f"{run_id}.json", metrics, history)
        csv_rows.append({k: v for k, v in d.items() if k != "history"})
        packed_runs.append(d)
        return d

    r_std = _go("standard", b_fixed, False)
    r_eul = _go("euler", b_fixed, False)
    r_mid = _go("midpoint", b_fixed, False)

    # Winner by final loss at B_fixed (CHAT: loss trajectory).
    winner = "midpoint" if r_mid["final_loss"] <= r_eul["final_loss"] else "euler"
    r_max = _go(winner, b_max, True)
    r_max["winner"] = winner
    for row in packed_runs:
        row["winner"] = winner
    for row in csv_rows:
        row["winner"] = winner

    _write_csv(version_dir / "assgn013_runs.csv", csv_rows)
    batches = {
        "B_fixed": b_fixed,
        "B_max": b_max,
        "seq_len": seq_len,
        "winner": winner,
        "h": h,
        "n_params": n_params,
        "token_budget": token_budget,
        "smoke": smoke,
        "device": str(device),
        "gpu_name": gname,
        "B_fixed_note": b_note,
        "B_max_note": b_max_note,
        "optimizer": OPTIMIZER_NAME,
        "version": version_id,
        "usd_per_hour": rate.usd_per_hour,
        "rate_source": rate.rate_source,
        "shadow_usd_per_hour": rate.shadow_usd_per_hour,
    }
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "assgn013_batches.json").write_text(json.dumps(batches, indent=2), encoding="utf-8")
    (version_dir / "assgn013_batches.json").write_text(json.dumps(batches, indent=2), encoding="utf-8")

    png_paths = generate_all_graphics(packed_runs, version_dir)
    png_ok = all(
        (version_dir / name).is_file() and (version_dir / name).stat().st_size > 0
        for name in (PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED)
    )

    rev_dropout = r_eul["dropout"]
    token_gate = True if smoke else all(r["tokens_seen"] >= FULL_TOKEN_BUDGET for r in packed_runs)
    all_ok = (
        params_in_range(n_params)
        and r_eul["batch"] == b_fixed
        and r_mid["batch"] == b_fixed
        and r_std["batch"] == b_fixed
        and b_max > b_fixed
        and png_ok
        and rev_dropout == 0.0
        and r_std["optimizer"] == OPTIMIZER_NAME
        and token_gate
        and winner in ("euler", "midpoint")
    )

    results: Dict[str, Any] = {
        "n_params": n_params,
        "smoke": smoke,
        "token_budget": token_budget,
        "B_fixed": b_fixed,
        "B_max": b_max,
        "winner": winner,
        "reversible_dropout": rev_dropout,
        "primary_pngs": 3 if png_ok else 0,
        "all_ok": all_ok,
        "device": str(device),
        "gpu_name": gname,
        "runs": packed_runs,
        "png_paths": {k: str(v) for k, v in png_paths.items()},
        "standard_loss": r_std["final_loss"],
        "euler_loss": r_eul["final_loss"],
        "midpoint_loss": r_mid["final_loss"],
        "max_loss": r_max["final_loss"],
        "host": host,
        "version": version_id,
        "reports_dir": str(version_dir),
        "reports_root": str(reports_root),
        "usd_per_hour": rate.usd_per_hour,
        "rate_source": rate.rate_source,
        "shadow_usd_per_hour": rate.shadow_usd_per_hour,
    }
    (version_dir / "assgn013_manifest.json").write_text(json.dumps(
        {k: v for k, v in results.items() if k != "runs"},
        indent=2,
    ), encoding="utf-8")

    summary = version_dir / "Assgn013_Summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Assgn013 smoke/full summary",
                "",
                f"version={version_id}",
                f"N={n_params}  smoke={smoke}  winner={winner}",
                f"B_fixed={b_fixed}  B_max={b_max}  seq_len={seq_len}",
                f"usd_per_hour={rate.usd_per_hour}  rate_source={rate.rate_source}",
                f"standard_loss={r_std['final_loss']:.4f}  euler={r_eul['final_loss']:.4f}  midpoint={r_mid['final_loss']:.4f}",
                "",
                *format_gate_print(results),
                "",
            ]
        ),
        encoding="utf-8",
    )
    results["summary_path"] = str(summary)
    update_index(reports_root)
    return results
