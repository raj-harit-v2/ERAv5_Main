"""Generate thesis/diagnostic PNGs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg


def _setup_unicode_font() -> None:
    from matplotlib import font_manager

    candidates = [
        "Nirmala UI",
        "Mangal",
        "Segoe UI",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig, name: str) -> Path:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


_setup_unicode_font()


def fig_param_memory(ablation: dict[str, Any]) -> Path:
    kinds = list(ablation.keys())
    params = [ablation[k]["trainable_params"] for k in kinds]
    gb = [ablation[k]["adamw_train_state_gb"] for k in kinds]
    fig, ax = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(kinds) + 3), 3.5))
    ax[0].bar(kinds, params, color="#0891b2")
    ax[0].set_title("Trainable params")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(kinds, gb, color="#159447")
    ax[1].set_title("AdamW train-state GB (16 B/param)")
    ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("P03 param / memory budget")
    return _save(fig, "param_memory_budget.png")


# V5 session reference shapes (accounting only — never allocate these tensors).
V5_REF_V = 131_072
V5_REF_D = 8_096
V5_REF_POS_DIM = 32
V5_REF_FOURIER_CODE_DIM = 8_192  # analogy to Kronecker flatten width 256*32


def v5_reference_param_accounting(
    vocab: int = V5_REF_V,
    d_model: int = V5_REF_D,
    pos_dim: int = V5_REF_POS_DIM,
    fourier_code_dim: int = V5_REF_FOURIER_CODE_DIM,
) -> dict[str, Any]:
    """Pure-math V5 param budget. Does not materialise V×D weights."""
    dense = vocab * d_model
    kronecker = (256 * pos_dim) * d_model
    fourier = fourier_code_dim * d_model
    untied_head = d_model * vocab  # W_out shape D×V (prediction; Problem #5 territory)

    def _gb(n: int) -> float:
        return n * 16 / 1e9

    return {
        "vocab": vocab,
        "d_model": d_model,
        "pos_dim": pos_dim,
        "fourier_code_dim": fourier_code_dim,
        "dense_Vin_params": dense,
        "kronecker_proj_params": kronecker,
        "fourier_proj_params": fourier,
        "untied_head_Dout_params": untied_head,
        "dense_adamw_gb": _gb(dense),
        "kronecker_adamw_gb": _gb(kronecker),
        "fourier_adamw_gb": _gb(fourier),
        "untied_head_adamw_gb": _gb(untied_head),
        "note": "Reference accounting only; Assign_007 smoke uses V=512 D=64. Head D×V is not codec invertibility.",
    }


def fig_v5_param_budget_reference(
    vocab: int = V5_REF_V,
    d_model: int = V5_REF_D,
    pos_dim: int = V5_REF_POS_DIM,
    fourier_code_dim: int = V5_REF_FOURIER_CODE_DIM,
    also_docs: bool = True,
) -> Path:
    """Bar chart: dense V×D vs Kronecker/Fourier projection — no giant tensor alloc."""
    acc = v5_reference_param_accounting(vocab, d_model, pos_dim, fourier_code_dim)
    labels = [
        f"dense\n{vocab}×{d_model}",
        f"kronecker\n(256×{pos_dim})×{d_model}",
        f"fourier\n{fourier_code_dim}×{d_model}",
        f"untied head\n{d_model}×{vocab}\n(#5 / logits)",
    ]
    params = [
        acc["dense_Vin_params"],
        acc["kronecker_proj_params"],
        acc["fourier_proj_params"],
        acc["untied_head_Dout_params"],
    ]
    gb = [
        acc["dense_adamw_gb"],
        acc["kronecker_adamw_gb"],
        acc["fourier_adamw_gb"],
        acc["untied_head_adamw_gb"],
    ]
    colors = ["#dc2626", "#0891b2", "#159447", "#94a3b8"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].bar(labels, params, color=colors)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("parameters (log)")
    ax[0].set_title("Input path params (reference)")
    ax[0].tick_params(axis="x", labelsize=8)
    for i, p in enumerate(params):
        ax[0].text(i, p * 1.15, f"{p:,}", ha="center", va="bottom", fontsize=7, rotation=0)
    ax[1].bar(labels, gb, color=colors)
    ax[1].set_ylabel("AdamW train-state GB (16 B/param)")
    ax[1].set_title("Memory if those params were trained")
    ax[1].tick_params(axis="x", labelsize=8)
    for i, g in enumerate(gb):
        ax[1].text(i, g + 0.3, f"{g:.2f} GB", ha="center", va="bottom", fontsize=7)
    fig.suptitle(
        f"V5 reference: dense {vocab}×{d_model} = {acc['dense_Vin_params']:,}  "
        f"(Fourier/Kronecker store projection only — not a V×D table)",
        fontsize=10,
    )
    path = _save(fig, "v5_param_budget_reference.png")
    if also_docs:
        docs_dir = cfg.ROOT / "demo_07" / "figures"
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / path.name
        docs_path.write_bytes(path.read_bytes())
    return path


def fig_fourier_composition(example: str = "भारत") -> Path:
    from src.embeddings.fourier_baseline import FourierCodepointEmbedding

    emb = FourierCodepointEmbedding(d_model=8, code_dim=cfg.FOURIER_CODE_DIM, n_freq=cfg.FOURIER_N_FREQ)
    waves = []
    for p, ch in enumerate(list(example)[:8]):
        waves.append(emb.wave(ord(ch), p).numpy())
    waves = np.stack(waves, axis=0)
    code = emb.encode_string(example).numpy()
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    im = ax[0].imshow(waves, aspect="auto", cmap="viridis")
    ax[0].set_ylabel("char index")
    ax[0].set_title(f"P07b Fourier per-char waves for '{example}'")
    fig.colorbar(im, ax=ax[0], fraction=0.02)
    ax[1].plot(code, color="#159447")
    ax[1].set_title("Summed + normalized token code")
    ax[1].set_xlabel("code dim")
    return _save(fig, "fourier_wave_composition.png")


def fig_kronecker_example(example: str = "train") -> Path:
    from src.embeddings import KroneckerByteEmbedding

    emb = KroneckerByteEmbedding(d_model=8, pos_dim=cfg.POS_DIM_KRON)
    raw = example.encode("utf-8")
    L = min(len(raw), emb.pos_dim)
    grid = np.zeros((256, emb.pos_dim))
    for p in range(L):
        grid[raw[p], p] = 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(grid[:64, :], aspect="auto", cmap="Blues")
    ax.set_title(f"P07 Kronecker byte×pos marks (first 64 byte rows) for '{example}'")
    ax.set_xlabel("byte position")
    ax.set_ylabel("byte value (0-63 shown)")
    return _save(fig, "kronecker_byte_grid_example.png")


def fig_truncation(collision: dict[str, Any]) -> Path:
    k32 = collision["kronecker"]["32"]
    scripts = sorted(set(k32["tokens_by_script"]) | set(k32["truncation_by_script"]))
    totals = [k32["tokens_by_script"].get(s, 0) for s in scripts]
    trunc = [k32["truncation_by_script"].get(s, 0) for s in scripts]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(scripts))
    ax.bar(x - 0.2, totals, width=0.4, label="tokens", color="#94a3b8")
    ax.bar(x + 0.2, trunc, width=0.4, label="truncated@32", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(scripts)
    ax.legend()
    ax.set_title("P08 pos_dim truncation bars (Kronecker@32)")
    return _save(fig, "pos_dim_truncation_bars.png")


def fig_collisions(collision: dict[str, Any]) -> Path:
    labels = []
    vals = []
    for pd in ("32", "48", "64"):
        labels.append(f"kron@{pd}")
        vals.append(collision["kronecker"][pd]["n_collision_groups"])
    labels.append("fourier")
    vals.append(collision["fourier"]["n_collision_groups"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labels, vals, color=["#0891b2", "#0284c7", "#0369a1", "#159447"])
    ax.set_title("P09 collision groups by codec")
    ax.set_ylabel("n_collision_groups")
    return _save(fig, "collision_heatmap_scripts.png")


def fig_grad_spike(ablation: dict[str, Any]) -> Path:
    key = "fourier_canine" if "fourier_canine" in ablation else "fourier"
    g = ablation[key]["grad_proj_trace"]
    l1 = ablation[key]["grad_l1_trace"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(g, label="proj grad", color="#d97706")
    ax.plot(l1, label="layer1 grad", color="#7c3aed", alpha=0.8)
    ax.set_title(f"P10 frozen-shift grad norms ({key})")
    ax.legend()
    return _save(fig, "frozen_shift_grad_spike.png")


def fig_stage_dashboard(stage_audits: list[dict[str, Any]]) -> Path:
    stages = [a["stage"] for a in stage_audits]
    indic = []
    for a in stage_audits:
        shares = a["mixture_compliance"]["actual_script_shares"]
        indic.append(shares.get("hi", 0) + shares.get("te", 0))
    losses = [a["learning_report_card"]["mean_loss"] or 0 for a in stage_audits]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].bar(stages, indic, color="#159447")
    ax[0].set_title("Indic share by stage")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].plot(stages, losses, marker="o", color="#0891b2")
    ax[1].set_title("Mean train loss by stage")
    ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("P14 stage audit dashboard")
    return _save(fig, "stage_audit_dashboard.png")


def fig_proof_metrics(ablation: dict[str, Any]) -> Path:
    kinds = list(ablation.keys())
    hi = [ablation[k]["val_ce_hi"] for k in kinds]
    disc = [ablation[k]["discrimination_acc"] for k in kinds]
    fig, ax = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(kinds) + 3), 3.5))
    ax[0].bar(kinds, hi, color="#0891b2")
    ax[0].set_title("Val CE (Hindi slice)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(kinds, disc, color="#159447")
    ax[1].set_title("Discrimination accuracy")
    ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("P15 Problem #4 proof metrics (RoPE LM)")
    return _save(fig, "problem4_proof_metrics.png")


def fig_loss_curves(ablation: dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for kind in ablation:
        hist = ablation[kind]["history"]
        ax.plot([h["loss"] for h in hist], label=kind, alpha=0.9)
    ax.set_title(f"P15b train loss curves (position={cfg.POSITION_POLICY})")
    ax.legend(fontsize=8)
    return _save(fig, "train_val_loss_curves.png")


def fig_upgrade_roadmap(ablation: dict[str, Any]) -> Path:
    labels, vals = [], []
    for k in ("fourier", "fourier_canine", "fourier_vq"):
        if k in ablation:
            labels.append(k)
            vals.append(ablation[k]["val_ce_hi"])
    if not labels:
        labels, vals = ["fourier"], [ablation["fourier"]["val_ce_hi"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labels, vals, color=["#94a3b8", "#159447", "#d97706"][: len(labels)])
    ax.set_title("Upgrade ladder: Fourier -> CANINE -> VQ(#5 gated)")
    ax.set_ylabel("val_ce_hi (lower better)")
    return _save(fig, "upgrade_ladder_hi_ce.png")


def generate_all_figures(ablation: dict[str, Any], collision: dict[str, Any], stage_audits: list[dict[str, Any]]) -> list[str]:
    paths = [
        fig_param_memory(ablation),
        fig_v5_param_budget_reference(),
        fig_kronecker_example(),
        fig_fourier_composition(),
        fig_truncation(collision),
        fig_collisions(collision),
        fig_grad_spike(ablation),
        fig_stage_dashboard(stage_audits),
        fig_proof_metrics(ablation),
        fig_loss_curves(ablation),
        fig_upgrade_roadmap(ablation),
    ]
    return [p.name for p in paths]
