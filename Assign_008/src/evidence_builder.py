"""Write evidence.md / evidence.json / heatmap PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import config
from src.fertility import effective_content
from src.kv_cache_math import yardstick_one_user_32k
from src.pipeline_user_sentence import PipelineResult
from src.self_diagnosis import Check
from src.utils import ensure_dir


CORRECTIONS = [
    "Full Document attention OCR broken → use closed-form Attention/CausalAttention",
    "Transformer year is 2017 (not '2018 and 17')",
    "Compute bill is quadratic T^2 (not exponential)",
    "Transcript NQA → MQA",
    "KV bytes: 2*L*H_KV*d_head*T*B*P_b (no extra trailing x2)",
    "Delta S k = v needs ||k||_2=1 (or documented norm)",
]


def write_evidence(
    result: PipelineResult,
    checks: list[Check],
    artifacts: Path = config.ARTIFACTS,
) -> dict:
    ensure_dir(artifacts)
    ensure_dir(config.DEMO_FIGURES)

    heatmap = artifacts / "attention_weights_heatmap.png"
    if result.attn_weights.numel() > 0:
        w = result.attn_weights[0, 0].detach().cpu().numpy()  # head 0
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(w, aspect="auto", cmap="viridis")
        ax.set_title("Causal attention weights (layer last, head 0)")
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(heatmap, dpi=120)
        plt.close(fig)
        # copy into demo figures
        fig2_path = config.DEMO_FIGURES / heatmap.name
        if heatmap.exists():
            fig2_path.write_bytes(heatmap.read_bytes())

    payload = {
        "schema": config.SCHEMA_VERSION,
        "phase": 1,
        "example_sentence": result.text,
        "pipeline_steps": result.steps,
        "shapes": {
            "input_ids": list(result.input_ids.shape),
            "hidden": list(result.hidden.shape),
            "attn_out": list(result.attn_out.shape),
            "logits": list(result.logits.shape),
        },
        "reply_sketch": result.reply_sketch,
        "kv_yardstick_bytes_1user_32k": yardstick_one_user_32k(),
        "fertility_demo": {
            "T": config.PEDAGOGICAL_CTX,
            "en_f": 1.0,
            "te_f_teaching": 3.0,
            "en_effective": effective_content(config.PEDAGOGICAL_CTX, 1.0),
            "te_effective": effective_content(config.PEDAGOGICAL_CTX, 3.0),
        },
        "checks": [{"id": c.id, "ok": c.ok, "detail": c.detail} for c in checks],
        "corrections": CORRECTIONS,
        "widgets": "Phase 2 — see demo_08/coach_demo/widgets/README.md",
    }

    (artifacts / "evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Assign_008 Phase 1 evidence (`{config.SCHEMA_VERSION}`)",
        "",
        f"**Example sentence:** {result.text}",
        "",
        "## Pipeline steps",
        "",
    ]
    for s in result.steps:
        lines.append(f"- {s}")
    lines += [
        "",
        "## Shapes",
        "",
        f"- input_ids `{tuple(result.input_ids.shape)}`",
        f"- hidden `[B,T,D]` `{tuple(result.hidden.shape)}`",
        f"- attn_out `{tuple(result.attn_out.shape)}`",
        f"- logits `{tuple(result.logits.shape)}`",
        f"- reply sketch: `{result.reply_sketch}`",
        "",
        "## Corrections applied",
        "",
    ]
    for c in CORRECTIONS:
        lines.append(f"- {c}")
    lines += ["", "## Self-diagnosis", ""]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"- **{mark}** `{c.id}` — {c.detail}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- heatmap: `{heatmap.name}`" if heatmap.exists() else "- heatmap: (skipped)",
        "- Widgets: **Phase 2** (stub only)",
        "",
    ]
    (artifacts / "evidence.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
