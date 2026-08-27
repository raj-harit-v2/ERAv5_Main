"""Browser widget engine parity: exported weights forward matches harness loss0."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.llm.chunked_ce import shift_logits_and_targets
from src.llm.nano_lm import NanoLM
from src.pipeline.export_widget import export_widget_weights
from src.pipeline.harness import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_export_bundle_keys():
    bundle = export_widget_weights(seed=0, device="cpu")
    assert "state_dict" in bundle
    assert "token_to_id" in bundle
    assert bundle["vocab_size"] > 4
    assert "blocks.0.attn.qkv.weight" in bundle["state_dict"]


def test_widget_loss_matches_harness():
    """Same sentence + seed=0 → loss within tolerance of seven_numbers harness."""
    tok = load_tokenizer()
    sentence = "First Citizen: Let us kill him, and we'll have corn."
    torch.manual_seed(0)
    model = NanoLM(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq=256,
        tie_weights=False,
    )
    model.eval()
    ids = tok.encode(sentence, add_bos=True, add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(tokens, return_hidden=True)
    sl, st = shift_logits_and_targets(logits, tokens)
    mask = (st != tok.pad_id).float()
    loss = F.cross_entropy(
        sl.reshape(-1, sl.size(-1)),
        st.reshape(-1),
        reduction="none",
    ).view_as(st)
    loss0 = float((loss * mask).sum() / mask.sum())

    seven_path = ROOT / "data" / "evaluation" / "seven_numbers.json"
    if seven_path.is_file():
        seven = json.loads(seven_path.read_text(encoding="utf-8"))
        assert abs(loss0 - seven["loss0"]) < 1e-4
        assert abs(math.exp(loss0) - seven["ppl0"]) < 0.5


def test_js_engine_parity_node():
    """Optional: Node.js check that S9Engine matches seven_numbers.json."""
    import shutil
    import subprocess

    if not shutil.which("node"):
        return
    script = ROOT / "tests" / "check_js_parity.js"
    if not script.is_file():
        return
    proc = subprocess.run(["node", str(script)], cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_dist_layout_after_build():
    dist = ROOT / "dist"
    if not (dist / "index.html").is_file():
        return  # skip if build_dist not run yet
    required = [
        "dist/index.html",
        "dist/widgets/s9_widget_0_loss_flow.html",
        "dist/widgets/s9_engine.js",
        "dist/widgets/s9_swiglu_diagram.js",
        "dist/data/widget_weights.json",
        "dist/data/seven_numbers.json",
        "dist/reports/Curriculum_Required_stats_03_swiglu.png",
    ]
    for rel in required:
        assert (ROOT / rel).is_file(), f"missing {rel}"
