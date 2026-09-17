"""Pytest checks for Session 12 ZeRO assignment artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def lab_results():
    from src.pipeline.zero_lab import run_all

    return run_all(REPORTS)


def test_bytes_per_param_formulas():
    from src.utils.zero_accounting import FULL_BYTES_W8, bytes_per_param

    assert abs(bytes_per_param("DP", 8) - 16.0) < 1e-9
    assert abs(bytes_per_param("ZeRO-1", 8) - 5.5) < 1e-9
    assert abs(bytes_per_param("ZeRO-2", 8) - 3.75) < 1e-9
    assert abs(bytes_per_param("ZeRO-3", 8) - 2.0) < 1e-9
    for stage, target in FULL_BYTES_W8.items():
        assert abs(bytes_per_param(stage, 8) - target) < 1e-2
    # Reject Study Guide 10/W myth: ZeRO-1 at W=8 is 5.50 not 4+10/8=5.25
    assert abs(bytes_per_param("ZeRO-1", 8) - (4 + 10 / 8)) > 0.2


def test_memory_ladder_matches_full(lab_results):
    from src.utils.zero_accounting import FULL_LADDER_GIB, bytes_per_param, gib_from_bytes_per_param, N_30B

    assert lab_results["task3_zero"]["ok"]
    for w, targets in FULL_LADDER_GIB.items():
        for stage, target in targets.items():
            got = gib_from_bytes_per_param(N_30B, bytes_per_param(stage, w))
            assert abs(got - target) <= 0.2, f"W={w} {stage}: {got} vs {target}"


def test_w32_fit_flags():
    from src.utils.zero_accounting import stage_row

    assert stage_row("DP", 32).fit == "NEVER_FIT"
    assert stage_row("ZeRO-1", 64).fit == "NEVER_FIT"
    assert stage_row("ZeRO-2", 32).fit == "FITS"
    assert stage_row("ZeRO-2", 16).fit == "OOM"
    assert stage_row("ZeRO-3", 8).fit == "FITS"


def test_smoke_and_dp(lab_results):
    assert lab_results["task1_smoke"]["ok"]
    assert lab_results["task2_dp"]["ok"]
    assert lab_results["task2_dp"]["n_params"] > 0
    assert (REPORTS / "assgn012_rank_smoke.txt").is_file()
    assert (REPORTS / "assgn012_dp_baseline.md").is_file()


def test_primary_graphics_exist(lab_results):
    assert lab_results["task4_graphics"]["ok"]
    for name in (
        "assgn012_memory_ladder.png",
        "assgn012_w32_dashboard.png",
        "assgn012_compute_comm.png",
    ):
        png = REPORTS / name
        assert png.is_file() and png.stat().st_size > 0
    assert (REPORTS / "assgn012_graphics_manifest.md").is_file()


def test_zero_table_artifacts(lab_results):
    assert (REPORTS / "assgn012_zero_table.csv").is_file()
    text = (REPORTS / "assgn012_zero_table.md").read_text(encoding="utf-8")
    assert "4 + 12/W" in text
    assert "PASS" in text
    assert "10/W" not in text.replace("12/W", "")  # soft check


def test_comm_labels():
    from src.utils.zero_accounting import comm_label

    assert comm_label("DP") == "2P"
    assert comm_label("ZeRO-1") == "2P"
    assert comm_label("ZeRO-2") == "2P"
    assert comm_label("ZeRO-3") == "3P"


def test_demo_model_forward():
    import torch
    from src.llm.demo_model import DemoConfig, TinyDecoder, synthetic_batch, cross_entropy_loss

    cfg = DemoConfig()
    model = TinyDecoder(cfg)
    x, y = synthetic_batch(cfg, rank=0)
    logits = model(x)
    assert logits.shape == (cfg.batch, cfg.seq_len, cfg.vocab_size)
    loss = cross_entropy_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_verification_gate(lab_results):
    assert lab_results["all_ok"]
    assert (REPORTS / "Assgn012_Summary.md").is_file()
    assert (REPORTS / "assgn012_manifest.json").is_file()
    summary = (REPORTS / "Assgn012_Summary.md").read_text(encoding="utf-8")
    assert "Undisclosed" in summary
    assert "ZeRO-3" in summary
    assert "all_ok=True" in summary
    assert "Overall gate: **PASS**" in summary
    assert "W32_GiB=447.0/122.2/68.1/14.0" in summary
    assert "mismatches=0" in summary
    manifest = __import__("json").loads((REPORTS / "assgn012_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("all_ok") is True
    assert "smoke_sum" in manifest["task1_smoke"]
    assert "n_params" in manifest["task2_dp"]


def test_deepspeed_json_compare_pass():
    from src.utils.deepspeed_json_compare import run_compare

    # Ensure lab CSV exists
    if not (REPORTS / "assgn012_zero_table.csv").is_file():
        from src.pipeline.zero_lab import run_all

        run_all(REPORTS)
    result = run_compare(ROOT)
    assert result["ok"] is True
    assert len(result["rows"]) == 3
    assert all(r.verdict == "PASS" for r in result["rows"])
    report = REPORTS / "assgn012_zero_json_compare.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Overall: **PASS**" in text
    assert "ds_zero1.json" in text
    assert "ds_zero2.json" in text
    assert "ds_zero3.json" in text
    for name in ("configs/deepspeed/ds_zero1.json", "configs/deepspeed/ds_zero2.json", "configs/deepspeed/ds_zero3.json"):
        assert (ROOT / name).is_file()
