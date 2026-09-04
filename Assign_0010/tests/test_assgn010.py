"""Pytest checks for Session 10 assignment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@pytest.fixture(scope="module")
def lab_results():
    if str(ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(ROOT))
    from src.pipeline.training_loop_lab import run_all

    return run_all(REPORTS)


def test_gradient_check_passes(lab_results):
    assert lab_results["task2_gradient"]["passed"]
    assert lab_results["task2_gradient"]["diff"] < 1e-4


def test_accumulation_bug_magnitude(lab_results):
    t3 = lab_results["task3_accumulation"]
    assert abs(t3["correct"] - 2.6) < 1e-6
    assert abs(t3["buggy"] - 3.0) < 1e-6
    assert abs(t3["discrepancy_pct"] - 15.384615) < 0.15


def test_primary_plots_exist(lab_results):
    for name in ("assgn010_accumulation_bug.png", "assgn010_gradnorm_vs_loss.png"):
        p = REPORTS / name
        assert p.is_file(), f"missing {p}"
        assert p.stat().st_size > 0


def test_grad_norm_csv_rows(lab_results):
    assert lab_results["task4_grad_norm"]["n_rows"] >= 100


def test_tensor_shape_log_lines(lab_results):
    assert lab_results["task1_shapes"]["n_lines"] >= 10


def test_mfu_report_written(lab_results):
    path = Path(lab_results["task5_mfu"]["path"])
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "MFU" in text
    assert "40%" in text


def test_precision_bf16_01(lab_results):
    enc = lab_results["task6_precision"]
    assert "01111011" in enc["bf16_bits_01"]
    assert abs(enc["bf16_stored_01"] - 0.099609375) < 1e-9


def test_fp_format_table_has_twelve_formats(lab_results):
    tables = lab_results["task6_fp_tables"]
    names = tables["format_names"]
    required = {
        "fp32",
        "fp16",
        "bf16",
        "fp8 E4M3",
        "fp8 E5M2",
        "fp4 E2M1",
        "Custom",
        "int8",
        "MXFP4",
        "NVFP4",
        "MXFP4 + MBS",
        "MXFP4 + MBS + OAS",
    }
    assert required <= set(names)
    assert len(names) == 12
    assert (REPORTS / "assgn010_fp_format_table.md").is_file()
    assert (REPORTS / "assgn010_mbs_oas_table.md").is_file()
    text = (REPORTS / "assgn010_fp_format_table.md").read_text(encoding="utf-8")
    assert "MXFP4 + MBS" in text
    assert "MXFP4 + MBS + OAS" in text
    assert "Rescued!" in text


def test_mxfp4_underflow_and_oas_rescue(lab_results):
    tables = lab_results["task6_fp_tables"]
    assert tables["mxfp4_underflow"] is True
    assert tables["oas_reads_back"] != 0.0
    assert "Rescued" in tables["oas_status"]


def test_verification_gate(lab_results):
    assert lab_results["all_ok"]


def test_mini_gpt_forward():
    from src.llm.mini_gpt import MiniGPT, MiniGPTConfig

    cfg = MiniGPTConfig(block_size=16, vocab_size=64, n_embd=32, n_head=4, n_layer=1)
    model = MiniGPT(cfg)
    x = torch.randint(0, 64, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 64)
