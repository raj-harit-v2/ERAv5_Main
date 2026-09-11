"""Pytest checks for Session 11 assignment artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


@pytest.fixture(scope="module")
def lab_results():
    if str(ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(ROOT))
    from src.pipeline.optimizers_lab import run_all

    return run_all(REPORTS)


def test_adam_hand_vs_pytorch(lab_results):
    t1 = lab_results["task1_adam_hand"]
    assert t1["ok"]
    assert t1["max_abs_err"] < 1e-6
    assert (REPORTS / "assgn011_adam_hand_table.md").is_file()
    assert (REPORTS / "assgn011_adam_hand_check.txt").is_file()


def test_bias_correction_ratio_and_plot(lab_results):
    t2 = lab_results["task2_bias"]
    assert t2["ok"]
    assert abs(t2["measured_ratio_t1"] - t2["theory_ratio_t1"]) < 1e-2
    assert abs(t2["theory_ratio_t1"] - 3.162) < 0.01
    png = REPORTS / "assgn011_bias_correction_20.png"
    assert png.is_file() and png.stat().st_size > 0
    assert (REPORTS / "assgn011_bias_correction_report.md").is_file()


def test_update_weight_ratio_csv(lab_results):
    t3 = lab_results["task3_ratio"]
    assert t3["ok"]
    assert t3["t_star"] is not None
    assert t3["n_rows"] > 0
    assert (REPORTS / "assgn011_update_weight_ratio.csv").is_file()
    text = (REPORTS / "assgn011_warmup_end.md").read_text(encoding="utf-8")
    assert "T*" in text or "Detected T*" in text


def test_schedule_compare_losses(lab_results):
    t4 = lab_results["task4_schedules"]
    assert t4["ok"]
    assert t4["cosine_loss_200"] is not None
    assert t4["wsd_loss_200"] is not None
    assert t4["keep"] in ("WSD", "cosine")
    text = (REPORTS / "assgn011_schedule_compare.md").read_text(encoding="utf-8")
    assert "Keep" in text or "keep" in text.lower()
    png = REPORTS / "assgn011_cosine_vs_wsd.png"
    assert png.is_file() and png.stat().st_size > 0


def test_lr_sweep_minima_and_confidence(lab_results):
    t5 = lab_results["task5_lr_sweep"]
    assert t5["ok"]
    assert len(t5["minima"]) == 3
    assert t5["eta_4096_sp"] is not None
    png = REPORTS / "assgn011_lr_width_sweep.png"
    assert png.is_file() and png.stat().st_size > 0
    text = (REPORTS / "assgn011_lr_sweep_table.md").read_text(encoding="utf-8")
    assert "LOW" in text
    assert "4096" in text


def test_verification_gate(lab_results):
    assert lab_results["all_ok"]
    assert (REPORTS / "Assgn011_Summary.md").is_file()
    assert (REPORTS / "assgn011_manifest.json").is_file()


def test_format_gate_print_both_sides():
    """Stub-dict check: gate print and summary show measured vs other side (no run_all)."""
    if str(ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(ROOT))
    from src.pipeline.optimizers_lab import _write_summary, format_gate_print

    stub = {
        "task1_adam_hand": {"ok": True, "max_abs_err": 1.1102230246251565e-16, "tol": 1e-6},
        "task2_bias": {
            "ok": True,
            "measured_ratio_t1": 3.1622757234151555,
            "theory_ratio_t1": 3.1622776601683773,
            "png_path": "reports/assgn011_bias_correction_20.png",
        },
        "task3_ratio": {"ok": True, "t_star": 50, "warmup_steps": 50},
        "task4_schedules": {
            "ok": True,
            "cosine_loss_200": 5.558470726013184,
            "wsd_loss_200": 5.561608791351318,
            "keep": "WSD",
        },
        "task5_lr_sweep": {
            "ok": True,
            "minima": {1024: {"eta": 0.0007196856730011526, "loss": 2.0}},
            "eta_4096_sp": 0.00017992141825028815,
            "png_path": "reports/assgn011_lr_width_sweep.png",
        },
    }

    lines = format_gate_print(stub)
    assert len(lines) == 5
    joined = "\n".join(lines)
    assert "task1_adam_hand -> True  max_abs_err=1.110e-16  vs  tol=1e-6" in joined
    assert "measured=3.162276  vs  theory=3.162278" in joined
    assert "T*=50  vs  W=50" in joined
    assert "cosine@200=5.558471  vs  WSD@200=5.561609  keep=WSD" in joined
    assert "eta_1024=0.000719686  vs  eta_4096_sp=0.000179921  conf=LOW" in joined

    out = ROOT / "reports" / "_gate_format_unit_test.md"
    try:
        _write_summary(out, stub)
        text = out.read_text(encoding="utf-8")
    finally:
        if out.is_file():
            out.unlink()

    assert "| Task | Measured | Other side | Criterion | OK |" in text
    assert "max_abs_err < tol" in text
    assert "T*=50" in text
    assert "keep=WSD" in text
    assert "eta_4096" in text or "0.000179921" in text


def test_mini_gpt_forward():
    from src.llm.mini_gpt import MiniGPT, MiniGPTConfig

    cfg = MiniGPTConfig(block_size=16, vocab_size=64, n_embd=32, n_head=4, n_layer=1)
    model = MiniGPT(cfg)
    x = torch.randint(0, 64, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 64)
