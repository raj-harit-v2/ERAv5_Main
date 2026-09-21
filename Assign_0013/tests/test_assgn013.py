"""Pytest checks for Session 13 reversible 20M lab."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.reversible import midpoint_reconstruct, midpoint_step
from src.llm.tiny_lm import LMConfig, TinyReversibleLM
from src.pipeline.rev_graphics import PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED
from src.utils.metrics import (
    FULL_TOKEN_BUDGET,
    N_PARAMS_MAX,
    N_PARAMS_MIN,
    params_in_range,
    steps_for_budget,
    tokens_per_step,
    usd_cost,
)


def test_param_count_in_range():
    model = TinyReversibleLM(LMConfig(mode="standard", dropout=0.0))
    n = model.count_parameters()
    assert params_in_range(n), f"N={n} not in [{N_PARAMS_MIN},{N_PARAMS_MAX}]"


def test_reversible_dropout_must_be_zero():
    with pytest.raises(ValueError):
        TinyReversibleLM(LMConfig(mode="midpoint", dropout=0.1, h=0.25))


def test_token_budget_helper():
    assert tokens_per_step(4, 512) == 2048
    assert steps_for_budget(50_000_000, 4, 512) == (50_000_000 + 2047) // 2048
    assert usd_cost(3600.0, 1.25) == pytest.approx(1.25)


def test_midpoint_reconstruct_tiny_tensor():
    torch.manual_seed(0)
    linear = nn.Linear(8, 8, bias=False)

    def f_mod(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(linear(x))

    p_prev = torch.randn(2, 4, 8)
    p_curr = torch.randn(2, 4, 8)
    h = 0.25
    p_next = p_prev + 2.0 * h * f_mod(p_curr)
    recon = midpoint_reconstruct(p_next, p_curr, f_mod, h)
    assert torch.allclose(recon, p_prev, atol=1e-5, rtol=1e-5)


def test_midpoint_autograd_step_runs():
    torch.manual_seed(1)
    blk = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8), nn.Tanh())
    p_prev = torch.randn(1, 3, 8, requires_grad=True)
    p_curr = torch.randn(1, 3, 8, requires_grad=True)
    p_next = midpoint_step(p_prev, p_curr, blk, 0.25)
    p_next.sum().backward()
    assert p_prev.grad is not None
    assert p_curr.grad is not None


def test_primary_graphics_exist():
    from src.utils.versioning import resolve_reports_for_eval

    reports = resolve_reports_for_eval(REPORTS)
    missing = []
    for name in (PRIMARY_LOSS, PRIMARY_MEM, PRIMARY_SPEED):
        png = reports / name
        if not (png.is_file() and png.stat().st_size > 0):
            missing.append(name)
    if missing:
        pytest.skip(
            f"PRIMARY PNGs missing {missing} under {reports}; "
            "run tests/assgn013_rev_lab.py --smoke first"
        )


def test_colab_t4_rate_catalog():
    from src.utils.rates import resolve_usd_rate

    pick = resolve_usd_rate(host="colab", gpu_name="Tesla T4", env_usd=None)
    assert pick.usd_per_hour == pytest.approx(0.12)
    assert pick.gpu_slug == "t4"
    free = resolve_usd_rate(host="colab", gpu_name="Tesla T4", env_usd=0.0, free_colab=True)
    assert free.usd_per_hour == 0.0
    assert free.shadow_usd_per_hour == pytest.approx(0.12)


def test_full_budget_constant():
    assert FULL_TOKEN_BUDGET == 50_000_000
