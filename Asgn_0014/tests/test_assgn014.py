"""Unit tests for Assgn_014 MoE upcycling lab."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.moe import upcycle_dense_to_moe
from src.llm.tiny_lm import LMConfig, TinyDenseLM, count_parameters
from src.pipeline.moe_graphics import PRIMARY_LOAD, PRIMARY_LOSS, PRIMARY_SPEED
from src.pipeline.moe_lab import run_all
from src.utils.metrics import N_PARAMS_MAX, N_PARAMS_MIN, params_in_range


def test_dense_param_range() -> None:
    m = TinyDenseLM(LMConfig())
    n = m.count_parameters()
    assert params_in_range(n), f"N={n} not in [{N_PARAMS_MIN},{N_PARAMS_MAX}]"


def test_upcycle_grows_total() -> None:
    dense = TinyDenseLM(LMConfig())
    moe = upcycle_dense_to_moe(dense, n_experts=8, top_k=2)
    assert moe.count_parameters() > dense.count_parameters()
    assert moe.moe_cfg.use_ce_aux is False
    dtype = next(moe.blocks[0].moe.router.parameters()).dtype
    assert dtype == torch.float32


def test_smoke_lab(tmp_path: Path, monkeypatch) -> None:
    # Unit smoke stays on synthetic for speed; graded path uses ASSGN014_DATA=corpus.
    monkeypatch.setenv("ASSGN014_DATA", "synthetic")
    reports = tmp_path / "reports"
    reports.mkdir()
    results = run_all(
        reports,
        root=ROOT,
        tokens_dense=1024,
        tokens_moe=2048,
        seq_len=64,
        batch=2,
        smoke=True,
        n_experts=4,
        top_k=2,
    )
    vdir = Path(results["reports_dir"])
    for name in (PRIMARY_LOSS, PRIMARY_LOAD, PRIMARY_SPEED):
        p = vdir / name
        assert p.is_file() and p.stat().st_size > 0, name
    assert results["n_moe_total"] > results["n_dense"]
    assert results["use_ce_aux"] is False
    # smoke loss_ok may use slope; require gates computed
    assert "gates" in results
    assert results["gates"]["moe_bigger"] is True
    assert results["gates"]["aux_off"] is True
    assert results["gates"]["router_fp32"] is True
