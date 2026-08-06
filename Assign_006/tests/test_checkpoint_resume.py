"""Checkpoint and resume tests."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from src.checkpoint_manager import CheckpointManager
from src.ledger import ConsumptionLedger
from src.replay_engine import ReplayEngine
from src.tiny_model import TinyTransformerLM
from src.utils import set_seed


def test_checkpoint_saves_all_6_components(tmp_path):
    set_seed(0)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    mgr = CheckpointManager(tmp_path / "ckpts")
    cid = mgr.save(
        step=3,
        model=model,
        optimizer=opt,
        scheduler=sched,
        ledger_offset=10,
        loss=2.0,
        run_id="r",
        branch_id="main",
        expected_next_batch_id="r:4:0",
    )
    d = tmp_path / "ckpts" / cid
    for name in ("model.pt", "optimizer.pt", "scheduler.pt", "rng_state.pt", "metadata.json"):
        assert (d / name).exists()
    # sixth conceptual component: ledger_offset inside metadata
    import json

    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ledger_offset"] == 10


def test_resume_next_batch_matches_expected(tmp_path):
    set_seed(1)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts")
    expected = "runx:12:0"
    cid = mgr.save(
        step=11,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=0,
        loss=1.0,
        run_id="runx",
        branch_id="main",
        expected_next_batch_id=expected,
    )
    led = ConsumptionLedger(tmp_path / "c.jsonl")
    eng = ReplayEngine(mgr, led)
    meta = eng.resume(cid, model, opt, None, expected_batch_id=expected)
    assert meta["expected_next_batch_id"] == expected


def test_rng_state_restored_correctly(tmp_path):
    set_seed(7)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts")
    _ = torch.rand(3)
    cid = mgr.save(
        step=1,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=0,
        loss=1.0,
        run_id="r",
        branch_id="main",
    )
    expected = torch.rand(3)
    _ = torch.rand(3)  # diverge
    mgr.load(cid, model, opt, None)
    restored = torch.rand(3)
    assert torch.allclose(expected, restored)


def test_optimizer_state_restored(tmp_path):
    set_seed(3)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    out["loss"].backward()
    opt.step()
    mgr = CheckpointManager(tmp_path / "ckpts")
    cid = mgr.save(
        step=2,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=0,
        loss=float(out["loss"].detach()),
        run_id="r",
        branch_id="main",
    )
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr.load(cid, model, opt2, None)
    assert opt2.state_dict()["state"]
