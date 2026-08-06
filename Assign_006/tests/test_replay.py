"""Replay and fork tests."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_manager import CheckpointManager
from src.ledger import ConsumptionLedger
from src.replay_engine import ReplayEngine, ReplayIntegrityError
from src.tiny_model import TinyTransformerLM
from src.utils import batch_content_hash, set_seed


def test_replay_hash_matches_original(tmp_path):
    tokens = [1, 2, 3, 4, 5]
    h = batch_content_hash(tokens)
    orig = [
        {
            "global_step": 0,
            "batch_id": "r:0:0",
            "token_span_ids": [{"seq": 0, "start": 0, "end": 5}],
            "batch_hash": h,
        }
    ]
    rebuilt = [
        {
            "batch_id": "r:0:0",
            "token_span_ids": [{"seq": 0, "start": 0, "end": 5}],
            "token_ids": tokens,
        }
    ]
    eng = ReplayEngine(CheckpointManager(tmp_path / "c"), ConsumptionLedger(tmp_path / "l.jsonl"))
    reports = eng.replay(orig, rebuilt)
    assert reports[0]["ok"]


def test_fork_creates_new_branch_id(tmp_path):
    set_seed(0)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts")
    cid = mgr.save(
        step=5,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=0,
        loss=1.0,
        run_id="runA",
        branch_id="main",
    )
    led = ConsumptionLedger(tmp_path / "c.jsonl")
    eng = ReplayEngine(mgr, led)
    fork = eng.fork(cid, model, opt, None)
    assert fork["new_branch_id"] != "main"
    assert fork["original_run_id"] == "runA"


def test_fork_point_recorded_in_ledger(tmp_path):
    set_seed(0)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts")
    cid = mgr.save(
        step=5,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=0,
        loss=1.0,
        run_id="runA",
        branch_id="main",
    )
    led = ConsumptionLedger(tmp_path / "c.jsonl")
    eng = ReplayEngine(mgr, led)
    eng.fork(cid, model, opt, None)
    assert any(r.get("event") == "fork_point" for r in led.records)


def test_resume_zero_skipped_zero_repeated_batches(tmp_path):
    """Resume metadata next batch equals expected; no skip/repeat at boundary."""
    set_seed(0)
    model = TinyTransformerLM(vocab_size=64, max_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts")
    expected = "run:10:0"
    cid = mgr.save(
        step=9,
        model=model,
        optimizer=opt,
        scheduler=None,
        ledger_offset=100,
        loss=1.0,
        run_id="run",
        branch_id="main",
        expected_next_batch_id=expected,
        dataloader_state={"next_step": 10},
    )
    led = ConsumptionLedger(tmp_path / "c.jsonl")
    eng = ReplayEngine(mgr, led)
    meta = eng.resume(cid, model, opt, None, expected_batch_id=expected)
    assert meta["dataloader_state"]["next_step"] == 10
    assert meta["expected_next_batch_id"] == expected


def test_replay_mismatch_raises(tmp_path):
    orig = [
        {
            "global_step": 0,
            "batch_id": "r:0:0",
            "token_span_ids": [],
            "batch_hash": "abc",
        }
    ]
    rebuilt = [{"batch_id": "r:0:0", "token_span_ids": [], "token_ids": [1, 2]}]
    eng = ReplayEngine(CheckpointManager(tmp_path / "c"), ConsumptionLedger(tmp_path / "l.jsonl"))
    try:
        eng.replay(orig, rebuilt)
        assert False, "expected ReplayIntegrityError"
    except ReplayIntegrityError:
        pass
