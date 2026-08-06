"""Ledger tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ledger import ConsumptionLedger, LearningLedger, read_jsonl


def test_consumption_ledger_append_only(tmp_path):
    path = tmp_path / "c.jsonl"
    led = ConsumptionLedger(path)
    led.append({"event": "consumption", "global_step": 0, "batch_id": "a"})
    led.append({"event": "consumption", "global_step": 1, "batch_id": "b"})
    rows = read_jsonl(path)
    assert len(rows) == 2
    # cannot update in place via API — only append
    assert rows[0]["batch_id"] == "a"


def test_ledger_offset_advances_correctly(tmp_path):
    path = tmp_path / "c.jsonl"
    led = ConsumptionLedger(path)
    o0 = led.append({"event": "consumption", "global_step": 0})
    o1 = led.append({"event": "consumption", "global_step": 1})
    assert o1 > o0
    assert led.offset > o1


def test_learning_ledger_shard_classification(tmp_path):
    led = LearningLedger(tmp_path / "l.jsonl")
    assert led.classify(-0.05, 1.0) == "useful"
    assert led.classify(0.1, 1.0) == "harmful"
    assert led.classify(0.0, 1.0) == "neutral"


def test_ppl_threshold_skip_fires_at_1_2():
    import config as cfg

    assert cfg.PPL_THRESHOLD_SKIP == 1.2
    mastered = 1.1 < cfg.PPL_THRESHOLD_SKIP
    assert mastered
