"""Eval firewall tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import CANARY_EVAL, generate_corpus
from src.eval_firewall import EvalFirewall


def test_eval_shard_blocked_from_training():
    corpus = generate_corpus(docs_per_lane=2, seed=4)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    ok, reason = fw.check_document(corpus["eval"][0])
    assert not ok
    assert "content_hash" in reason or "doc_id" in reason


def test_canary_string_detected():
    fw = EvalFirewall()
    ok, reason = fw.check_text(f"noise {CANARY_EVAL} noise")
    assert not ok
    assert reason == "canary_detected"


def test_clean_shard_passes_firewall():
    corpus = generate_corpus(docs_per_lane=2, seed=4)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    ok, reason = fw.check_document(corpus["web"][0])
    assert ok
    assert reason == ""


def test_rejection_event_logged():
    fw = EvalFirewall()
    ev = fw.block(shard_id="shard_eval_x", reason="canary_detected")
    assert ev["event"] == "eval_shard_blocked"
    assert len(fw.blocked_events) == 1
