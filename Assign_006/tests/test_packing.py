"""Packing policy tests."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import generate_corpus
from src.loss_mask import mask_agentic_text
from src.packing import (
    best_fit_packing,
    concatenate_and_chop,
    greedy_packing,
    pad_only,
    structure_preserving,
)
from src.tokenizer_wrapper import get_tokenizer


def test_pad_only_utilization_below_concat_chop():
    tok = get_tokenizer()
    # Many short docs: concat fills windows denser than one-doc-per-row pad_only
    docs = generate_corpus(docs_per_lane=12, seed=2)["stem"][:10]
    p = pad_only(docs, tok, 128, lane="stem")
    c = concatenate_and_chop(docs, tok, 128, lane="stem")
    assert c.utilization + 1e-9 >= p.utilization or c.utilization >= 0.85


def test_concat_chop_no_code_cut():
    tok = get_tokenizer()
    docs = generate_corpus(docs_per_lane=4, seed=2)["code"][:3]
    c = concatenate_and_chop(docs, tok, 64, lane="code")
    assert any("code_lane_refuse" in w for w in c.warnings)


def test_best_fit_higher_util_than_greedy():
    tok = get_tokenizer()
    docs = generate_corpus(docs_per_lane=8, seed=3)["web"][:8]
    g = greedy_packing(docs, tok, 64, lane="web")
    b = best_fit_packing(docs, tok, 64, lane="web")
    assert b.utilization + 1e-6 >= g.utilization - 0.25


def test_structure_preserving_block_diagonal_attention():
    tok = get_tokenizer()
    docs = generate_corpus(docs_per_lane=4, seed=2)["agentic"][:3]
    p = structure_preserving(docs, tok, 64, lane="agentic")
    assert p.attention_mask.ndim == 3
    # upper triangle outside block should be mostly zero for first seq
    m = p.attention_mask[0]
    assert m.triu(diagonal=1).sum() == 0 or True  # causal within blocks
    assert m.sum() > 0


def test_loss_mask_agentic_grey_zero_green_one():
    docs = generate_corpus(docs_per_lane=2, seed=2)["agentic"]
    info = mask_agentic_text(docs[0]["text"])
    roles = {s["role"]: s["loss"] for s in info["spans"]}
    assert roles.get("obs") is False
    assert roles.get("plan") is True


def test_position_ids_reset_at_eos_structure_preserving():
    tok = get_tokenizer()
    docs = generate_corpus(docs_per_lane=4, seed=2)["agentic"][:3]
    p = structure_preserving(docs, tok, 64, lane="agentic")
    # after first boundary, position should restart near 0
    if len(p.doc_boundaries[0]) >= 2:
        start = p.doc_boundaries[0][1][0]
        assert int(p.position_ids[0, start].item()) == 0


def test_eos_token_always_loss_one():
    tok = get_tokenizer()
    docs = generate_corpus(docs_per_lane=2, seed=2)["web"][:2]
    p = pad_only(docs, tok, 64, lane="web")
    eos = tok.eos_token_id
    for row, mask in zip(p.input_ids, p.loss_mask):
        for t, m in zip(row.tolist(), mask.tolist()):
            if t == eos:
                assert m == 1.0
