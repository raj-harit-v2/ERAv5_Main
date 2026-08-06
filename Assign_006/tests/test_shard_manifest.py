"""Shard manifest and admission gate tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import generate_corpus
from src.eval_firewall import EvalFirewall
from src.shard_builder import build_shard, compute_cleaning_hash
from src.shard_manifest import check_admission, validate_manifest
from src.tokenizer_wrapper import compute_tokenizer_hash, get_tokenizer
from src.utils import sha256_file


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()


@pytest.fixture
def corpus():
    return generate_corpus(docs_per_lane=4, eval_docs=2, test_docs=2, seed=1)


def test_manifest_all_fields_required(tmp_path, tok, corpus):
    th = compute_tokenizer_hash(tok)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    m = build_shard(
        corpus["web"][:2],
        tokenizer_hash=th,
        lane="web",
        sequence_length=64,
        shard_index=0,
        output_dir=tmp_path,
        cleaning_pipeline_hash=compute_cleaning_hash(),
        capability_lane="web",
        eval_registry=set(fw.doc_ids),
        tok=tok,
    )
    ok, errs = validate_manifest(m)
    assert ok, errs


def test_admission_gate_blocks_eval_overlap(tmp_path, tok, corpus):
    th = compute_tokenizer_hash(tok)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    with pytest.raises(ValueError, match="eval_overlap"):
        build_shard(
            corpus["eval"][:1],
            tokenizer_hash=th,
            lane="eval",
            sequence_length=64,
            shard_index=0,
            output_dir=tmp_path,
            cleaning_pipeline_hash=compute_cleaning_hash(),
            capability_lane="eval",
            eval_registry=set(fw.doc_ids),
            tok=tok,
        )


def test_admission_gate_blocks_missing_tokenizer_hash(tmp_path, tok, corpus):
    th = compute_tokenizer_hash(tok)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    m = build_shard(
        corpus["web"][:2],
        tokenizer_hash=th,
        lane="web",
        sequence_length=64,
        shard_index=1,
        output_dir=tmp_path,
        cleaning_pipeline_hash=compute_cleaning_hash(),
        capability_lane="web",
        eval_registry=set(fw.doc_ids),
        tok=tok,
    )
    m["tokenizer_hash"] = ""
    ok, reason = check_admission(m, tmp_path / m["shard_path"])
    assert not ok
    assert "tokenizer" in reason


def test_content_hash_verified(tmp_path, tok, corpus):
    th = compute_tokenizer_hash(tok)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    m = build_shard(
        corpus["code"][:2],
        tokenizer_hash=th,
        lane="code",
        sequence_length=64,
        shard_index=0,
        output_dir=tmp_path,
        cleaning_pipeline_hash=compute_cleaning_hash(),
        capability_lane="code",
        eval_registry=set(fw.doc_ids),
        tok=tok,
    )
    assert sha256_file(tmp_path / m["shard_path"]) == m["content_hash"]


def test_immutability_new_hash_on_modification(tmp_path, tok, corpus):
    th = compute_tokenizer_hash(tok)
    fw = EvalFirewall()
    fw.register_corpus_holdouts(corpus)
    m = build_shard(
        corpus["stem"][:2],
        tokenizer_hash=th,
        lane="stem",
        sequence_length=64,
        shard_index=0,
        output_dir=tmp_path,
        cleaning_pipeline_hash=compute_cleaning_hash(),
        capability_lane="stem",
        eval_registry=set(fw.doc_ids),
        tok=tok,
    )
    path = tmp_path / m["shard_path"]
    arr = np.load(path)
    arr[0, 0] = (int(arr[0, 0]) + 1) % 50
    alt = tmp_path / "mutated.npy"
    np.save(alt, arr)
    assert sha256_file(alt) != m["content_hash"]
