"""Unit tests for Assign_007 Problem #4 (+ RoPE / CANINE upgrades)."""
from __future__ import annotations

import math

import torch

import config as cfg
from src.collision_audit import collision_report
from src.embedding_policy import build_embedding_policy
from src.embeddings import KroneckerByteEmbedding, build_embedding
from src.embeddings.canine_vq import FourierCanineEmbedding
from src.embeddings.fourier_baseline import FourierCodepointEmbedding, znormalize
from src.rope import apply_rope, build_rope_cache


def test_fourier_determinism():
    f = FourierCodepointEmbedding(8, code_dim=cfg.FOURIER_CODE_DIM, n_freq=cfg.FOURIER_N_FREQ)
    assert torch.allclose(f.encode_string("हिंदी"), f.encode_string("हिंदी"))


def test_fourier_anagram():
    f = FourierCodepointEmbedding(8, code_dim=cfg.FOURIER_CODE_DIM, n_freq=cfg.FOURIER_N_FREQ)
    assert not torch.allclose(f.encode_string("अब"), f.encode_string("बा"))


def test_fourier_composition():
    f = FourierCodepointEmbedding(8, code_dim=cfg.FOURIER_CODE_DIM, n_freq=cfg.FOURIER_N_FREQ)
    text = "राम"
    waves = [f.wave(ord(ch), i) for i, ch in enumerate(text)]
    composed = znormalize(torch.stack(waves).sum(0) / math.sqrt(len(waves)))
    assert torch.allclose(composed, f.encode_string(text), atol=1e-5)


def test_param_independent_of_vocab():
    a = build_embedding(
        kind="fourier",
        vocab_size=64,
        d_model=32,
        pad_id=0,
        pos_dim=32,
        fourier_code_dim=128,
        fourier_n_freq=8,
        max_chars=32,
    )
    b = build_embedding(
        kind="fourier",
        vocab_size=512,
        d_model=32,
        pad_id=0,
        pos_dim=32,
        fourier_code_dim=128,
        fourier_n_freq=8,
        max_chars=32,
    )
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_kronecker_collision_audit_runs():
    report = collision_report(["apple", "train", "भारत", "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"])
    assert "kronecker" in report and "fourier" in report
    assert "32" in report["kronecker"]


def test_policy_json_schema():
    pol = build_embedding_policy("a" * 64)
    assert pol["problem_id"] == 4
    assert pol["embedding_type"] == "fourier_codepoint_sum"
    assert pol["position_policy"] == "rope"


def test_rope_and_canine():
    cos, sin = build_rope_cache(4, 8, torch.device("cpu"))
    x = torch.randn(1, 2, 4, 8)
    assert apply_rope(x, cos, sin).shape == x.shape
    c = FourierCanineEmbedding(d_model=16, code_dim=64, n_freq=8, max_chars=16, stride=2)
    assert not torch.allclose(c.encode_string("अब"), c.encode_string("बा"))
