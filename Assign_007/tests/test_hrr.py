"""Unit tests for HRR bind/unbind."""
from __future__ import annotations

import torch

from src.hrr import bind, cosine, random_hv, unbind, unbind_div


def test_bind_preserves_dim():
    d = 2048
    a, b = random_hv(d, seed=1), random_hv(d, seed=2)
    c = bind(a, b)
    assert c.shape == a.shape


def test_unbind_identity():
    d = 2048
    french = random_hv(d, seed=10)
    apple = random_hv(d, seed=11)
    bound = bind(french, apple)
    retrieved = unbind(bound, french)
    assert cosine(retrieved, apple) > 0.99
    assert abs(cosine(retrieved, french)) < 0.15


def test_conjugate_stable_vs_division():
    d = 512
    a, b = random_hv(d, seed=3), random_hv(d, seed=4)
    bound = bind(a, b)
    conj_ok = cosine(unbind(bound, a), b) > 0.95
    div_v = unbind_div(bound, a)
    assert conj_ok
    assert torch.isfinite(div_v).all()
