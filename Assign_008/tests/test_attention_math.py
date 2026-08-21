import math

import torch

from src.attention.scaled_dot_product import scaled_dot_product_attention


def test_scale_divides_by_sqrt_dk():
    torch.manual_seed(0)
    b, h, t, d = 1, 1, 4, 16
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    out, w = scaled_dot_product_attention(q, k, v, causal=False)
    raw = torch.matmul(q, k.transpose(-2, -1))
    scaled = raw / math.sqrt(d)
    # softmax of scaled should match weights when causal=False
    from torch.nn.functional import softmax

    expect = softmax(scaled, dim=-1)
    assert torch.allclose(w, expect, atol=1e-5)
    assert out.shape == (b, h, t, d)
