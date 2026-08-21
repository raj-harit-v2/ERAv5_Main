import torch

from src.attention.scaled_dot_product import causal_mask, scaled_dot_product_attention


def test_causal_mask_zeros_future():
    torch.manual_seed(1)
    b, h, t, d = 1, 1, 5, 8
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    _, w = scaled_dot_product_attention(q, k, v, causal=True)
    # strict upper triangle (future) ~ 0
    for i in range(t):
        for j in range(i + 1, t):
            assert w[0, 0, i, j].item() < 1e-6


def test_mask_matrix_values():
    m = causal_mask(4)
    assert m[0, 0].item() == 0.0
    assert m[2, 1].item() == 0.0
    assert m[1, 2].item() == float("-inf")
