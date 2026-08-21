import torch

from src.attention.delta_rule import delta_write, normalize_key
from src.attention.linear_state import build_state, direct_no_softmax, read_state


def test_linear_associative_matches_direct():
    torch.manual_seed(2)
    t, d = 5, 4
    keys = torch.randn(t, d)
    values = torch.randn(t, d)
    q = torch.randn(d)
    s = build_state(keys, values)
    y_state = read_state(s, q)
    y_direct = direct_no_softmax(q, keys, values)
    assert torch.allclose(y_state, y_direct, atol=1e-5)


def test_delta_normed_overwrite():
    d = 3
    s = torch.zeros(d, d)
    k = normalize_key(torch.tensor([1.0, 2.0, 3.0]))
    v1 = torch.tensor([4.0, 5.0, 6.0])
    s = delta_write(s, v1, k)
    assert torch.allclose(s @ k, v1, atol=1e-5)
    v2 = torch.tensor([7.0, 8.0, 9.0])
    s = delta_write(s, v2, k)
    assert torch.allclose(s @ k, v2, atol=1e-5)
