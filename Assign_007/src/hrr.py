"""Holographic Reduced Representation (HRR) bind/unbind via Real FFT.

From Asgn_07_All_docs_xfr/HRR binding and unbinding TEST.md (Tony Plate HRR):
circular convolution in O(N log N) via torch.fft.rfft / irfft.
Unbinding uses circular correlation (complex conjugate), not division.

Vectors are projected to **unitary** (flat FFT magnitude) by default so
correlation is an exact inverse (cos ~ 1). Non-unitary Gaussians only
approximate (cos ~ 0.7) because retrieval applies |FFT(a)|^2 filtering.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def to_unitary(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Force flat spectrum so circular correlation inverts binding exactly."""
    n = x.shape[-1]
    fx = torch.fft.rfft(x)
    fx = fx / fx.abs().clamp_min(eps)
    return torch.fft.irfft(fx, n=n)


def random_hv(
    dim: int,
    seed: int | None = None,
    device: torch.device | None = None,
    *,
    unitary: bool = True,
) -> torch.Tensor:
    """Random hypervector; unitary=True (default) for exact HRR invertibility."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    v = torch.randn(dim, generator=g) / (dim**0.5)
    if unitary:
        v = to_unitary(v)
    if device is not None:
        v = v.to(device)
    return v


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item())


def bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Circular convolution via Real FFT (Convolution Theorem)."""
    if x.shape != y.shape:
        raise ValueError(f"bind shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    x_freq = torch.fft.rfft(x)
    y_freq = torch.fft.rfft(y)
    bound_freq = x_freq * y_freq
    return torch.fft.irfft(bound_freq, n=x.shape[-1])


def unbind(bound: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Circular correlation (conjugate) — exact inverse for unitary operands."""
    if bound.shape != x.shape:
        raise ValueError(f"unbind shape mismatch {tuple(bound.shape)} vs {tuple(x.shape)}")
    bound_freq = torch.fft.rfft(bound)
    x_freq = torch.fft.rfft(x)
    unbound_freq = bound_freq * torch.conj(x_freq)
    return torch.fft.irfft(unbound_freq, n=bound.shape[-1])


def unbind_div(bound: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Frequency division — fragile near spectral zeros (negative control)."""
    bound_freq = torch.fft.rfft(bound)
    x_freq = torch.fft.rfft(x)
    unbound_freq = bound_freq / (x_freq + eps)
    return torch.fft.irfft(unbound_freq, n=bound.shape[-1])


def dropout_noise(x: torch.Tensor, frac: float, seed: int = 0) -> torch.Tensor:
    """Zero out `frac` of elements (holographic robustness test)."""
    g = torch.Generator()
    g.manual_seed(seed)
    mask = torch.rand(x.shape, generator=g) >= frac
    return x * mask.to(dtype=x.dtype)


def bundle(*vectors: torch.Tensor) -> torch.Tensor:
    """Superposition (element-wise sum)."""
    out = vectors[0].clone()
    for v in vectors[1:]:
        out = out + v
    return out
