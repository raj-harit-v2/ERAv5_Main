"""Part 1 memory: peak full CE vs hand-written chunked CE + ratio."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.llm.chunked_ce import ChunkedCrossEntropy, full_cross_entropy
from src.utils.peak_vram import bytes_to_mib, estimate_logits_bytes, track_peak_bytes


def measure_full_vs_chunked(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> dict:
    """
    hidden: [N, D] or [B, T, D]
    weight: [V, D] lm head weight
    targets: [N] or [B, T]
    Materialise full logits once; chunked path materialises per chunk.
    """
    device = hidden.device
    if hidden.dim() == 3:
        b, t, d = hidden.shape
        hidden_flat = hidden.reshape(b * t, d)
        targets_flat = targets.reshape(b * t)
    else:
        hidden_flat = hidden
        targets_flat = targets

    n, d = hidden_flat.shape
    v = weight.shape[0]

    # Full CE path
    with track_peak_bytes(device) as full_stats:
        logits_full = F.linear(hidden_flat, weight)  # [N, V]
        if device.type != "cuda":
            full_stats["_note"](logits_full)
        loss_full = full_cross_entropy(logits_full, targets_flat, ignore_index=ignore_index)
        # Keep reference for parity
        loss_full_val = float(loss_full.detach().cpu())

    # Chunked path: never hold full [N,V] if N > chunk
    chunked = ChunkedCrossEntropy(chunk_size=chunk_size, ignore_index=ignore_index)
    with track_peak_bytes(device) as chunk_stats:
        peak_proxy = 0
        total = hidden_flat.new_zeros(())
        count = 0
        for i in range(0, n, chunk_size):
            sl = slice(i, min(i + chunk_size, n))
            chunk_h = hidden_flat[sl]
            chunk_t = targets_flat[sl]
            chunk_logits = F.linear(chunk_h, weight)
            if device.type != "cuda":
                peak_proxy = max(peak_proxy, chunk_logits.numel() * chunk_logits.element_size())
                chunk_stats["_note"](chunk_logits)
            loss_sum = F.cross_entropy(
                chunk_logits, chunk_t, ignore_index=ignore_index, reduction="sum"
            )
            valid = (chunk_t != ignore_index).sum().item()
            total = total + loss_sum
            count += valid
            del chunk_logits
        loss_chunk = total / max(count, 1)
        loss_chunk_val = float(loss_chunk.detach().cpu())
        if device.type != "cuda":
            chunk_stats["peak_bytes"] = peak_proxy

    # Also report theoretical activation estimate
    full_est = estimate_logits_bytes(n, v, bytes_per=4)
    chunk_est = estimate_logits_bytes(min(chunk_size, n), v, bytes_per=4)

    peak_full = full_stats["peak_bytes"] or full_est
    peak_chunk = chunk_stats["peak_bytes"] or chunk_est
    ratio = peak_full / max(peak_chunk, 1)

    return {
        "loss_full": loss_full_val,
        "loss_chunked": loss_chunk_val,
        "loss_abs_diff": abs(loss_full_val - loss_chunk_val),
        "peak_full_bytes": int(peak_full),
        "peak_chunked_bytes": int(peak_chunk),
        "peak_full_mib": bytes_to_mib(int(peak_full)),
        "peak_chunked_mib": bytes_to_mib(int(peak_chunk)),
        "ratio_full_over_chunked": float(ratio),
        "backend": full_stats["backend"],
        "chunk_size": chunk_size,
        "n_tokens": n,
        "vocab": v,
        "estimate_full_bytes": full_est,
        "estimate_chunk_bytes": chunk_est,
    }
