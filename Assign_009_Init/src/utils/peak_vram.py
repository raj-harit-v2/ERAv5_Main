"""Peak memory measurement: CUDA peak allocated or labeled CPU proxy."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def track_peak_bytes(device: torch.device) -> Iterator[dict]:
    """
    Yields a dict that will contain peak_bytes and backend label after exit.
    """
    result: dict = {"peak_bytes": 0, "backend": "cpu_proxy"}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        result["backend"] = "cuda"
        try:
            yield result
        finally:
            torch.cuda.synchronize(device)
            result["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        # CPU proxy: track max tensor nbytes allocated inside the block via hook list.
        peaks = [0]

        def _note(t: torch.Tensor) -> None:
            peaks[0] = max(peaks[0], t.numel() * t.element_size())

        result["_note"] = _note
        result["backend"] = "cpu_proxy"
        try:
            yield result
        finally:
            result["peak_bytes"] = int(peaks[0])


def bytes_to_mib(n: int) -> float:
    return n / (1024 ** 2)


def estimate_logits_bytes(n_tokens: int, vocab: int, bytes_per: int = 4) -> int:
    """Activation estimate for materialised logits [N, V]."""
    return n_tokens * vocab * bytes_per
