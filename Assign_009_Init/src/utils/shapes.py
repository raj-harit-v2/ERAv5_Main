"""Print tensor shapes with one-line dimension legends."""

from __future__ import annotations

from typing import Any

import torch


def explain_shape(name: str, tensor: torch.Tensor, legend: str) -> dict[str, Any]:
    info = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "legend": legend,
    }
    print(f"{name}: shape={info['shape']} dtype={info['dtype']} | {legend}")
    return info


def dump_batch_shapes(
    tokens: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows = [
        explain_shape("tokens", tokens, "B=batch, T=sequence length (token ids)"),
        explain_shape("hidden", hidden, "B=batch, T=seq, D=model width"),
        explain_shape("logits", logits, "B=batch, T=seq, V=vocab (pre-softmax scores)"),
        explain_shape("targets", targets, "B=batch, T-1=next-token ids after shift"),
    ]
    if mask is not None:
        rows.append(
            explain_shape("mask", mask, "B=batch, T-1=1 contribute / 0 ignore")
        )
    return rows
