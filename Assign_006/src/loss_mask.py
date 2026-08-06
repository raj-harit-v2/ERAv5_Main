"""Agentic loss masking: green = model tokens, grey = observations/user."""
from __future__ import annotations

import re
from typing import Any

_GREEN = ("plan", "tool", "final")
_TAG_RE = re.compile(r"<(user|plan|tool|obs|final)>(.*?)</\1>", re.DOTALL)


def mask_agentic_text(text: str) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    matches = list(_TAG_RE.finditer(text))
    if not matches:
        return {
            "spans": [{"start": 0, "end": len(text), "role": "lm", "loss": True}],
            "n_loss_chars": len(text),
            "n_context_chars": 0,
            "tagged": False,
        }
    for m in matches:
        role = m.group(1)
        start, end = m.span()
        spans.append({"start": start, "end": end, "role": role, "loss": role in _GREEN})
    n_loss = sum(s["end"] - s["start"] for s in spans if s["loss"])
    n_ctx = sum(s["end"] - s["start"] for s in spans if not s["loss"])
    return {
        "spans": spans,
        "n_loss_chars": n_loss,
        "n_context_chars": n_ctx,
        "tagged": True,
    }


def agentic_token_loss_mask(text: str, token_ids: list[int], pad_id: int = 0) -> list[float]:
    """Approximate char-span roles onto tokens by proportional allocation."""
    info = mask_agentic_text(text)
    n = len(token_ids)
    if n == 0:
        return []
    if not info["tagged"]:
        return [0.0 if t == pad_id else 1.0 for t in token_ids]

    # Map each char to loss flag
    char_loss = [1.0] * max(1, len(text))
    for s in info["spans"]:
        for i in range(s["start"], min(s["end"], len(char_loss))):
            char_loss[i] = 1.0 if s["loss"] else 0.0

    # Allocate chars proportionally across non-pad tokens
    real_idx = [i for i, t in enumerate(token_ids) if t != pad_id]
    if not real_idx:
        return [0.0] * n
    mask = [0.0] * n
    for j, ti in enumerate(real_idx):
        c0 = int(j * len(char_loss) / len(real_idx))
        c1 = int((j + 1) * len(char_loss) / len(real_idx))
        chunk = char_loss[c0:c1] or [1.0]
        mask[ti] = 1.0 if sum(chunk) / len(chunk) >= 0.5 else 0.0
    return mask


def assert_obs_not_trained(mask_info: dict[str, Any]) -> bool:
    for s in mask_info.get("spans", []):
        if s.get("role") == "obs" and s.get("loss"):
            return False
    return True
