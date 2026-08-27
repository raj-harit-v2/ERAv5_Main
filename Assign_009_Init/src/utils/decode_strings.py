"""Decode token ids to strings for shift verification (never print ID walls only)."""

from __future__ import annotations

from typing import Sequence

from src.llm.tokenizer import WordTokenizer


def print_shift_table(
    tokenizer: WordTokenizer,
    tokens_1d: Sequence[int],
    max_rows: int = 16,
) -> list[tuple[str, str]]:
    """
    Inputs = tokens[:-1], targets = tokens[1:].
    Print string pairs side by side.
    """
    ids = list(tokens_1d)
    inputs = ids[:-1]
    targets = ids[1:]
    pairs: list[tuple[str, str]] = []
    print(f"{'Input Token':<20} | {'Target Token (next)':<20}")
    print("-" * 45)
    for inp, tgt in zip(inputs[:max_rows], targets[:max_rows]):
        si = tokenizer.decode_one(int(inp))
        st = tokenizer.decode_one(int(tgt))
        pairs.append((si, st))
        print(f"{si:<20} | {st:<20}")
    if len(inputs) > max_rows:
        print(f"... ({len(inputs) - max_rows} more pairs)")
    return pairs


def verify_shift_strings(
    tokenizer: WordTokenizer,
    tokens_1d: Sequence[int],
) -> bool:
    """True if every printed pair is consecutive in the original string stream."""
    ids = list(tokens_1d)
    decoded = tokenizer.decode(ids)
    for i, (a, b) in enumerate(zip(decoded[:-1], decoded[1:])):
        if tokenizer.decode_one(ids[i]) != a or tokenizer.decode_one(ids[i + 1]) != b:
            return False
    return True
