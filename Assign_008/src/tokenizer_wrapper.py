"""Hermetic whitespace tokenizer (HF gated off by default).

Assumption A3: dense path is default for attention tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import config


@dataclass
class EncodeResult:
    input_ids: torch.Tensor  # [B, T]
    token_strings: list[list[str]]


class HermeticTokenizer:
    """Tiny word-level map built from the demo sentence + reserved IDs."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE) -> None:
        self.pad_id = config.PAD_ID
        self.bos_id = config.BOS_ID
        self.eos_id = config.EOS_ID
        self.vocab_size = vocab_size
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {
            self.pad_id: "<pad>",
            self.bos_id: "<bos>",
            self.eos_id: "<eos>",
        }
        next_id = 3
        for w in "the cat sat on mat a is bank river".split():
            if w not in self._stoi and next_id < vocab_size:
                self._stoi[w] = next_id
                self._itos[next_id] = w
                next_id += 1
        self._unk = next_id if next_id < vocab_size else self.pad_id
        if self._unk not in self._itos:
            self._itos[self._unk] = "<unk>"

    def encode(self, text: str, seq_len: int = config.SEQ_LEN) -> EncodeResult:
        words = text.lower().replace(".", "").split()
        ids = [self.bos_id] + [self._stoi.get(w, self._unk) for w in words] + [self.eos_id]
        ids = ids[:seq_len]
        strings = ["<bos>"] + words[: max(0, seq_len - 2)] + (["<eos>"] if len(ids) > 1 else [])
        strings = strings[: len(ids)]
        if len(ids) < seq_len:
            pad_n = seq_len - len(ids)
            ids = ids + [self.pad_id] * pad_n
            strings = strings + ["<pad>"] * pad_n
        t = torch.tensor([ids], dtype=torch.long)
        return EncodeResult(input_ids=t, token_strings=[strings])

    def decode_ids(self, ids: list[int]) -> str:
        toks = [self._itos.get(i, "<unk>") for i in ids if i not in (self.pad_id, self.bos_id)]
        toks = [t for t in toks if t != "<eos>"]
        return " ".join(toks)
