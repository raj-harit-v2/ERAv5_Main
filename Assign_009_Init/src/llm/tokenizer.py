"""Simple whitespace word tokenizer for hermetic Shakespeare smoke."""

from __future__ import annotations

import re
from pathlib import Path


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL = [PAD, BOS, EOS, UNK]


class WordTokenizer:
    """Whitespace + punctuation split; deterministic vocab from corpus."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.pad_id = token_to_id[PAD]
        self.bos_id = token_to_id[BOS]
        self.eos_id = token_to_id[EOS]
        self.unk_id = token_to_id[UNK]
        self.vocab_size = len(token_to_id)

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        # Keep punctuation as separate tokens for readable shift tables.
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", text)

    @classmethod
    def from_corpus(cls, text: str) -> "WordTokenizer":
        counts: dict[str, int] = {}
        for tok in cls.tokenize_text(text):
            counts[tok] = counts.get(tok, 0) + 1
        # Stable order: specials first, then alpha-sorted types.
        types = sorted(counts.keys())
        token_to_id = {t: i for i, t in enumerate(SPECIAL)}
        for t in types:
            if t not in token_to_id:
                token_to_id[t] = len(token_to_id)
        return cls(token_to_id)

    @classmethod
    def from_file(cls, path: str | Path) -> "WordTokenizer":
        return cls.from_corpus(Path(path).read_text(encoding="utf-8"))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in self.tokenize_text(text):
            ids.append(self.token_to_id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK) for i in ids]

    def decode_one(self, tid: int) -> str:
        return self.id_to_token.get(tid, UNK)
