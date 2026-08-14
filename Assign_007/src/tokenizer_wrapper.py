"""Hermetic whitespace/char tokenizer with stable string table for codecs."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils import sha256_text


_TOKEN_RE = re.compile(r"\S+|\s+")


@dataclass
class LocalHashTokenizer:
    vocab_size: int
    pad_id: int = 0
    unk_id: int = 1
    id_to_token: dict[int, str] | None = None
    token_to_id: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.id_to_token = self.id_to_token or {self.pad_id: "<pad>", self.unk_id: "<unk>"}
        self.token_to_id = self.token_to_id or {"<pad>": self.pad_id, "<unk>": self.unk_id}

    def fit(self, texts: list[str]) -> None:
        assert self.token_to_id is not None and self.id_to_token is not None
        counts: dict[str, int] = {}
        for t in texts:
            for tok in _TOKEN_RE.findall(t):
                if tok.isspace():
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        next_id = 2
        for tok, _ in ranked:
            if next_id >= self.vocab_size:
                break
            if tok in self.token_to_id:
                continue
            self.token_to_id[tok] = next_id
            self.id_to_token[next_id] = tok
            next_id += 1

    def encode(self, text: str) -> list[int]:
        assert self.token_to_id is not None
        ids: list[int] = []
        for tok in _TOKEN_RE.findall(text):
            if tok.isspace():
                continue
            ids.append(self.token_to_id.get(tok, self.unk_id))
        return ids

    def decode_id(self, idx: int) -> str:
        assert self.id_to_token is not None
        return self.id_to_token.get(idx, "<unk>")

    def token_string(self, idx: int) -> str:
        return self.decode_id(idx)

    def hash(self) -> str:
        assert self.id_to_token is not None
        payload = "|".join(f"{i}:{self.id_to_token[i]}" for i in sorted(self.id_to_token))
        return sha256_text(payload)

    def vocabulary_strings(self) -> list[str]:
        assert self.id_to_token is not None
        return [self.id_to_token[i] for i in sorted(self.id_to_token) if i not in (self.pad_id,)]
