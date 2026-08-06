"""Frozen tokenizer wrapper with SHA-256 hash binding."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any

import config as cfg


class LocalHashTokenizer:
    """Hermetic offline tokenizer (no internet). Maps words/bytes into fixed vocab."""

    def __init__(self, vocab_size: int = cfg.LOCAL_VOCAB_SIZE, eos_id: int = cfg.LOCAL_EOS_ID):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_id
        self.pad_token_id = cfg.PAD_TOKEN_ID
        self.model_max_length = 1024
        self.name_or_path = "local-hash-tokenizer-v1"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        ids: list[int] = []
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            # reserve 0=pad, eos_id=eos
            tid = 1 + (h % (self.vocab_size - 2))
            if tid == self.eos_token_id:
                tid = 1
            ids.append(tid)
        return ids

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


def compute_tokenizer_hash(tok: Any) -> str:
    payload = f"{tok.vocab_size}:{tok.__class__.__name__}:{getattr(tok, 'model_max_length', 0)}"
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=2)
def get_tokenizer(model_name: str | None = None, use_hf: bool | None = None) -> Any:
    use_hf = cfg.USE_HF_TOKENIZER if use_hf is None else use_hf
    model_name = model_name or cfg.TOKENIZER_HF_NAME
    if use_hf:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception:
            pass
    return LocalHashTokenizer()


def effective_eos_id(tok: Any) -> int:
    return int(getattr(tok, "eos_token_id", None) or cfg.LOCAL_EOS_ID)


def encode_document(
    text: str,
    tok: Any,
    sequence_length: int,
    add_eos: bool = True,
) -> list[int]:
    """Tokenize text, append EOS, truncate to sequence_length (no pad here)."""
    if hasattr(tok, "encode"):
        ids = list(tok.encode(text, add_special_tokens=False))
    else:
        ids = list(tok(text)["input_ids"])
    eos = effective_eos_id(tok)
    if add_eos:
        if not ids or ids[-1] != eos:
            ids.append(eos)
    if len(ids) > sequence_length:
        ids = ids[: sequence_length - 1] + [eos]
    return ids
