"""Encode docs to padded tensors + token strings for seam-crossing codecs."""
from __future__ import annotations

import torch

import config as cfg
from src.corpus_indic_toy import Doc
from src.tokenizer_wrapper import LocalHashTokenizer


def encode_docs(
    docs: list[Doc],
    tok: LocalHashTokenizer,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, list[list[str]]]:
    seq_len = seq_len or cfg.SEQ_LEN
    rows_ids: list[list[int]] = []
    rows_str: list[list[str]] = []
    for d in docs:
        ids = tok.encode(d.text)[: seq_len]
        strs = [tok.decode_id(i) for i in ids]
        # pad
        while len(ids) < seq_len:
            ids.append(cfg.PAD_ID)
            strs.append("<pad>")
        rows_ids.append(ids)
        rows_str.append(strs)
    return torch.tensor(rows_ids, dtype=torch.long), rows_str
