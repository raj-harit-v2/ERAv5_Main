"""Five packing policies with loss/attention masks and position IDs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import config as cfg
from src.loss_mask import agentic_token_loss_mask
from src.tokenizer_wrapper import encode_document, effective_eos_id


@dataclass
class PackedBatch:
    input_ids: torch.LongTensor
    loss_mask: torch.FloatTensor
    attention_mask: torch.LongTensor
    position_ids: torch.LongTensor
    doc_boundaries: list[list[tuple[int, int]]]
    document_ids: list[str]
    packing_policy: str
    utilization: float
    lane: str = "web"
    warnings: list[str] = field(default_factory=list)
    row_document_ids: list[list[str]] = field(default_factory=list)

    def token_list(self) -> list[int]:
        return self.input_ids.detach().cpu().numpy().astype(np.int64).flatten().tolist()


def _pad_row(ids: list[int], seq_len: int, pad_id: int) -> list[int]:
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [pad_id] * (seq_len - len(ids))


def _doc_token_ids(
    d: dict[str, Any],
    tok: Any,
    sequence_length: int,
    *,
    add_eos: bool = True,
) -> list[int]:
    """Prefer immutable shard row when present; else tokenize text."""
    pad = cfg.PAD_TOKEN_ID
    eos = effective_eos_id(tok)
    if d.get("_input_ids"):
        ids = [int(x) for x in d["_input_ids"]]
        while ids and ids[-1] == pad:
            ids.pop()
        if add_eos and (not ids or ids[-1] != eos):
            ids = ids + [eos]
        return ids[:sequence_length]
    return encode_document(d.get("text", ""), tok, sequence_length, add_eos=add_eos)


def _loss_for_lane(lane: str, text: str, ids: list[int], pad_id: int, eos_id: int) -> list[float]:
    if lane == "agentic":
        mask = agentic_token_loss_mask(text, ids, pad_id=pad_id)
    else:
        mask = [0.0 if t == pad_id else 1.0 for t in ids]
    # EOS always trains
    for i, t in enumerate(ids):
        if t == eos_id:
            mask[i] = 1.0
    return mask


def _linear_positions(seq_len: int) -> list[int]:
    return list(range(seq_len))


def _reset_positions(boundaries: list[tuple[int, int]], seq_len: int) -> list[int]:
    pos = [0] * seq_len
    for start, end in boundaries:
        for i, p in enumerate(range(start, end)):
            if p < seq_len:
                pos[p] = i
    return pos


def _attention_causal(seq_len: int) -> list[list[int]]:
    # Store as 1D usable mask: 1 = real token position (causal applied in model)
    return [[1] * seq_len]


def _attention_block_diag(boundaries: list[tuple[int, int]], seq_len: int) -> torch.Tensor:
    """Block-diagonal causal: token i attends to j only if same doc and j<=i."""
    m = torch.zeros(seq_len, seq_len, dtype=torch.long)
    for start, end in boundaries:
        for i in range(start, end):
            for j in range(start, i + 1):
                m[i, j] = 1
    return m


def pad_only(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str = "web",
) -> PackedBatch:
    eos = effective_eos_id(tok)
    pad = cfg.PAD_TOKEN_ID
    rows, masks, pos_rows, bounds_all, doc_ids = [], [], [], [], []
    useful = 0
    total = 0
    for d in documents:
        ids = _doc_token_ids(d, tok, sequence_length, add_eos=True)
        ids = _pad_row(ids, sequence_length, pad)
        lm = _loss_for_lane(lane, d.get("text", ""), ids, pad, eos)
        end = next((i for i, t in enumerate(ids) if t == pad), sequence_length)
        bounds = [(0, end)]
        rows.append(ids)
        masks.append(lm)
        pos_rows.append(_linear_positions(sequence_length))
        bounds_all.append(bounds)
        doc_ids.append(d["doc_id"])
        useful += sum(1 for t in ids if t != pad)
        total += sequence_length
    util = useful / max(1, total)
    return PackedBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        loss_mask=torch.tensor(masks, dtype=torch.float),
        attention_mask=torch.ones(len(rows), sequence_length, dtype=torch.long),
        position_ids=torch.tensor(pos_rows, dtype=torch.long),
        doc_boundaries=bounds_all,
        document_ids=doc_ids,
        packing_policy="pad_only",
        utilization=util,
        lane=lane,
        row_document_ids=[[d] for d in doc_ids],
    )


def concatenate_and_chop(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str = "web",
) -> PackedBatch:
    warnings: list[str] = []
    if lane == "code":
        warnings.append("code_lane_refuse_concat_chop_fallback_pad_only")
        pb = pad_only(documents, tok, sequence_length, lane=lane)
        pb.warnings = warnings
        pb.packing_policy = "pad_only"  # effective policy
        return pb

    eos = effective_eos_id(tok)
    pad = cfg.PAD_TOKEN_ID
    stream: list[int] = []
    stream_docs: list[str] = []
    stream_bounds_meta: list[tuple[str, int, int]] = []  # doc, start, end in stream
    for d in documents:
        ids = _doc_token_ids(d, tok, sequence_length * 4, add_eos=True)
        start = len(stream)
        stream.extend(ids)
        stream_bounds_meta.append((d["doc_id"], start, len(stream)))
        stream_docs.append(d["doc_id"])

    rows, masks, pos_rows, bounds_all, row_docs, flat_docs = [], [], [], [], [], []
    useful = total = 0
    for offset in range(0, max(1, len(stream)), sequence_length):
        chunk = stream[offset : offset + sequence_length]
        chunk = _pad_row(chunk, sequence_length, pad)
        lm = [0.0 if t == pad else 1.0 for t in chunk]
        for i, t in enumerate(chunk):
            if t == eos:
                lm[i] = 1.0
        # boundaries within this window
        bounds = []
        dids = []
        for did, s, e in stream_bounds_meta:
            a, b = max(s, offset), min(e, offset + sequence_length)
            if a < b:
                bounds.append((a - offset, b - offset))
                dids.append(did)
        rows.append(chunk)
        masks.append(lm)
        pos_rows.append(_linear_positions(sequence_length))
        bounds_all.append(bounds or [(0, sum(1 for t in chunk if t != pad))])
        row_docs.append(list(dids))
        flat_docs.extend(dids)
        useful += sum(1 for t in chunk if t != pad)
        total += sequence_length

    return PackedBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        loss_mask=torch.tensor(masks, dtype=torch.float),
        attention_mask=torch.ones(len(rows), sequence_length, dtype=torch.long),
        position_ids=torch.tensor(pos_rows, dtype=torch.long),
        doc_boundaries=bounds_all,
        document_ids=list(dict.fromkeys(flat_docs or stream_docs)),
        packing_policy="concatenate_and_chop",
        utilization=useful / max(1, total),
        lane=lane,
        warnings=warnings,
        row_document_ids=row_docs,
    )


def _first_fit(
    doc_token_lists: list[tuple[str, list[int], str]],
    sequence_length: int,
    eos: int,
    pad: int,
    best_fit: bool,
) -> tuple[list[list[int]], list[list[tuple[int, int]]], list[list[str]], list[str]]:
    # open bins: list of (remaining, tokens, bounds, doc_ids)
    items = list(doc_token_lists)
    if best_fit:
        items.sort(key=lambda x: len(x[1]), reverse=True)

    bins: list[dict[str, Any]] = []
    warnings: list[str] = []

    for did, ids, _text in items:
        need = len(ids)
        if need > sequence_length:
            ids = ids[: sequence_length - 1] + [eos]
            need = len(ids)
            warnings.append(f"truncated:{did}")

        placed = False
        candidates = []
        for bi, b in enumerate(bins):
            if b["remaining"] >= need:
                candidates.append((b["remaining"] - need, bi))
        if candidates:
            if best_fit:
                candidates.sort()  # tightest fit
            _, bi = candidates[0]
            b = bins[bi]
            start = len(b["tokens"])
            b["tokens"].extend(ids)
            b["bounds"].append((start, start + need))
            b["doc_ids"].append(did)
            b["remaining"] -= need
            placed = True
        if not placed:
            bins.append(
                {
                    "tokens": list(ids),
                    "bounds": [(0, need)],
                    "doc_ids": [did],
                    "remaining": sequence_length - need,
                }
            )

    rows, bounds_all, doc_rows = [], [], []
    for b in bins:
        rows.append(_pad_row(b["tokens"], sequence_length, pad))
        bounds_all.append(b["bounds"])
        doc_rows.append(b["doc_ids"])
    return rows, bounds_all, doc_rows, warnings


def greedy_packing(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str = "web",
) -> PackedBatch:
    eos = effective_eos_id(tok)
    pad = cfg.PAD_TOKEN_ID
    items = []
    texts = {}
    for d in documents:
        ids = _doc_token_ids(d, tok, sequence_length, add_eos=True)
        items.append((d["doc_id"], ids, d.get("text", "")))
        texts[d["doc_id"]] = d.get("text", "")
    rows, bounds_all, doc_rows, warnings = _first_fit(items, sequence_length, eos, pad, best_fit=False)
    return _finalize_pack(rows, bounds_all, doc_rows, texts, lane, sequence_length, eos, pad, "greedy_packing", warnings)


def best_fit_packing(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str = "web",
) -> PackedBatch:
    eos = effective_eos_id(tok)
    pad = cfg.PAD_TOKEN_ID
    items = []
    texts = {}
    for d in documents:
        ids = _doc_token_ids(d, tok, sequence_length, add_eos=True)
        items.append((d["doc_id"], ids, d.get("text", "")))
        texts[d["doc_id"]] = d.get("text", "")
    rows, bounds_all, doc_rows, warnings = _first_fit(items, sequence_length, eos, pad, best_fit=True)
    return _finalize_pack(rows, bounds_all, doc_rows, texts, lane, sequence_length, eos, pad, "best_fit_packing", warnings)


def structure_preserving(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str = "agentic",
) -> PackedBatch:
    """Pad-only geometry + reset position IDs + block-diagonal attention metadata."""
    eos = effective_eos_id(tok)
    pad = cfg.PAD_TOKEN_ID
    # Pack multiple short docs into sequences without cross-doc attention
    items = []
    texts = {}
    for d in documents:
        ids = _doc_token_ids(d, tok, sequence_length, add_eos=True)
        items.append((d["doc_id"], ids, d.get("text", "")))
        texts[d["doc_id"]] = d.get("text", "")
    rows, bounds_all, doc_rows, warnings = _first_fit(items, sequence_length, eos, pad, best_fit=False)

    masks, pos_rows = [], []
    attn_flags = []
    useful = total = 0
    for row, bounds, dids in zip(rows, bounds_all, doc_rows):
        # loss mask: agentic per doc span
        lm = [0.0] * sequence_length
        for (start, end), did in zip(bounds, dids):
            span_ids = row[start:end]
            span_lm = _loss_for_lane(lane, texts[did], span_ids, pad, eos)
            for i, v in enumerate(span_lm):
                if start + i < sequence_length:
                    lm[start + i] = v
        for i, t in enumerate(row):
            if t == pad:
                lm[i] = 0.0
            if t == eos:
                lm[i] = 1.0
        masks.append(lm)
        pos_rows.append(_reset_positions(bounds, sequence_length))
        attn_flags.append(1)  # marker: block-diagonal required
        useful += sum(1 for t in row if t != pad)
        total += sequence_length

    pb = PackedBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        loss_mask=torch.tensor(masks, dtype=torch.float),
        attention_mask=torch.ones(len(rows), sequence_length, dtype=torch.long),
        position_ids=torch.tensor(pos_rows, dtype=torch.long),
        doc_boundaries=bounds_all,
        document_ids=[d for row in doc_rows for d in row],
        packing_policy="structure_preserving",
        utilization=useful / max(1, total),
        lane=lane,
        warnings=warnings,
        row_document_ids=[list(dids) for dids in doc_rows],
    )
    # stash first sequence block-diagonal mask for tests
    if bounds_all:
        pb.attention_mask = _attention_block_diag(bounds_all[0], sequence_length).unsqueeze(0)
        if len(rows) > 1:
            mats = [_attention_block_diag(b, sequence_length) for b in bounds_all]
            pb.attention_mask = torch.stack(mats, dim=0)
    return pb


def _finalize_pack(rows, bounds_all, doc_rows, texts, lane, sequence_length, eos, pad, policy, warnings):
    masks, pos_rows = [], []
    useful = total = 0
    flat_docs = []
    for row, bounds, dids in zip(rows, bounds_all, doc_rows):
        lm = [0.0] * sequence_length
        for (start, end), did in zip(bounds, dids):
            span_ids = row[start:end]
            span_lm = _loss_for_lane(lane, texts.get(did, ""), span_ids, pad, eos)
            for i, v in enumerate(span_lm):
                if start + i < sequence_length:
                    lm[start + i] = v
        for i, t in enumerate(row):
            if t == pad:
                lm[i] = 0.0
            if t == eos:
                lm[i] = 1.0
        masks.append(lm)
        pos_rows.append(_linear_positions(sequence_length))
        flat_docs.extend(dids)
        useful += sum(1 for t in row if t != pad)
        total += sequence_length
    return PackedBatch(
        input_ids=torch.tensor(rows, dtype=torch.long),
        loss_mask=torch.tensor(masks, dtype=torch.float),
        attention_mask=torch.ones(len(rows), sequence_length, dtype=torch.long),
        position_ids=torch.tensor(pos_rows, dtype=torch.long),
        doc_boundaries=bounds_all,
        document_ids=list(dict.fromkeys(flat_docs)),
        packing_policy=policy,
        utilization=useful / max(1, total),
        lane=lane,
        warnings=warnings,
        row_document_ids=[list(dids) for dids in doc_rows],
    )


POLICY_FN = {
    "pad_only": pad_only,
    "concatenate_and_chop": concatenate_and_chop,
    "greedy_packing": greedy_packing,
    "best_fit_packing": best_fit_packing,
    "structure_preserving": structure_preserving,
}


def policy_for_lane(lane: str) -> str:
    if lane == "agentic":
        return "structure_preserving"
    if lane == "code":
        return "greedy_packing"
    if lane == "long_context":
        return "concatenate_and_chop"
    if lane in ("web", "stem", "indic"):
        return "best_fit_packing"
    if lane == "reasoning":
        return "structure_preserving"
    return "greedy_packing"


def pack_documents(
    documents: list[dict[str, Any]],
    tok: Any,
    sequence_length: int,
    lane: str,
    policy: str | None = None,
) -> PackedBatch:
    name = policy or policy_for_lane(lane)
    fn = POLICY_FN[name]
    return fn(documents, tok, sequence_length, lane=lane)
