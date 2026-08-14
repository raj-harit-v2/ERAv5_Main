"""Collision / truncation audits for Kronecker and Fourier codecs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.embeddings import KroneckerByteEmbedding
from src.embeddings.fourier_baseline import FourierCodepointEmbedding
from src.utils import write_json


def _script_of_token(tok: str) -> str:
    for ch in tok:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
    return "en"


def kronecker_code_key(emb: KroneckerByteEmbedding, text: str) -> bytes:
    return emb.encode_string(text).detach().cpu().numpy().tobytes()


def fourier_code_key(emb: FourierCodepointEmbedding, text: str) -> bytes:
    return emb.encode_string(text).detach().cpu().numpy().tobytes()


def collision_report(
    tokens: list[str],
    *,
    pos_dims: tuple[int, ...] = (32, 48, 64),
    fourier_code_dim: int = 512,
    fourier_n_freq: int = 16,
    max_chars: int = 64,
) -> dict[str, Any]:
    tokens = [t for t in tokens if t and t not in ("<pad>", "<unk>")]
    by_script: dict[str, list[str]] = defaultdict(list)
    for t in tokens:
        by_script[_script_of_token(t)].append(t)

    kron: dict[str, Any] = {}
    for pd in pos_dims:
        emb = KroneckerByteEmbedding(d_model=8, pos_dim=pd)
        buckets: dict[bytes, list[str]] = defaultdict(list)
        trunc = defaultdict(int)
        total = defaultdict(int)
        for t in tokens:
            sc = _script_of_token(t)
            total[sc] += 1
            raw = t.encode("utf-8")
            if len(raw) > pd:
                trunc[sc] += 1
            buckets[kronecker_code_key(emb, t)].append(t)
        collisions = {k: v for k, v in buckets.items() if len(v) > 1}
        kron[str(pd)] = {
            "n_tokens": len(tokens),
            "n_unique_codes": len(buckets),
            "n_collision_groups": len(collisions),
            "n_colliding_tokens": sum(len(v) for v in collisions.values()),
            "truncation_by_script": dict(trunc),
            "tokens_by_script": dict(total),
            "examples": [[a, b] for group in list(collisions.values())[:5] for a, b in [group[:2]] if len(group) >= 2],
        }

    f_emb = FourierCodepointEmbedding(
        d_model=8, code_dim=fourier_code_dim, n_freq=fourier_n_freq, max_chars=max_chars
    )
    f_buckets: dict[bytes, list[str]] = defaultdict(list)
    f_trunc = defaultdict(int)
    f_total = defaultdict(int)
    for t in tokens:
        sc = _script_of_token(t)
        f_total[sc] += 1
        if len(list(t)) > max_chars:
            f_trunc[sc] += 1
        f_buckets[fourier_code_key(f_emb, t)].append(t)
    f_coll = {k: v for k, v in f_buckets.items() if len(v) > 1}
    fourier = {
        "n_tokens": len(tokens),
        "n_unique_codes": len(f_buckets),
        "n_collision_groups": len(f_coll),
        "n_colliding_tokens": sum(len(v) for v in f_coll.values()),
        "truncation_by_script": dict(f_trunc),
        "tokens_by_script": dict(f_total),
        "by_script_unique_ratio": {
            sc: (len({fourier_code_key(f_emb, t) for t in toks}) / max(len(toks), 1))
            for sc, toks in by_script.items()
        },
    }
    return {"kronecker": kron, "fourier": fourier, "n_vocab_tokens": len(tokens)}


def write_collision_report(report: dict[str, Any], path) -> None:
    write_json(path, report)
