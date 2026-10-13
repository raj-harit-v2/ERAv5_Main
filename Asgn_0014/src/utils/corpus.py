"""Structured corpus for Assgn_014 — packed next-token batches (not randint).

Default: TinyStories (HF) + BPE vocab=8192, with offline sample fallback.
Env: ASSGN014_DATA=corpus|synthetic (default corpus).
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch import Tensor

VOCAB_SIZE = 8192
CE_FLOOR_LN_V = math.log(VOCAB_SIZE)
DEFAULT_DATA_MODE = "corpus"

# Bundled fallback text path relative to project root.
SAMPLE_REL = Path("data") / "corpus_sample.txt"
TOKENIZER_REL = Path("configs") / "assgn014_bpe_8192.json"
CACHE_REL = Path("data") / "tinystories"


def resolve_data_mode(env: Optional[dict] = None) -> str:
    raw = ""
    if env is not None:
        raw = str(env.get("ASSGN014_DATA", "") or "").strip()
    if not raw:
        raw = os.environ.get("ASSGN014_DATA", "").strip()
    if not raw:
        raw = DEFAULT_DATA_MODE
    mode = raw.lower()
    if mode not in ("corpus", "synthetic"):
        raise ValueError(f"ASSGN014_DATA must be corpus|synthetic, got {raw!r}")
    return mode


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_sample_corpus(root: Path) -> Path:
    """Write a minimal structured English sample if missing (offline fallback)."""
    path = root / SAMPLE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 2000:
        return path
    stories = []
    for i in range(40):
        stories.append(
            f"Once upon a time a small cat named Mia lived in town number {i}. "
            f"Mia liked warm milk and soft blankets. One day Mia met a dog named Bo. "
            f"Bo said hello and they became friends. They played in the garden until sunset. "
            f"Then Mia went home and slept. The end of story {i}.\n"
        )
    path.write_text("\n".join(stories), encoding="utf-8")
    return path


def _load_tinystories_texts(root: Path, max_docs: int = 5000) -> List[str]:
    """Try HuggingFace TinyStories; return [] on failure."""
    cache = root / CACHE_REL
    cache.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            cache_dir=str(cache / "hf"),
        )
    except Exception:
        return []
    texts: List[str] = []
    n = min(max_docs, len(ds))
    for i in range(n):
        t = ds[i].get("text") or ""
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
    return texts


def _corpus_texts(root: Path) -> tuple[List[str], str]:
    texts = _load_tinystories_texts(root)
    if texts:
        return texts, "tinystories_hf"
    sample = ensure_sample_corpus(root)
    blob = sample.read_text(encoding="utf-8")
    # Split into pseudo-docs on blank lines / story markers
    parts = [p.strip() for p in blob.split("\n\n") if p.strip()]
    if len(parts) < 2:
        parts = [blob]
    return parts, f"sample:{SAMPLE_REL.as_posix()}"


def train_or_load_tokenizer(root: Path, texts: List[str]):
    """BPE vocab=8192 via HuggingFace tokenizers; persist under configs/."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tok_path = root / TOKENIZER_REL
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    if tok_path.is_file():
        return Tokenizer.from_file(str(tok_path))

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    # tokenizers trainer wants an iterator of strings
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(tok_path))
    return tokenizer


def encode_corpus(tokenizer, texts: List[str]) -> List[int]:
    ids: List[int] = []
    eos_id = 3  # [EOS] from special_tokens order
    for t in texts:
        enc = tokenizer.encode(t)
        ids.extend(enc.ids)
        ids.append(eos_id)
    if len(ids) < VOCAB_SIZE:
        # pad cycling so short samples still stream
        base = list(ids) if ids else [1]
        while len(ids) < 50_000:
            ids.extend(base)
    return ids


class PackedCorpusSource:
    """Cycles a packed 1D token stream into (B, S+1) batches on device."""

    def __init__(
        self,
        token_ids: List[int],
        *,
        vocab_size: int = VOCAB_SIZE,
        data_source: str = "corpus",
        seed: int = 13,
    ) -> None:
        if len(token_ids) < 64:
            raise ValueError("token stream too short")
        self.ids = token_ids
        self.vocab_size = vocab_size
        self.data_source = data_source
        self.pos = 0
        self.rng = random.Random(seed)
        # start at random offset for diversity across runs
        self.pos = self.rng.randrange(0, max(1, len(self.ids) - 1))

    def next_batch(self, batch: int, seq_len: int, device: torch.device) -> Tensor:
        need = batch * (seq_len + 1)
        out: List[int] = []
        n = len(self.ids)
        while len(out) < need:
            take = min(need - len(out), n - self.pos)
            out.extend(self.ids[self.pos : self.pos + take])
            self.pos += take
            if self.pos >= n:
                self.pos = 0
        # clamp to vocab (safety)
        t = torch.tensor(out[:need], dtype=torch.long, device=device)
        t = t.clamp(0, self.vocab_size - 1)
        return t.view(batch, seq_len + 1)


def build_batch_source(
    root: Optional[Path] = None,
    *,
    mode: Optional[str] = None,
    env: Optional[dict] = None,
    seed: int = 13,
) -> tuple[object, dict]:
    """Return (source, meta). source has next_batch(batch, seq_len, device)."""
    root = root or _project_root()
    mode = mode or resolve_data_mode(env)
    meta = {
        "data_mode": mode,
        "vocab_size": VOCAB_SIZE,
        "ce_floor_ln_v": CE_FLOOR_LN_V,
        "tokenizer": "none",
        "data_source": mode,
    }
    if mode == "synthetic":

        class _Synth:
            data_source = "synthetic"

            def next_batch(self, batch: int, seq_len: int, device: torch.device) -> Tensor:
                return torch.randint(0, VOCAB_SIZE, (batch, seq_len + 1), device=device)

        return _Synth(), meta

    cache_ids = root / CACHE_REL / "packed_ids.pt"
    tok_path = root / TOKENIZER_REL
    if cache_ids.is_file() and tok_path.is_file():
        ids = torch.load(cache_ids, map_location="cpu", weights_only=False)
        if not isinstance(ids, list):
            ids = list(ids)
        meta.update(
            {
                "data_source": "cache:data/tinystories/packed_ids.pt",
                "tokenizer": TOKENIZER_REL.as_posix(),
                "n_stream_tokens": len(ids),
            }
        )
        return PackedCorpusSource(ids, data_source=str(meta["data_source"]), seed=seed), meta

    texts, src_name = _corpus_texts(root)
    tokenizer = train_or_load_tokenizer(root, texts)
    ids = encode_corpus(tokenizer, texts)
    cache_ids.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_ids)
    meta.update(
        {
            "data_source": src_name,
            "tokenizer": TOKENIZER_REL.as_posix(),
            "n_stream_tokens": len(ids),
            "n_docs": len(texts),
        }
    )
    return PackedCorpusSource(ids, data_source=src_name, seed=seed), meta


def iter_packed_batches(
    source: object,
    *,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Tensor]:
    while True:
        yield source.next_batch(batch, seq_len, device)  # type: ignore[attr-defined]
