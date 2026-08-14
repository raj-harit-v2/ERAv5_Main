"""Curriculum stages + sampling with Indic floors."""
from __future__ import annotations

import random
from typing import Iterator

import config as cfg
from src.corpus_indic_toy import Doc


def docs_for_stage(docs: list[Doc], stage: str) -> list[Doc]:
    train = [d for d in docs if d.split == "train"]
    # Prefer stage affinity, then backfill
    primary = [d for d in train if d.stage_affinity == stage]
    if len(primary) >= 8:
        pool = primary
    else:
        pool = train
    return pool


def sample_batch_docs(
    docs: list[Doc],
    stage: str,
    batch_size: int,
    rng: random.Random,
) -> list[Doc]:
    pool = docs_for_stage(docs, stage)
    indic = [d for d in pool if d.script in ("hi", "te")]
    other = [d for d in pool if d.script == "en"]
    floor = cfg.STAGE_INDIC_FLOOR.get(stage, 0.2)
    n_indic = max(1, int(round(batch_size * floor))) if indic else 0
    n_indic = min(n_indic, batch_size, len(indic)) if indic else 0
    n_other = batch_size - n_indic
    batch: list[Doc] = []
    if n_indic:
        batch.extend(rng.choices(indic, k=n_indic))
    if n_other and other:
        batch.extend(rng.choices(other, k=n_other))
    while len(batch) < batch_size:
        batch.append(rng.choice(pool))
    return batch[:batch_size]


def iter_stage_steps(
    docs: list[Doc],
    stage: str,
    steps: int,
    batch_size: int,
    seed: int,
) -> Iterator[list[Doc]]:
    rng = random.Random(seed + hash(stage) % 10_000)
    for _ in range(steps):
        yield sample_batch_docs(docs, stage, batch_size, rng)
