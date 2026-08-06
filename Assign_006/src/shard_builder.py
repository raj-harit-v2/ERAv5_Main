"""Document → immutable tokenized shard (.npy) + manifest JSON."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

import config as cfg
from src.shard_manifest import check_admission
from src.tokenizer_wrapper import encode_document
from src.utils import sha256_file, write_json


def compute_cleaning_hash() -> str:
    path = Path(__file__).resolve().parent / "corpus.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_shard(
    documents: list[dict[str, Any]],
    tokenizer_hash: str,
    lane: str,
    sequence_length: int,
    shard_index: int,
    output_dir: Path,
    cleaning_pipeline_hash: str,
    capability_lane: str,
    eval_registry: set[str],
    tok: Any,
) -> dict[str, Any]:
    """Build one shard. Raises ValueError if eval docs present."""
    doc_ids = [d["doc_id"] for d in documents]
    overlap = set(doc_ids) & eval_registry
    if overlap:
        raise ValueError(f"eval_overlap:{sorted(overlap)}")

    for d in documents:
        lic = d.get("license", "restricted")
        if lic not in cfg.ADMITTED_LICENSES:
            raise ValueError(f"unsafe_license:{lic}")

    rows: list[list[int]] = []
    for d in documents:
        ids = encode_document(d["text"], tok, sequence_length, add_eos=True)
        # pad to sequence_length for fixed shape
        if len(ids) < sequence_length:
            ids = ids + [cfg.PAD_TOKEN_ID] * (sequence_length - len(ids))
        rows.append(ids[:sequence_length])

    arr = np.asarray(rows, dtype=np.int32)

    shards_dir = output_dir / "shards"
    manifests_dir = output_dir / "manifests"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Write to a temp name, hash file bytes (includes npy header), then rename with hash id
    tmp_path = shards_dir / f"_tmp_{lane}_{shard_index:04d}.npy"
    np.save(tmp_path, arr)
    content_hash = sha256_file(tmp_path)
    shard_id = f"shard_{lane}_{shard_index:04d}_{content_hash[:8]}"
    shard_path = shards_dir / f"{shard_id}.npy"
    if shard_path.exists():
        shard_path.unlink()
    tmp_path.rename(shard_path)

    langs = sorted({d.get("lang", "en") for d in documents})
    licenses = sorted({d.get("license", "cc0") for d in documents})
    license_tier = f"{licenses[0]}:tier_{documents[0].get('tier', 'b')}"

    manifest: dict[str, Any] = {
        "schema_version": cfg.SCHEMA_VERSION,
        "shard_id": shard_id,
        "source_ids": ["synthetic_a06"],
        "document_ids": doc_ids,
        "tokenizer_hash": tokenizer_hash,
        "token_count": int(arr.size),
        "language_script": ",".join(langs),
        "capability_lane": capability_lane,
        "license_provenance_tier": license_tier,
        "cleaning_pipeline_hash": cleaning_pipeline_hash,
        "deduplication_status": "smoke_exact_dedup_applied",
        "contamination_status": "clean",
        "eval_test_overlap_status": "clean",
        "content_hash": content_hash,
        "parent_shard_ids": [],
        "shard_path": str(shard_path.relative_to(output_dir)),
        "sequence_length": sequence_length,
        "n_sequences": int(arr.shape[0]),
    }

    admitted, reason = check_admission(manifest, shard_path)
    if not admitted:
        shard_path.unlink(missing_ok=True)
        raise ValueError(f"admission_failed:{reason}")

    write_json(manifests_dir / f"{shard_id}.json", manifest)
    return manifest


def build_lane_shards(
    corpus: dict[str, list[dict[str, Any]]],
    tok: Any,
    tokenizer_hash: str,
    output_dir: Path,
    docs_per_shard: int = 4,
    sequence_length: int = cfg.SEQUENCE_LENGTH,
    eval_registry: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build train shards for all capability lanes. Skips eval/test lanes."""
    eval_registry = eval_registry or set()
    cleaning = compute_cleaning_hash()
    manifests: list[dict[str, Any]] = []
    for lane in cfg.CAPABILITY_LANES:
        docs = corpus.get(lane, [])
        for i in range(0, len(docs), docs_per_shard):
            chunk = docs[i : i + docs_per_shard]
            if not chunk:
                continue
            m = build_shard(
                chunk,
                tokenizer_hash=tokenizer_hash,
                lane=lane,
                sequence_length=sequence_length,
                shard_index=i // docs_per_shard,
                output_dir=output_dir,
                cleaning_pipeline_hash=cleaning,
                capability_lane=lane,
                eval_registry=eval_registry,
                tok=tok,
            )
            manifests.append(m)
    return manifests
