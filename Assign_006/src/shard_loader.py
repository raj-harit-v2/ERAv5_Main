"""Load immutable shard .npy rows for training and replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_shard_array(artifacts_dir: Path, manifest: dict[str, Any]) -> np.ndarray:
    rel = manifest.get("shard_path")
    if not rel:
        raise FileNotFoundError("manifest missing shard_path")
    path = artifacts_dir / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def attach_shard_tokens(
    doc: dict[str, Any],
    manifest: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Attach pre-tokenized row from immutable shard for doc_id."""
    out = dict(doc)
    arr = load_shard_array(artifacts_dir, manifest)
    doc_ids = manifest.get("document_ids") or []
    did = out.get("doc_id")
    if did not in doc_ids:
        return out
    row = doc_ids.index(did)
    if row >= arr.shape[0]:
        return out
    out["_input_ids"] = arr[row].astype(np.int64).tolist()
    out["_shard_row"] = row
    out["_shard_ids"] = [manifest["shard_id"]]
    out["_manifest"] = manifest
    return out
