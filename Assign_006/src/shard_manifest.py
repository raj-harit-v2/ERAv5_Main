"""14-field manifest schema, validation, SHA-256 admission gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config as cfg
from src.utils import sha256_file

MANIFEST_FIELDS = (
    "shard_id",
    "source_ids",
    "document_ids",
    "tokenizer_hash",
    "token_count",
    "language_script",
    "capability_lane",
    "license_provenance_tier",
    "cleaning_pipeline_hash",
    "deduplication_status",
    "contamination_status",
    "eval_test_overlap_status",
    "content_hash",
    "parent_shard_ids",
)

ADMITTED_LICENSES = cfg.ADMITTED_LICENSES


def validate_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"missing_field:{field}")
        elif manifest[field] is None:
            errors.append(f"null_field:{field}")
    return (len(errors) == 0, errors)


def check_admission(manifest: dict[str, Any], shard_file: Path) -> tuple[bool, str]:
    """Run 7 admission checks. Returns (admitted, reason)."""
    ok, errs = validate_manifest(manifest)
    if not ok:
        return False, ";".join(errs)

    if not manifest.get("tokenizer_hash"):
        return False, "missing_tokenizer_hash"
    if not manifest.get("cleaning_pipeline_hash"):
        return False, "missing_cleaning_pipeline_hash"
    if manifest.get("deduplication_status") in (None, "", "unknown"):
        return False, "unverified_deduplication"
    if manifest.get("contamination_status") == "contaminated":
        return False, "contaminated"
    if manifest.get("eval_test_overlap_status") != "clean":
        return False, "eval_test_overlap"
    license_tier = str(manifest.get("license_provenance_tier", "")).split(":")[0]
    if license_tier not in ADMITTED_LICENSES:
        return False, f"unsafe_license:{license_tier}"
    if not shard_file.exists():
        return False, "shard_file_missing"
    actual = sha256_file(shard_file)
    if actual != manifest.get("content_hash"):
        return False, "content_hash_mismatch"
    return True, ""
