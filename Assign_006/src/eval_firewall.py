"""Evaluation / validation firewall — content-hash + canary registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.corpus import CANARY_EVAL, CANARY_TEST


@dataclass
class EvalFirewall:
    """Registry of never-train content hashes, canaries, and access log."""

    content_hashes: set[str] = field(default_factory=set)
    doc_ids: set[str] = field(default_factory=set)
    canaries: set[str] = field(default_factory=lambda: {CANARY_EVAL, CANARY_TEST})
    access_log: list[dict[str, Any]] = field(default_factory=list)
    blocked_events: list[dict[str, Any]] = field(default_factory=list)

    def register_document(self, doc: dict[str, Any]) -> None:
        text = doc.get("text", "")
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.content_hashes.add(h)
        self.doc_ids.add(doc["doc_id"])
        self.access_log.append(
            {
                "action": "register",
                "doc_id": doc["doc_id"],
                "content_hash": h,
                "lane": doc.get("lane"),
                "never_train": True,
            }
        )

    def register_corpus_holdouts(self, corpus: dict[str, list[dict[str, Any]]]) -> None:
        for lane in ("eval", "test"):
            for doc in corpus.get(lane, []):
                self.register_document(doc)

    def check_text(self, text: str, doc_id: str | None = None) -> tuple[bool, str]:
        """Return (allowed, reason). allowed=False means blocked."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self.content_hashes:
            return False, "content_hash_match"
        if doc_id and doc_id in self.doc_ids:
            return False, "doc_id_in_eval_registry"
        for canary in self.canaries:
            if canary in text:
                return False, "canary_detected"
        return True, ""

    def check_document(self, doc: dict[str, Any]) -> tuple[bool, str]:
        return self.check_text(doc.get("text", ""), doc.get("doc_id"))

    def check_shard_manifest(self, manifest: dict[str, Any]) -> tuple[bool, str]:
        for did in manifest.get("document_ids", []):
            if did in self.doc_ids:
                return False, f"doc_id_in_eval_registry:{did}"
        if manifest.get("eval_test_overlap_status") != "clean":
            return False, "manifest_eval_overlap_status"
        if manifest.get("capability_lane") in ("eval", "test"):
            return False, "eval_lane_shard"
        return True, ""

    def block(self, *, shard_id: str, reason: str, detail: str = "") -> dict[str, Any]:
        event = {
            "event": "eval_shard_blocked",
            "shard_id": shard_id,
            "reason": reason,
            "detail": detail,
        }
        self.blocked_events.append(event)
        return event

    def assert_not_in_training_batch(
        self, document_ids: list[str], loss_mask_sum: float
    ) -> None:
        bad = [d for d in document_ids if d in self.doc_ids]
        if bad and loss_mask_sum > 0:
            raise RuntimeError(f"eval_in_loss_bearing_batch:{bad}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "content_hashes": sorted(self.content_hashes),
                    "doc_ids": sorted(self.doc_ids),
                    "canaries": sorted(self.canaries),
                    "blocked_events": self.blocked_events,
                    "access_log": self.access_log,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
