"""Append-only consumption + learning ledgers (JSONL)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import config as cfg


def append_jsonl(path: Path, record: dict[str, Any]) -> int:
    """Append one JSON record; return byte offset before write (ledger_offset)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = path.stat().st_size if path.exists() else 0
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    return offset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass
class ConsumptionLedger:
    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    _offset: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.records = read_jsonl(self.path)
            self._offset = self.path.stat().st_size

    @property
    def offset(self) -> int:
        return self._offset

    def append(self, event: dict[str, Any]) -> int:
        event = {
            "schema_version": cfg.SCHEMA_VERSION,
            **event,
        }
        off = append_jsonl(self.path, event)
        self.records.append(event)
        self._offset = self.path.stat().st_size
        return off

    def get_by_step(self, global_step: int) -> dict[str, Any] | None:
        for r in self.records:
            if r.get("global_step") == global_step:
                return r
        return None

    def last(self) -> dict[str, Any] | None:
        return self.records[-1] if self.records else None


@dataclass
class LearningLedger:
    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.records = read_jsonl(self.path)

    def append(self, event: dict[str, Any]) -> int:
        event = {"schema_version": cfg.SCHEMA_VERSION, **event}
        off = append_jsonl(self.path, event)
        self.records.append(event)
        return off

    def classify(self, loss_delta: float, grad_norm: float) -> str:
        if loss_delta < -0.01 and grad_norm < 50:
            return "useful"
        if loss_delta > 0.05 or grad_norm > 200:
            return "harmful"
        return "neutral"
