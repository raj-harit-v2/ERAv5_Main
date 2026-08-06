"""Resume / Replay / Fork with hash verification."""
from __future__ import annotations

import uuid
from typing import Any, Iterator

import torch

from src.checkpoint_manager import CheckpointManager
from src.ledger import ConsumptionLedger, append_jsonl
from src.utils import batch_content_hash


class ReplayIntegrityError(RuntimeError):
    pass


class SimulatedCrash(Exception):
    def __init__(self, step: int, expected_next_batch_id: str):
        super().__init__(f"SimulatedCrash at step={step}")
        self.step = step
        self.expected_next_batch_id = expected_next_batch_id


class ReplayEngine:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        consumption_ledger: ConsumptionLedger,
        fork_log_path: Any = None,
    ):
        self.ckpts = checkpoint_manager
        self.ledger = consumption_ledger
        self.fork_log_path = fork_log_path
        self.branch_id: str | None = None
        self.run_id: str | None = None

    def resume(
        self,
        checkpoint_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        expected_batch_id: str | None = None,
    ) -> dict[str, Any]:
        meta = self.ckpts.load(checkpoint_id, model, optimizer, scheduler)
        self.run_id = meta["run_id"]
        self.branch_id = meta["branch_id"]
        result = {
            "ledger_offset": meta["ledger_offset"],
            "step": meta["step"],
            "expected_next_batch_id": meta.get("expected_next_batch_id"),
            "checkpoint_id": checkpoint_id,
            "branch_id": meta["branch_id"],
            "run_id": meta["run_id"],
            "dataloader_state": meta.get("dataloader_state", {}),
        }
        if expected_batch_id is not None:
            got = meta.get("expected_next_batch_id")
            if got != expected_batch_id:
                raise ReplayIntegrityError(
                    f"resume_next_batch_mismatch expected={expected_batch_id} got={got}"
                )
        return result

    def replay(
        self,
        original_records: list[dict[str, Any]],
        rebuilt_batches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Compare rebuilt batches to original ledger records.
        Each rebuilt batch: {batch_id, token_span_ids, token_ids or batch_hash}
        """
        reports = []
        for orig, reb in zip(original_records, rebuilt_batches):
            oh = orig.get("batch_hash")
            if "batch_hash" in reb:
                rh = reb["batch_hash"]
            else:
                rh = batch_content_hash(reb["token_ids"])
            ok = (
                oh == rh
                and orig.get("batch_id") == reb.get("batch_id")
                and orig.get("token_span_ids") == reb.get("token_span_ids")
            )
            reports.append(
                {
                    "step": orig.get("global_step"),
                    "original_hash": oh,
                    "replay_hash": rh,
                    "batch_id_match": orig.get("batch_id") == reb.get("batch_id"),
                    "span_match": orig.get("token_span_ids") == reb.get("token_span_ids"),
                    "ok": ok,
                }
            )
            if not ok:
                raise ReplayIntegrityError(
                    f"replay_hash_mismatch step={orig.get('global_step')} "
                    f"original={oh} replay={rh}"
                )
        return reports

    def fork(
        self,
        checkpoint_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        meta = self.ckpts.load(checkpoint_id, model, optimizer, scheduler)
        new_branch = str(uuid.uuid4())
        fork_point = {
            "original_run_id": meta["run_id"],
            "original_checkpoint_id": checkpoint_id,
            "fork_step": meta["step"],
            "new_branch_id": new_branch,
            "parent_branch_id": meta["branch_id"],
            "ledger_offset": meta["ledger_offset"],
        }
        self.branch_id = new_branch
        self.run_id = meta["run_id"]
        if self.fork_log_path is not None:
            append_jsonl(self.fork_log_path, {"event": "fork", **fork_point})
        self.ledger.append(
            {
                "event": "fork_point",
                "global_step": meta["step"],
                "run_id": meta["run_id"],
                "branch_id": new_branch,
                **fork_point,
            }
        )
        return {"event": "fork", **fork_point}
