"""Checkpoint save/load: model + optimizer + scheduler + RNG + ledger_offset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import config as cfg
from src.utils import capture_rng_state, restore_rng_state, sha256_text, write_json


class CheckpointManager:
    def __init__(self, ckpt_dir: Path | None = None):
        self.ckpt_dir = ckpt_dir or cfg.CKPT_DIR
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ledger_offset: int,
        loss: float,
        run_id: str,
        branch_id: str,
        dataloader_state: dict[str, Any] | None = None,
        expected_next_batch_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        checkpoint_id = f"checkpoint_{step:06d}"
        dest = self.ckpt_dir / checkpoint_id
        tmp = self.ckpt_dir / f".tmp_{checkpoint_id}"
        if tmp.exists():
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()
        tmp.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
        else:
            torch.save({}, tmp / "scheduler.pt")
        torch.save(capture_rng_state(), tmp / "rng_state.pt")

        metadata = {
            "schema_version": cfg.SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "step": step,
            "ledger_offset": ledger_offset,
            "loss": loss,
            "run_id": run_id,
            "branch_id": branch_id,
            "dataloader_state": dataloader_state or {},
            "expected_next_batch_id": expected_next_batch_id,
            "code_hash": sha256_text(cfg.SCHEMA_VERSION + str(cfg.SEED)),
            **(extra or {}),
        }
        write_json(tmp / "metadata.json", metadata)

        # atomic replace
        if dest.exists():
            for p in dest.iterdir():
                p.unlink()
            dest.rmdir()
        tmp.rename(dest)
        return checkpoint_id

    def load(
        self,
        checkpoint_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        dest = self.ckpt_dir / checkpoint_id
        model.load_state_dict(torch.load(dest / "model.pt", weights_only=True))
        optimizer.load_state_dict(torch.load(dest / "optimizer.pt", weights_only=True))
        sched_state = torch.load(dest / "scheduler.pt", weights_only=True)
        if scheduler is not None and sched_state:
            scheduler.load_state_dict(sched_state)
        rng_state = torch.load(dest / "rng_state.pt", weights_only=False)
        restore_rng_state(rng_state)
        metadata = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
        return metadata

    def list_checkpoints(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.ckpt_dir.glob("checkpoint_*")):
            meta = p / "metadata.json"
            if meta.exists():
                out.append(json.loads(meta.read_text(encoding="utf-8")))
        return out

    def latest(self) -> str | None:
        ckpts = self.list_checkpoints()
        if not ckpts:
            return None
        return max(ckpts, key=lambda m: m["step"])["checkpoint_id"]
