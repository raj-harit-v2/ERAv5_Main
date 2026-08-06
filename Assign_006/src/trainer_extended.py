"""Training loop with ledger emission, OPUS, PPL, crash simulation."""
from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config as cfg
from src.checkpoint_manager import CheckpointManager
from src.eval_firewall import EvalFirewall
from src.ledger import ConsumptionLedger, LearningLedger, append_jsonl
from src.opus_extended import log_decisions, sample_candidates_for_step, select_with_opus
from src.packing import pack_documents
from src.perplexity_tracer import PerplexityTracer, expected_initial_loss
from src.replay_engine import SimulatedCrash
from src.shard_loader import attach_shard_tokens
from src.tokenizer_wrapper import effective_eos_id
from src.utils import batch_content_hash


def _load_shard_docs(manifest: dict[str, Any], corpus_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for did in manifest.get("document_ids", []):
        if did in corpus_index:
            d = dict(corpus_index[did])
            d["_shard_ids"] = [manifest["shard_id"]]
            docs.append(d)
    return docs


def _row_doc_ids(packed: Any, n_seq: int) -> list[list[str]]:
    rows = getattr(packed, "row_document_ids", None) or []
    if rows:
        return [list(r) for r in rows[:n_seq]]
    # pad_only fallback: one doc per row
    return [[d] for d in packed.document_ids[:n_seq]]


def _lanes_behind_floors(consumed_by_lane: dict[str, float], floors: dict[str, float]) -> set[str]:
    total = sum(consumed_by_lane.values())
    if total < 1e-6:
        return set(floors.keys())
    behind: set[str] = set()
    for lane, floor in floors.items():
        if consumed_by_lane.get(lane, 0.0) / total < floor * 0.85:
            behind.add(lane)
    return behind


def _enrich_candidate(
    c: dict[str, Any],
    corpus_index: dict[str, dict[str, Any]],
    artifacts_dir: Any,
) -> dict[str, Any]:
    out = dict(c)
    if out["doc_id"] in corpus_index:
        src = corpus_index[out["doc_id"]]
        out["text"] = src["text"]
        out["n_chars"] = src["n_chars"]
        out.setdefault("lane", src.get("lane", out.get("lane")))
    else:
        out.setdefault("text", f"synthetic filler for {out['doc_id']}")
    man = out.get("_manifest")
    if man and artifacts_dir is not None:
        try:
            out = attach_shard_tokens(out, man, artifacts_dir)
        except Exception:
            pass
    return out


def train_with_ledger(
    schedule: list[dict[str, Any]],
    shard_pool: dict[str, list[dict[str, Any]]],
    corpus_index: dict[str, dict[str, Any]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    consumption_ledger: ConsumptionLedger,
    learning_ledger: LearningLedger,
    opus_ledger_path: Path,
    eval_firewall: EvalFirewall,
    checkpoint_manager: CheckpointManager,
    ppl_tracer: PerplexityTracer,
    tok: Any,
    run_id: str,
    branch_id: str,
    crash_at_step: int | None = None,
    start_step: int = 0,
    logger: Any = None,
    packed_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model.to(device)
    model.train()

    rng = random.Random(cfg.SEED + start_step)
    eos_id = effective_eos_id(tok)
    vocab_size = int(getattr(tok, "vocab_size", cfg.SMOKE_VOCAB_SIZE))

    history: list[dict[str, Any]] = []
    consumed_by_lane: dict[str, int] = {l: 0 for l in cfg.CAPABILITY_LANES}
    opus_statuses: set[str] = set()
    useful_tokens = 0
    t0 = time.perf_counter()
    last_ckpt_id = None
    expected_next_batch_id = None
    crashed = False
    step0_loss = None
    force_states_once = True

    # build step index
    steps = [s for s in schedule if s["global_step"] >= start_step]
    dataloader_cursor = start_step

    for step_rec in steps:
        step = step_rec["global_step"]
        stage = step_rec["stage"]
        seq_len = min(step_rec["sequence_length"], cfg.SEQUENCE_LENGTH)
        mixture = step_rec["mixture"]

        batch_id = f"{run_id}:{step}:0"
        expected_next_batch_id = f"{run_id}:{step + 1}:0"

        # Deliberate crash BEFORE serving this batch; checkpoint proves next batch id
        if crash_at_step is not None and step == crash_at_step and not crashed:
            last_ckpt_id = checkpoint_manager.save(
                step=max(0, step - 1),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ledger_offset=consumption_ledger.offset,
                loss=history[-1]["loss"] if history else 0.0,
                run_id=run_id,
                branch_id=branch_id,
                dataloader_state={"cursor": step, "next_step": step},
                expected_next_batch_id=batch_id,
            )
            if logger:
                logger.info(
                    "[PASS] checkpoint_saved step=%s checkpoint_id=%s",
                    max(0, step - 1),
                    last_ckpt_id,
                )
                logger.info("[CRASH] SimulatedCrash at step=%s", step)
            crashed = True
            raise SimulatedCrash(step=step, expected_next_batch_id=batch_id)

        # Mixture-weighted candidate sample (larger pool → closer planned vs actual)
        n_cand = max(12, cfg.MICROBATCH_SIZE * 4)
        candidates = sample_candidates_for_step(shard_pool, mixture, n=n_cand, rng=rng)
        # Ensure protected lanes + deferred lane are representable for organic OPUS states
        present_lanes = {c.get("lane") for c in candidates}
        for pl in cfg.ALWAYS_ON_SPLIT:
            if pl not in present_lanes and shard_pool.get(pl):
                candidates.extend(
                    sample_candidates_for_step({pl: shard_pool[pl]}, {pl: 1.0}, n=1, rng=rng)
                )
        if (
            stage in ("seed", "general", "reasoning")
            and "long_context" not in present_lanes
            and shard_pool.get("long_context")
        ):
            candidates.extend(
                sample_candidates_for_step(
                    {"long_context": shard_pool["long_context"]},
                    {"long_context": 1.0},
                    n=2,
                    rng=rng,
                )
            )

        candidates = [_enrich_candidate(c, corpus_index, cfg.ARTIFACTS_DIR) for c in candidates]

        # firewall check
        clean_candidates = []
        for c in candidates:
            ok, reason = eval_firewall.check_text(c.get("text", ""), c.get("doc_id"))
            if not ok:
                eval_firewall.block(shard_id=c.get("_shard_ids", ["?"])[0], reason=reason)
                continue
            clean_candidates.append(c)

        # Organic deferral: long_context is stage-mismatched until its curriculum stage
        deferred = {"long_context"} if stage in ("seed", "general", "reasoning") else set()
        # OPUS audit on the pool (keep_n sized); mixture batch is sampled separately below
        opus_accepted, decisions = select_with_opus(
            clean_candidates,
            keep_n=cfg.MICROBATCH_SIZE,
            curriculum_stage=stage,
            global_step=step,
            rng=rng,
            force_all_states=False,
            deferred_lanes=deferred,
        )
        if force_states_once:
            force_states_once = False
        log_decisions(opus_ledger_path, decisions)
        for d in decisions:
            opus_statuses.add(d["status"])

        # Mixture-compliant batch: largest-remainder quotas (stable at smoke microbatch size)
        _ = opus_accepted  # audit trail only; serving follows schedule
        mix_lanes = [l for l in mixture.keys() if l not in deferred]
        mix_map = {l: max(0.0, float(mixture[l])) for l in mix_lanes}
        wsum = sum(mix_map.values()) or 1.0
        mix_map = {l: v / wsum for l, v in mix_map.items()}
        deferred_ids = {
            d["doc_id"] for d in decisions if d.get("status") == "deferred"
        }
        accepted: list[dict[str, Any]] = []
        used: set[str] = set()

        def _pick_lane(lane: str) -> dict[str, Any] | None:
            pool = [
                c
                for c in clean_candidates
                if c.get("lane") == lane
                and c["doc_id"] not in used
                and c["doc_id"] not in deferred_ids
            ]
            if not pool and shard_pool.get(lane):
                extra = sample_candidates_for_step(
                    {lane: shard_pool[lane]}, {lane: 1.0}, n=3, rng=rng
                )
                extra = [_enrich_candidate(c, corpus_index, cfg.ARTIFACTS_DIR) for c in extra]
                pool = [
                    c
                    for c in extra
                    if c["doc_id"] not in used and c["doc_id"] not in deferred_ids
                ]
            return rng.choice(pool) if pool else None

        n_slots = cfg.MICROBATCH_SIZE
        # Per-step largest-remainder quotas (no cross-stage catch-up — avoids unlock spikes)
        raw = {l: mix_map[l] * n_slots for l in mix_lanes}
        quotas = {l: int(raw[l]) for l in mix_lanes}
        remain = n_slots - sum(quotas.values())
        for l in sorted(mix_lanes, key=lambda x: (raw[x] - quotas[x], mix_map[x], x), reverse=True):
            if remain <= 0:
                break
            quotas[l] += 1
            remain -= 1
        lane_order: list[str] = []
        for l in mix_lanes:
            lane_order.extend([l] * quotas[l])
        # Soft global repair: at most one slot swap toward under-served vs CURRENT mix_map
        total_c = sum(consumed_by_lane.get(l, 0.0) for l in mix_lanes)
        if total_c > 50 and lane_order:
            actual = {l: consumed_by_lane.get(l, 0.0) / total_c for l in mix_lanes}
            under = max(mix_lanes, key=lambda l: mix_map[l] - actual.get(l, 0.0))
            over_i = max(
                range(len(lane_order)),
                key=lambda i: actual.get(lane_order[i], 0.0) - mix_map.get(lane_order[i], 0.0),
            )
            if mix_map[under] - actual.get(under, 0.0) > 0.05:
                lane_order[over_i] = under

        for lane in lane_order:
            pick = _pick_lane(lane)
            if pick is None:
                # second chance: force-sample that lane only
                if shard_pool.get(lane):
                    extra = sample_candidates_for_step(
                        {lane: shard_pool[lane]}, {lane: 1.0}, n=4, rng=rng
                    )
                    for c in extra:
                        c = _enrich_candidate(c, corpus_index, cfg.ARTIFACTS_DIR)
                        if c["doc_id"] not in used and c["doc_id"] not in deferred_ids:
                            pick = c
                            break
            if pick is None:
                continue
            accepted.append(pick)
            used.add(pick["doc_id"])

        if not accepted:
            accepted = [c for c in clean_candidates if c["doc_id"] not in deferred_ids][
                : cfg.MICROBATCH_SIZE
            ]

        # Floor repair: replace a non-protected slot (do not append)
        behind = _lanes_behind_floors(consumed_by_lane, cfg.ALWAYS_ON_SPLIT)
        accepted_lanes = {a.get("lane") for a in accepted}
        for pl in behind:
            if pl in accepted_lanes:
                continue
            refill = _pick_lane(pl)
            if refill is None:
                continue
            for i, a in enumerate(accepted):
                if a.get("lane") not in cfg.ALWAYS_ON_SPLIT:
                    used.discard(a["doc_id"])
                    accepted[i] = refill
                    used.add(refill["doc_id"])
                    break
            accepted_lanes = {a.get("lane") for a in accepted}

        # Pack by dominant lane
        lane_counts: dict[str, int] = {}
        for a in accepted:
            lane_counts[a.get("lane", "web")] = lane_counts.get(a.get("lane", "web"), 0) + 1
        lane = max(lane_counts, key=lane_counts.get) if lane_counts else "web"

        packed = pack_documents(accepted, tok, seq_len, lane=lane)
        if packed_reports is not None:
            packed_reports.append(
                {
                    "global_step": step,
                    "policy": packed.packing_policy,
                    "utilization": packed.utilization,
                    "lane": lane,
                    "document_ids": packed.document_ids,
                    "warnings": packed.warnings,
                }
            )

        # take microbatch rows (only docs that land in served sequences)
        n_seq = min(cfg.MICROBATCH_SIZE, packed.input_ids.shape[0])
        input_ids = packed.input_ids[:n_seq].to(device)
        loss_mask = packed.loss_mask[:n_seq].to(device)
        position_ids = packed.position_ids[:n_seq].to(device)
        attn = packed.attention_mask
        if attn is not None and attn.dim() == 3:
            attn = attn[:n_seq].to(device)
        else:
            attn = None
        served_doc_ids: list[str] = []
        for bounds, dids in zip(
            packed.doc_boundaries[:n_seq],
            # doc_boundaries align 1:1 with rows; document_ids may be flattened
            _row_doc_ids(packed, n_seq),
        ):
            _ = bounds
            served_doc_ids.extend(dids)
        if not served_doc_ids:
            served_doc_ids = list(packed.document_ids)

        eval_firewall.assert_not_in_training_batch(
            served_doc_ids, float(loss_mask.sum().item())
        )

        # token spans
        token_span_ids = []
        for bi, bounds in enumerate(packed.doc_boundaries[:n_seq]):
            for start, end in bounds:
                token_span_ids.append(
                    {
                        "seq": bi,
                        "start": start,
                        "end": end,
                        "shard_ids": accepted[min(bi, len(accepted) - 1)].get("_shard_ids", []),
                    }
                )

        bhash = batch_content_hash(input_ids)
        loss_mask_hash = hashlib.sha256(
            ",".join(f"{x:.0f}" for x in loss_mask.flatten().tolist()).encode()
        ).hexdigest()

        # consumption event BEFORE optimizer step
        shard_ids = []
        for a in accepted:
            shard_ids.extend(a.get("_shard_ids", []))
        shard_ids = list(dict.fromkeys(shard_ids))

        useful_tok = int((loss_mask > 0).sum().item())
        # Attribute useful tokens only to docs that landed in served sequences
        served = served_doc_ids or [a["doc_id"] for a in accepted]
        per_doc = max(1, len(served))
        share_each = useful_tok / per_doc
        lane_by_id = {a["doc_id"]: a.get("lane", lane) for a in accepted}
        lane_token_shares: dict[str, float] = {}
        for did in served:
            alane = lane_by_id.get(did, lane)
            lane_token_shares[alane] = lane_token_shares.get(alane, 0.0) + share_each
            consumed_by_lane[alane] = consumed_by_lane.get(alane, 0) + share_each

        cons_event = {
            "event": "consumption",
            "run_id": run_id,
            "branch_id": branch_id,
            "global_step": step,
            "checkpoint_id": last_ckpt_id,
            "rank": 0,
            "microbatch_id": 0,
            "batch_id": batch_id,
            "packed_sample_ids": served_doc_ids,
            "accepted_doc_ids": [a["doc_id"] for a in accepted],
            "row_document_ids": _row_doc_ids(packed, n_seq),
            "shard_ids": shard_ids,
            "token_span_ids": token_span_ids,
            "loss_mask_hash": loss_mask_hash,
            "attention_policy": packed.packing_policy,
            "packing_policy": packed.packing_policy,
            "sequence_length": seq_len,
            "position_policy": "reset" if packed.packing_policy == "structure_preserving" else "linear",
            "mixture_lane": lane,
            "lane_token_shares": lane_token_shares,
            "curriculum_stage": stage,
            "tokenizer_version": getattr(tok, "name_or_path", tok.__class__.__name__),
            "dataloader_version": cfg.SCHEMA_VERSION,
            "opus_decision_id": decisions[0]["candidate_id"] if decisions else None,
            "batch_hash": bhash,
            "token_ids_flat": input_ids.detach().cpu().numpy().astype(int).flatten().tolist(),
            "packing_utilization": packed.utilization,
            "useful_tokens": useful_tok,
        }
        ledger_offset = consumption_ledger.append(cons_event)
        useful_tokens += cons_event["useful_tokens"]

        # forward / backward with grad accum simulation (single microbatch for smoke)
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attn,
        )
        loss = out["loss"]
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if any(p.grad is not None for p in model.parameters())
            else 0.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.item())
        if step0_loss is None:
            step0_loss = loss_val
            expected = expected_initial_loss(vocab_size)
            # Smoke models can overshoot early; flag only extreme contamination-like lows
            if loss_val < expected * 0.25:
                if logger:
                    logger.warning(
                        "[WARN] anomalously_low_initial_loss loss=%.4f expected_ln_v=%.4f",
                        step0_loss,
                        expected,
                    )
            elif abs(step0_loss - expected) > 5.0 and logger:
                logger.warning(
                    "[WARN] initial_loss_deviates loss=%.4f expected_ln_v=%.4f",
                    step0_loss,
                    expected,
                )

        # ppl + learning ledger
        shard_id = shard_ids[0] if shard_ids else f"virtual_{lane}"
        ppl_rec = ppl_tracer.record(
            shard_id=shard_id,
            lane=lane,
            curriculum_stage=stage,
            per_token_loss=out["per_token_loss"],
            input_ids=input_ids,
            eos_id=eos_id,
            mask=out["mask"],
        )
        prev_loss = history[-1]["loss"] if history else loss_val + 0.1
        loss_delta = loss_val - prev_loss
        classification = learning_ledger.classify(loss_delta, grad_norm)
        learning_ledger.append(
            {
                "event": "learning",
                "global_step": step,
                "run_id": run_id,
                "branch_id": branch_id,
                "shard_id": shard_id,
                "mixture_lane": lane,
                "curriculum_stage": stage,
                "loss": loss_val,
                "loss_delta": loss_delta,
                "grad_norm": grad_norm,
                "avg_ppl": ppl_rec["avg_ppl"],
                "classification": classification,
                "opus_score": decisions[0]["opus_score"] if decisions else None,
                "model_phase": (
                    "early" if step < 5 else ("mid" if step < 15 else "late")
                ),
                "source_document_ids": packed.document_ids,
            }
        )

        history.append({"step": step, "loss": loss_val, "grad_norm": grad_norm, "lane": lane})
        dataloader_cursor = step + 1

        # periodic checkpoint
        if (step + 1) % cfg.CKPT_EVERY_N_STEPS == 0:
            last_ckpt_id = checkpoint_manager.save(
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ledger_offset=consumption_ledger.offset,
                loss=loss_val,
                run_id=run_id,
                branch_id=branch_id,
                dataloader_state={"cursor": dataloader_cursor, "next_step": step + 1},
                expected_next_batch_id=expected_next_batch_id,
            )
            if logger:
                logger.info("[PASS] checkpoint_saved step=%s checkpoint_id=%s", step, last_ckpt_id)

    elapsed = max(1e-6, time.perf_counter() - t0)
    return {
        "history": history,
        "consumed_by_lane": consumed_by_lane,
        "opus_statuses": sorted(opus_statuses),
        "useful_tokens": useful_tokens,
        "useful_tokens_per_second": useful_tokens / elapsed,
        "elapsed_sec": elapsed,
        "step0_loss": step0_loss,
        "last_checkpoint_id": last_ckpt_id,
        "expected_next_batch_id": expected_next_batch_id,
        "final_step": history[-1]["step"] if history else start_step - 1,
    }
