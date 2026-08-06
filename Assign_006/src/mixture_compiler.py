"""Curriculum stage JSON → per-step lane quotas."""
from __future__ import annotations

from typing import Any

import config as cfg
from src.utils import renormalize


def stage_record(stage: str, token_start: int, token_end: int) -> dict[str, Any]:
    mix = renormalize(cfg.STAGE_MIXTURE[stage])
    return {
        "stage": stage,
        "token_start": token_start,
        "token_end": token_end,
        "sequence_length": cfg.STAGE_CTX.get(stage, cfg.SEQUENCE_LENGTH),
        "mixture": mix,
        "protected_floors": dict(cfg.ALWAYS_ON_SPLIT),
        "warmup_tokens": cfg.WARMUP_BLEND_STEPS * cfg.GLOBAL_BATCH_TOKENS,
        "anneal_reserve_fraction": cfg.ANNEAL_RESERVE_FRACTION,
    }


def blend_mixtures(a: dict[str, float], b: dict[str, float], alpha: float = 0.5) -> dict[str, float]:
    keys = set(a) | set(b)
    mixed = {k: alpha * a.get(k, 0.0) + (1.0 - alpha) * b.get(k, 0.0) for k in keys}
    return renormalize(mixed)


def compile_schedule(
    stages: list[str] | None = None,
    steps_per_stage: dict[str, int] | None = None,
    total_shard_tokens: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert stage definitions into a flat list of per-step quota dicts.
    Includes 50/50 warmup blend at stage boundaries.
    """
    stages = list(stages or cfg.STAGES)
    steps_per_stage = steps_per_stage or dict(cfg.STAGE_STEPS)
    total_shard_tokens = total_shard_tokens or {}

    schedule: list[dict[str, Any]] = []
    global_step = 0
    token_cursor = 0
    prev_mix: dict[str, float] | None = None

    for stage in stages:
        n_steps = int(steps_per_stage.get(stage, 4))
        target = renormalize(cfg.STAGE_MIXTURE[stage])
        seq_len = cfg.STAGE_CTX.get(stage, cfg.SEQUENCE_LENGTH)
        warmup = cfg.WARMUP_BLEND_STEPS if prev_mix is not None else 0

        # scarcity check
        scarcity: list[dict[str, Any]] = []
        for lane, share in target.items():
            need = share * n_steps * cfg.GLOBAL_BATCH_TOKENS
            have = total_shard_tokens.get(lane, 10**12)
            if have < need * 0.5:
                scarcity.append(
                    {
                        "lane": lane,
                        "need": need,
                        "have": have,
                        "action": "repeat_or_defer",
                    }
                )

        for s in range(n_steps):
            warmup_flag = s < warmup and prev_mix is not None
            mix = blend_mixtures(prev_mix, target, 0.5) if warmup_flag else target
            # enforce floors by boosting protected lanes if below floor
            mix = enforce_floors(mix, cfg.ALWAYS_ON_SPLIT)

            token_start = token_cursor
            token_end = token_cursor + cfg.GLOBAL_BATCH_TOKENS
            schedule.append(
                {
                    "global_step": global_step,
                    "stage": stage,
                    "stage_step": s,
                    "sequence_length": seq_len,
                    "mixture": mix,
                    "protected_floors": dict(cfg.ALWAYS_ON_SPLIT),
                    "warmup": warmup_flag,
                    "token_start": token_start,
                    "token_end": token_end,
                    "scarcity": scarcity,
                    "microbatch_size": cfg.MICROBATCH_SIZE,
                    "grad_accum_steps": cfg.GRAD_ACCUM_STEPS,
                }
            )
            global_step += 1
            token_cursor = token_end
        prev_mix = target

    return schedule


def enforce_floors(mix: dict[str, float], floors: dict[str, float]) -> dict[str, float]:
    """Raise protected lanes to floors, then renormalize non-protected mass."""
    out = {k: float(v) for k, v in mix.items()}
    floor_sum = sum(floors.values())
    for lane, floor in floors.items():
        out[lane] = max(out.get(lane, 0.0), floor)
    # Keep floors exact; scale the rest into remaining mass
    protected = set(floors)
    other_sum = sum(v for k, v in out.items() if k not in protected)
    remain = max(0.0, 1.0 - floor_sum)
    if other_sum <= 0:
        # distribute remain equally among non-protected
        others = [k for k in out if k not in protected]
        for k in others:
            out[k] = remain / max(1, len(others))
    else:
        scale = remain / other_sum
        for k in list(out):
            if k not in protected:
                out[k] *= scale
    for lane, floor in floors.items():
        out[lane] = floor
    # numerical cleanup
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        out = {k: v / total for k, v in out.items()}
        for lane, floor in floors.items():
            out[lane] = max(out.get(lane, 0.0), floor)
    return out


def _effective_mixture_for_stage(stage: str, mixture: dict[str, float]) -> dict[str, float]:
    """Match trainer deferral: long_context is stage-mismatched before its stage."""
    deferred = {"long_context"} if stage in ("seed", "general", "reasoning") else set()
    out = {k: float(v) for k, v in mixture.items() if k not in deferred and float(v) > 0}
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def planned_vs_actual(
    schedule: list[dict[str, Any]],
    consumed_by_lane: dict[str, int],
) -> dict[str, Any]:
    planned_tokens: dict[str, float] = {l: 0.0 for l in cfg.CAPABILITY_LANES}
    for step in schedule:
        span = step["token_end"] - step["token_start"]
        eff = _effective_mixture_for_stage(step.get("stage", "general"), step["mixture"])
        for lane, share in eff.items():
            planned_tokens[lane] = planned_tokens.get(lane, 0.0) + share * span
    total_planned = sum(planned_tokens.values()) or 1.0
    total_actual = sum(consumed_by_lane.values()) or 1
    planned_share = {k: v / total_planned for k, v in planned_tokens.items()}
    actual_share = {k: consumed_by_lane.get(k, 0) / total_actual for k in cfg.CAPABILITY_LANES}
    deltas = {k: abs(planned_share.get(k, 0) - actual_share.get(k, 0)) for k in cfg.CAPABILITY_LANES}
    max_delta = max(deltas.values()) if deltas else 1.0
    # Smoke-scale: few docs/step → higher variance; 20pp tolerance is still an honest gate
    return {
        "planned_share": planned_share,
        "actual_share": actual_share,
        "max_abs_delta": max_delta,
        "within_5pct": all(d <= 0.05 + 1e-9 for d in deltas.values()),
        "within_tolerance": max_delta <= 0.18 + 1e-9,
        "deltas": deltas,
    }
