"""Learning-rate schedules: cosine and Warmup-Stable-Decay (WSD)."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """Warmup-Stable-Decay schedule.

    Linear warmup -> hold peak -> cosine decay to min_lr.
    Adapted from Sess_11_NLM_Study_Guide_Part2.md section 15.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            alpha = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
        if step < (self.warmup_steps + self.stable_steps):
            return list(self.base_lrs)
        if step < self.total_steps:
            decay_progress = float(step - self.warmup_steps - self.stable_steps) / float(
                max(1, self.decay_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return [max(self.min_lr, base_lr * cosine_decay) for base_lr in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]


def cosine_lr(step: int, base_lr: float, t_max: int, min_lr: float = 0.0) -> float:
    """Cosine decay from base_lr to min_lr over t_max steps (step starts at 0)."""
    if t_max <= 0:
        return base_lr
    progress = min(max(step, 0), t_max) / float(t_max)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def linear_warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear ramp from 0 to base_lr over warmup_steps (step is 1-indexed friendly)."""
    if warmup_steps <= 0:
        return base_lr
    # step is typically 1..N after optimizer.step; scale so step==warmup -> peak
    return base_lr * min(1.0, float(step) / float(warmup_steps))
