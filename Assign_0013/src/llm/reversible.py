"""Euler and midpoint reversible residual steps (Gal et al. 2025).

Midpoint is algebraically invertible:

```text
p_next = p_prev + 2h * f(p_curr)
p_prev = p_next - 2h * f(p_curr)
```

Euler uses a first-order residual and recomputes f from saved p_curr
(f internals are not stored):

```text
p_next = p_curr + h * f(p_curr)
```

The nn.Module for f is bound by closure so autograd.Function.apply
only receives tensors.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch.autograd import Function


FModule = Callable[[Tensor], Tensor]


def midpoint_step(p_prev: Tensor, p_curr: Tensor, f_mod: FModule, h: float) -> Tensor:
    h_t = p_curr.new_tensor(float(h))

    class _MidpointFn(Function):
        @staticmethod
        def forward(ctx, p_prev_t: Tensor, p_curr_t: Tensor, h_tensor: Tensor) -> Tensor:
            step = float(h_tensor.item())
            ctx.step = step
            with torch.no_grad():
                f_out = f_mod(p_curr_t)
                p_next = p_prev_t + (2.0 * step) * f_out
            ctx.save_for_backward(p_curr_t, p_next)
            return p_next

        @staticmethod
        def backward(ctx, grad_p_next: Tensor):
            p_curr_t, p_next = ctx.saved_tensors
            step = ctx.step
            p_curr_g = p_curr_t.detach().requires_grad_(True)
            with torch.enable_grad():
                f_out = f_mod(p_curr_g)
            grad_p_prev = grad_p_next
            with torch.enable_grad():
                torch.autograd.backward(f_out, grad_p_next * (2.0 * step))
            grad_p_curr = p_curr_g.grad
            return grad_p_prev, grad_p_curr, None

    return _MidpointFn.apply(p_prev, p_curr, h_t)


def euler_step(p_curr: Tensor, f_mod: FModule, h: float) -> Tensor:
    h_t = p_curr.new_tensor(float(h))

    class _EulerFn(Function):
        @staticmethod
        def forward(ctx, p_curr_t: Tensor, h_tensor: Tensor) -> Tensor:
            step = float(h_tensor.item())
            ctx.step = step
            with torch.no_grad():
                f_out = f_mod(p_curr_t)
                p_next = p_curr_t + step * f_out
            ctx.save_for_backward(p_curr_t)
            return p_next

        @staticmethod
        def backward(ctx, grad_p_next: Tensor):
            (p_curr_t,) = ctx.saved_tensors
            step = ctx.step
            p_curr_g = p_curr_t.detach().requires_grad_(True)
            with torch.enable_grad():
                f_out = f_mod(p_curr_g)
                p_next = p_curr_g + step * f_out
            torch.autograd.backward(p_next, grad_p_next)
            return p_curr_g.grad, None

    return _EulerFn.apply(p_curr, h_t)


def midpoint_reconstruct(p_next: Tensor, p_curr: Tensor, f_mod: FModule, h: float) -> Tensor:
    """Algebraic inverse used by unit tests (no autograd)."""
    with torch.no_grad():
        f_out = f_mod(p_curr)
        return p_next - (2.0 * float(h)) * f_out
