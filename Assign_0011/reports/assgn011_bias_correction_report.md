# Assgn011 — Bias Correction Ablation (Task 2)

Constant gradient g = 0.5, eta = 0.001, beta1 = 0.9, beta2 = 0.999.

```text
t=1 without correction: step ≈ eta * (1 - beta1) / sqrt(1 - beta2) ≈ 3.162 * eta
t=1 with correction:    step ≈ eta * 1.000
```

| Metric | Value |
| :--- | :--- |
| Measured off/on ratio at t=1 | 3.162276 |
| Theory (1-beta1)/sqrt(1-beta2) | 3.162278 |
| Crossover T (rel err < 1e-3 thereafter) | not reached within 20 steps |
| Primary plot | `assgn011_bias_correction_20.png` |

## When the difference stops mattering

Within the first 20 steps the relative difference never stays below 1e-3 for all remaining t. Study Guide notes the asymptotic gap shrinks over ~1000 steps; the 20-step plot is the primary deliverable.

Production runs never disable bias correction; this ablation is pedagogical only.
