# Assgn011 — LR Width Sweep (Task 5)

Matched budget: 40 steps per (width, eta) point, batch=2, block_size=32, AdamW beta2=0.95.
Parameterization: **standard** (SP).

| Width | Best η | Late loss |
| ---: | ---: | ---: |
| 256 | 0.0001 | 5.004481 |
| 512 | 0.000719686 | 4.936011 |
| 1024 | 0.000719686 | 2.013629 |

## Extrapolation to width 4096

```text
SP:  eta_4096 ≈ eta_1024 / 4     (eta ∝ 1/width)
```

| Value | η at 4096 | Confidence | Why |
| :--- | ---: | :--- | :--- |
| Standard parameterization (SP) | 0.000179921 | **LOW** | FULL.txt Section 12: transferring a small-width η under SP can overstate by up to 16× (256→4096); without a μP day we do not claim high confidence |

Value we would use at width 4096 under SP: **0.000179921**, confidence **LOW**.

Primary plot: `assgn011_lr_width_sweep.png`
