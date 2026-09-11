# Assgn011 Summary — Session 11 Optimizers and Schedules

## 1. Adam bias correction at t=1

Without bias correction, a constant-sign first gradient takes a step of about 3.162 × η (theory 3.162). With correction the same step is ≈ 1.00 × η. That early overshoot is why warmup exists.

## 2. Primary plots

- Bias ablation (20 steps): `assgn011_bias_correction_20.png`
- LR width sweep (three minima): `assgn011_lr_width_sweep.png`

## 3. Schedule keep-decision (plan 300, checkpoint 200)

Keep **WSD**. Loss@200 — cosine: 5.558470726013184, WSD: 5.561608791351318. Planned horizon is 300 steps; the keep-decision is the early-stop checkpoint at 200 (FULL §10).

## 4. η at width 4096 + confidence

- SP extrapolation: η_4096 ≈ 0.00017992141825028815 — confidence **LOW** (FULL Section 12: up to 16× overstatement risk under SP when transferring small→large width).

## 5. Tune both sides

Tasks 4 and 5 used matched step budgets, seeds, peak LRs, and warmup lengths across arms.

## 6. Long-context / batch invariant

Inherited from Session 10: when scaling context length, keep **global batch token count fixed per phase**. Longer sequences reduce micro-batch rows and raise accumulation steps; do not silently retune η mid-curriculum.

## 7. Gate snapshot

| Task | OK |
| :--- | :---: |
| 1 Hand Adam | True |
| 2 Bias plot | True |
| 3 Ratio / warmup | True |
| 4 Cosine vs WSD | True |
| 5 LR sweep | True |
