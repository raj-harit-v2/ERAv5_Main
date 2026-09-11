# Assgn011 — Cosine vs WSD (Task 4)

Protocol (Task 6 tune-both-sides): identical MiniGPT, seed=42, AdamW peak_lr=3e-3, warmup=6, batch=4, block_size=64, 300 steps.

| Schedule | Loss @ 200 | Loss @ 300 | Keep? |
| :--- | ---: | ---: | :--- |
| cosine (T_max=300) | 5.558471 | 5.554020 | no |
| WSD (warmup=6, stable≈264, decay=30) | 5.561609 | 5.556404 | yes |

**Keep decision: WSD**

At the early-stop checkpoint (step 200 of a 300-step plan), WSD still holds peak LR while cosine has already decayed toward its fixed horizon. FULL.txt Section 10: cosine penalizes early stop because the remaining decay was budgeted for a longer run. WSD can checkpoint, branch, and continue — so we keep the WSD model.

Primary plot: `assgn011_cosine_vs_wsd.png`

Note: plan horizon = 300 steps; keep-decision checkpoint = step 200 (Loss@300 is optional continuation analysis).
