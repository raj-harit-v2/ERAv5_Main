# Assgn011 — Warmup End / Update-to-Weight Ratio (Task 3)

Model: MiniGPT n_embd=64, n_layer=2, block_size=64.
Optimizer: AdamW (beta1=0.9, beta2=0.95, wd=0.1), peak_lr=0.003, warmup=50 of 300.

```text
ratio_layer = ||w_after - w_before||_2 / (||w_after||_2 + eps)
T* = first step after warmup peak where median ratio changes < 5% across a 5-step window
```

| Metric | Value |
| :--- | :--- |
| Warmup length W | 50 |
| Detected T* | 50 |
| Late median ratio (after warmup) | 7.6709e-03 |
| Target | ~1e-3 |
| CSV rows | 8700 |

## Reasoning

Warmup stops changing the ratio once LR has reached peak (step 50) and the per-layer update magnitudes settle. Detected T* = 50. Healthy post-warmup ratios sit near 1e-3 (FULL.txt Section 9).
