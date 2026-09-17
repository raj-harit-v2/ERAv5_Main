# Assgn012 Summary — Distributed Training I (Data Parallel & ZeRO)

## 1. Continuity from Session 11

Mixed-precision AdamW stores **16 bytes per trainable parameter**. At 30B that is **447.0 GiB** before activations.

## 2. PRIMARY graphics

- Memory ladder: `reports/assgn012_memory_ladder.png`
- W=32 dashboard: `reports/assgn012_w32_dashboard.png`
- Compute vs comm: `reports/assgn012_compute_comm.png`
- Activation vs S (recommended): `reports/assgn012_activation_vs_seq.png`

## 3. Match ZeRO (Session 12 ladder)

- Gate: **PASS**
- Replicated floor: **111.76 GiB** (DP/ZeRO-1 never fit)
- W=32 GiB: DP 447.0 / Z1 122.2 / Z2 68.1 / Z3 14.0

## 4. Demo DP run

- N_demo = 49920; max |w_i-w_0| = 0.0
- Distributed = False; world_size = 1

## 5. Pros / cons per stage

| Stage | Pros | Cons |
| :--- | :--- | :--- |
| DP | Simple; math-identical larger batch | Full 16 B/param replica; never fits 30B |
| ZeRO-1 | Shards 12 B optimizer; same 2P | Still replicates params+grads; never fits 30B |
| ZeRO-2 | Fits from 32 GPUs; still 2P | Thin headroom (~6.4 GiB) before activations |
| ZeRO-3 | Fits from 8 GPUs; lowest static | Comm rises to 3P; more gather traffic |

## 6. Long context

Activations scale with batch × sequence length and sit **on top** of the ladder. ZeRO-2@32 leaves ~6.4 GiB on a 74.5 GiB card — long context forces checkpointing (~+30% compute) or ZeRO-3.

## 7. Industry honesty

Public: DeepSpeed ZeRO (2019), PyTorch FSDP / FSDP2.
Undisclosed: Anthropic Claude and current OpenAI/Gemini production sharding — not invented here.

## 8. Comm/compute reminder

H100: 2.4/7.1 ≈ 34%; B200: 2.4/3.12 ≈ 77% (Session 12).

## 9. Overall gate (boolean + full operand values)

task1_smoke.ok=True  smoke_sum=1.0  world_size=1  backend=none
task2_dp.ok=True  max_abs_diff=0.0  N_demo=49920  peak_rss_mb=329.56640625
task3_zero.ok=True  mismatches=0  W32_GiB=447.0/122.2/68.1/14.0  floor_gib=111.75870895385742
task4_graphics.ok=True  primary_pngs=3
task_json_compare.ok=True  stages_pass=3/3
all_ok=True  (PASS if True; FAIL if False)

Overall gate: **PASS** (all_ok=True)
