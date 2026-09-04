# Session 10 — MFU Report (Task 5)

## Formula

$$\text{MFU} = \frac{6 \times N \times \text{tokens/sec}}{\text{peak TFLOP/s}}$$

## Substituted values

| Metric | Value |
| :--- | :--- |
| Model parameters $N$ | 478,720 |
| Tokens per second | 11,380.2 |
| Achieved TFLOP/s | 0.0327 |
| Peak TFLOP/s | 0.5 |
| Peak source | Conservative CPU estimate (0.5 TFLOP/s) |
| **MFU %** | **6.54%** |
| Assignment target | 40% |
| Industry healthy band | 35-50% |
| Gap to 40% target | 33.46 percentage points |

## Bottleneck narrative

On **CPU** (cpu), this small MiniGPT run is primarily **memory- and kernel-launch-bound**: micro-batch is modest, the model is tiny compared to production LLMs, and CPU-side work limits throughput when CUDA is unavailable. ERA V4 explicitly traded MFU for RAM via reversibility — the same loss curve can hide 4× compute waste (e.g. ~8% vs ~45% MFU). Long-context phases increase activation memory; changing batch semantics mid-curriculum (V4 mistake) would further distort averages without showing up in loss alone.

## Long-context note

When scaling 4K → 8K → … → 1M context, keep **global batch token count fixed per phase**. Longer sequences reduce micro-batch rows per GPU and increase accumulation steps, which typically lowers MFU unless checkpointing and parallelism are retuned.
