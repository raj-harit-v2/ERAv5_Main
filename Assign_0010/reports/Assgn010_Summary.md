# Assgn010 Summary — Session 10 Training Loop

## What silent bugs did we prove?

We reproduced the **15.4% gradient-accumulation averaging bug** (token-weighted 2.6000 vs average-of-averages 3.0000) and demonstrated that **grad norm can move before loss** after injecting a poison step — matching the Session 10 thesis that serious bugs stay silent while loss still looks plausible. MFU reporting shows how identical loss curves can hide massive hardware under-utilization.

## Primary plots

- [Accumulation bug](assgn010_accumulation_bug.png) — discrepancy 15.38%
- [Grad norm vs loss](assgn010_gradnorm_vs_loss.png) — flagged step 15

## MFU table

| Metric | Value |
| :--- | :--- |
| MFU % | 6.54% |
| Achieved TFLOP/s | 0.0327 |
| Peak TFLOP/s | 0.5 |
| Gap to 40% target | 33.46 pp |

## BF16 recommendation

Train in **BF16** with FP32 master weights (16 bytes/param Adam state). See [assgn010_precision_bits.md](assgn010_precision_bits.md) and the ten-format table in [assgn010_fp_format_table.md](assgn010_fp_format_table.md). MXFP4 outlier rescue: [assgn010_mbs_oas_table.md](assgn010_mbs_oas_table.md).

## Long-context note

When curriculum moves 4K → 8K → … → 1M, **do not change global batch token semantics mid-phase** (ERA V4 mistake). Longer context increases activation memory and accumulation steps — retune MFU and checkpointing per phase.

## Artifacts

- [assgn010_tensor_shapes.log](assgn010_tensor_shapes.log)
- [assgn010_gradient_check.txt](assgn010_gradient_check.txt)
- [assgn010_step_metrics.csv](assgn010_step_metrics.csv)
- [assgn010_mfu_report.md](assgn010_mfu_report.md)
- [assgn010_precision_bits.md](assgn010_precision_bits.md)
- [assgn010_fp_format_table.md](assgn010_fp_format_table.md)
- [assgn010_mbs_oas_table.md](assgn010_mbs_oas_table.md)
