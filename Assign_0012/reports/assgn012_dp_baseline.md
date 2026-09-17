# Assgn012 DP Baseline

- Demo parameters N_demo = **49920**
- Process world_size = **1** (distributed=False)
- Steps = 3; losses = 5.0493, 4.9239, 4.7990
- Max |w_i - w_0| after sync = **0.000e+00** (expect ~0 under DP)
- Peak RSS (rank0) ≈ **329.6 MiB**
- DP local training state ≈ 16 × N_demo = **0.7617 MiB**

Averaging gradients keeps replicas identical (Session 12 Data Parallelism).
30B numbers are **not** loaded here — see overlay tables.
