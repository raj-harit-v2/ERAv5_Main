# Assgn014 smoke/full summary

version=v016_colab_t4_4k_smoke_wu_phase
N_dense=20856576  N_moe_total=79869696  N_moe_active=26785536
smoke=True  batch=8  seq_len=512
tokens_dense=4096  tokens_moe=4096
loss_at_convert=9.104097  final_moe_loss=9.077488
max_vio=0.7412  dead_experts=0.0
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
upcycle=partition_drop  n_experts=8  top_k=2  bias_gamma=0.001
use_ce_aux=False  router_dtype=float32
warmup_steps_floor=2000  warmup_steps_effective=2000  use_three_tier=True  version_tag=wu_phase  budget_slug=4k

n_dense=20856576  range=[18000000,22000000]  n_ok=True
n_moe_total=79869696  n_moe_active=26785536  moe_bigger=True
tokens_dense=4096  tokens_moe=4096  token_ok=True
loss_at_convert=9.104097366333008  final_moe_loss=9.07748794555664  loss_ok=True
max_vio=0.7412109375  dead_experts=0.0  balance_ok=True
optimizer=AdamW  use_ce_aux=False  router_dtype=float32
aux_off=True  router_fp32=True  png_ok=True
version=v016_colab_t4_4k_smoke_wu_phase  reports_dir=/content/drive/MyDrive/Assign_0014/reports/v016_colab_t4_4k_smoke_wu_phase
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
primary_pngs=3  (need 3)
all_ok=True  (PASS if True; FAIL if False)

## How to read this run

- Recipe tag `wu_phase`: probabilistic warm-up auto-extends to cover the MoE phase (floor=2000, effective=2000 steps), then hard top-k only if phase ends first; three-tier loss-free bias (gamma1=0.0005, gamma2=0.002, gamma3=0.01) with bias clamp +/-5.
- CE floor: vocab=8192 -> random-token floor ln(8192)~=9.011. Tiny late loss drops on synthetic data are expected.
- balance_ok needs dead_experts==0. Smoke no longer forgives dead experts (MaxVio slack up to 3.0). Full also needs MaxVio<=2.0.
- PRIMARY PNGs (graded proof): assgn014_loss_trajectories.png, assgn014_expert_load_health.png, assgn014_memory_speed_dashboard.png. pipeline_bubble_*.png is NOT required.
- Next: if all_ok and dead==0, continue ladder. If dead>0, stop before graded full. Failed full -> new version folder; do not RESUME across different budgets.
- Folder naming: vNNN_host_gpu_budget_kind_tag (this run budget=4k, tag=wu_phase). Older folders without _wu_phase are pre-phase-warmup history (wu2k ladder OK but full collapsed after step 2000).
