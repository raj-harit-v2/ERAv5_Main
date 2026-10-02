# Assgn014 smoke/full summary

version=v012_colab_t4_2m_smoke_wu2k
N_dense=20856576  N_moe_total=79869696  N_moe_active=26785536
smoke=True  batch=8  seq_len=512
tokens_dense=1003520  tokens_moe=1003520
loss_at_convert=9.025167  final_moe_loss=9.020344
max_vio=1.7295  dead_experts=0.0
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
upcycle=partition_drop  n_experts=8  top_k=2  bias_gamma=0.001
use_ce_aux=False  router_dtype=float32
warmup_steps=2000  use_three_tier=True  version_tag=wu2k  budget_slug=2m

n_dense=20856576  range=[18000000,22000000]  n_ok=True
n_moe_total=79869696  n_moe_active=26785536  moe_bigger=True
tokens_dense=1003520  tokens_moe=1003520  token_ok=True
loss_at_convert=9.025167465209961  final_moe_loss=9.020343780517578  loss_ok=True
max_vio=1.7294921875  dead_experts=0.0  balance_ok=True
optimizer=AdamW  use_ce_aux=False  router_dtype=float32
aux_off=True  router_fp32=True  png_ok=True
version=v012_colab_t4_2m_smoke_wu2k  reports_dir=/content/drive/MyDrive/Assign_0014/reports/v012_colab_t4_2m_smoke_wu2k
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
primary_pngs=3  (need 3)
all_ok=True  (PASS if True; FAIL if False)

## How to read this run

- Recipe tag `wu2k`: probabilistic warm-up for 2000 MoE steps after upcycle, then hard top-k; three-tier loss-free bias (gamma1=0.0005, gamma2=0.002, gamma3=0.01) with bias clamp +/-5.
- CE floor: vocab=8192 -> random-token floor ln(8192)~=9.011. Tiny late loss drops on synthetic data are expected.
- balance_ok needs dead_experts==0. Smoke no longer forgives dead experts (MaxVio slack up to 3.0). Full also needs MaxVio<=2.0.
- PRIMARY PNGs (graded proof): assgn014_loss_trajectories.png, assgn014_expert_load_health.png, assgn014_memory_speed_dashboard.png. pipeline_bubble_*.png is NOT required.
- Next: if all_ok and dead==0, continue ladder. If dead>0, stop before graded full. Failed full -> new version folder; do not RESUME across different budgets.
- Folder naming: vNNN_host_gpu_budget_kind_tag (this run budget=2m, tag=wu2k). Older folders without _wu2k are pre-harden history.
