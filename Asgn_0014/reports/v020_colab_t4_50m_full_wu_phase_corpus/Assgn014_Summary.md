# Assgn014 smoke/full summary

version=v020_colab_t4_50m_full_wu_phase_corpus
N_dense=20856576  N_moe_total=79869696  N_moe_active=26785536
smoke=False  batch=8  seq_len=512
tokens_dense=25001984  tokens_moe=25001984
loss_at_convert=1.282513  final_moe_loss=0.691455
max_vio=0.8584  dead_experts=0.0
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
upcycle=partition_drop  n_experts=8  top_k=2  bias_gamma=0.001
use_ce_aux=False  router_dtype=float32
warmup_steps_floor=2000  warmup_steps_effective=6104  use_three_tier=True  version_tag=wu_phase_corpus  budget_slug=50m

n_dense=20856576  range=[18000000,22000000]  n_ok=True
n_moe_total=79869696  n_moe_active=26785536  moe_bigger=True
tokens_dense=25001984  tokens_moe=25001984  token_ok=True
loss_at_convert=1.2825130224227905  final_moe_loss=0.6914553642272949  loss_ok=True
learn_dense_ok=True  learn_moe_ok=True  ce_floor_ln_v=9.010913347279288  data_source=cache:data/tinystories/packed_ids.pt
max_vio=0.8583984375  dead_experts=0.0  balance_ok=True
optimizer=AdamW  use_ce_aux=False  router_dtype=float32
aux_off=True  router_fp32=True  png_ok=True
version=v020_colab_t4_50m_full_wu_phase_corpus  reports_dir=/content/drive/MyDrive/Assign_0014/reports/v020_colab_t4_50m_full_wu_phase_corpus
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
primary_pngs=3  (need 3)
all_ok=True  (PASS if True; FAIL if False)

## How to read this run

- Recipe tag `wu_phase_corpus`: probabilistic warm-up auto-extends to cover the MoE phase (floor=2000, effective=6104 steps), then hard top-k only if phase ends first; three-tier loss-free bias (gamma1=0.0005, gamma2=0.002, gamma3=0.01) with bias clamp +/-5.
- Data: mode=corpus source=cache:data/tinystories/packed_ids.pt tokenizer=configs/assgn014_bpe_8192.json. Graded full uses structured corpus (not randint).
- CE floor ln(V)=9.0109 (V=8192). Full gates require final_dense_loss <= floor-0.5 and final_moe_loss <= convert-0.05.
- balance_ok needs dead_experts==0. Smoke no longer forgives dead experts (MaxVio slack up to 3.0). Full also needs MaxVio<=2.0.
- PRIMARY PNGs (graded proof): assgn014_loss_trajectories.png, assgn014_expert_load_health.png, assgn014_memory_speed_dashboard.png. pipeline_bubble_*.png is NOT required.
- Next: if all_ok and dead==0, continue ladder. If dead>0, stop before graded full. Failed full -> new version folder; do not RESUME across different budgets.
- Folder naming: vNNN_host_gpu_budget_kind_tag (this run budget=50m, tag=wu_phase_corpus). Synthetic-floor history (v018) is instrumentation-only; corpus runs prove learning.
