# Assgn014 smoke/full summary

version=v009_colab_t4_smoke
N_dense=20856576  N_moe_total=79869696  N_moe_active=26785536
smoke=True  batch=8  seq_len=512
tokens_dense=5001216  tokens_moe=5001216
loss_at_convert=9.015777  final_moe_loss=9.013443
max_vio=3.0000  dead_experts=6.0
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
upcycle=partition_drop  n_experts=8  top_k=2  bias_gamma=0.001
use_ce_aux=False  router_dtype=float32

n_dense=20856576  range=[18000000,22000000]  n_ok=True
n_moe_total=79869696  n_moe_active=26785536  moe_bigger=True
tokens_dense=5001216  tokens_moe=5001216  token_ok=True
loss_at_convert=9.015776634216309  final_moe_loss=9.013442993164062  loss_ok=True
max_vio=3.0  dead_experts=6.0  balance_ok=True
optimizer=AdamW  use_ce_aux=False  router_dtype=float32
aux_off=True  router_fp32=True  png_ok=True
version=v009_colab_t4_smoke  reports_dir=/content/drive/MyDrive/Assign_0014/reports/v009_colab_t4_smoke
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
primary_pngs=3  (need 3)
all_ok=True  (PASS if True; FAIL if False)
