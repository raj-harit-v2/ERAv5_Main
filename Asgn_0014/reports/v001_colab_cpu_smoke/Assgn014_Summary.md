# Assgn014 smoke/full summary

version=v001_colab_cpu_smoke
N_dense=20856576  N_moe_total=79869696  N_moe_active=26785536
smoke=True  batch=2  seq_len=512
tokens_dense=4096  tokens_moe=4096
loss_at_convert=9.087194  final_moe_loss=9.085471
max_vio=3.0000  dead_experts=5.0
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
upcycle=partition_drop  n_experts=8  top_k=2  bias_gamma=0.001
use_ce_aux=False  router_dtype=float32

n_dense=20856576  range=[18000000,22000000]  n_ok=True
n_moe_total=79869696  n_moe_active=26785536  moe_bigger=True
tokens_dense=4096  tokens_moe=4096  token_ok=True
loss_at_convert=9.087194442749023  final_moe_loss=9.085471153259277  loss_ok=True
max_vio=3.0  dead_experts=5.0  balance_ok=True
optimizer=AdamW  use_ce_aux=False  router_dtype=float32
aux_off=True  router_fp32=True  png_ok=True
version=v001_colab_cpu_smoke  reports_dir=/content/drive/MyDrive/Assign_0014/reports/v001_colab_cpu_smoke
usd_per_hour=0.12  rate_source=ASSGN014_USD_PER_HOUR=0.12
primary_pngs=3  (need 3)
all_ok=True  (PASS if True; FAIL if False)
