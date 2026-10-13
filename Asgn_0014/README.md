# Assign_0014 — Dense → MoE Upcycling 

Train a **~20.9M dense** decoder on a **structured corpus**, **convert** FFNs to MoE (partition/drop), continue training with **loss-free three-tier bias** + probabilistic warm-up, and prove **material CE drops** (dense + post-convert) plus routing health with **PRIMARY graphics**. Graded host: **Google Colab Tesla T4**.

> **Graded full:** `reports/v020_colab_t4_50m_full_wu_phase_corpus` — corpus TinyStories/BPE-8192, Colab T4, 25M+25M.  
> **Prior runs:** `v018_…_wu_phase` (synthetic routing baseline); `v015_…_wu2k` (warm-up FAIL, dead=6).

**Takeaway:** Graded claim = **structured next-token learning** (dense CE well below `ln(V)`, MoE CE drops after convert) **and** routing health (`dead=0`, MaxVio bound).

---

## High-level architecture

```mermaid
flowchart TB
    classDef entry fill:#1e3a5f,stroke:#0d2137,color:#ffffff,font-weight:bold
    classDef orch fill:#0f766e,stroke:#115e59,color:#ffffff,font-weight:bold
    classDef model fill:#0369a1,stroke:#075985,color:#ffffff,font-weight:bold
    classDef util fill:#b45309,stroke:#92400e,color:#ffffff,font-weight:bold
    classDef out fill:#be123c,stroke:#9f1239,color:#ffffff,font-weight:bold

    Lab["assgn014_moe_lab.py"]
    Nb["assgn014_moe_train_ladder.ipynb"]
    Pytest["test_assgn014.py"]

    ML["moe_lab.run_all"]

    Tiny["tiny_lm TinyDenseLM"]
    MoE["moe UpcycleRouter"]
    Train["train_moe"]
    Graph["moe_graphics"]
    Ver["versioning"]
    Rates["rates USD hour"]

    Rep["reports/vNNN_* csv json"]
    PNG["PRIMARY PNGs"]

    Lab --> ML
    Nb --> ML
    Pytest --> MoE
    Pytest --> ML

    ML --> Tiny
    ML --> MoE
    ML --> Train
    ML --> Graph
    ML --> Ver
    ML --> Rates
    Graph --> PNG
    Graph --> Rep
    Train --> Rep
    Ver --> Rep

    class Lab,Nb,Pytest entry
    class ML orch
    class Tiny,MoE model
    class Train,Graph,Ver,Rates util
    class Rep,PNG out
```

## Lab process

```mermaid
flowchart TB
    classDef stage fill:#1e3a5f,stroke:#0d2137,color:#ffffff,font-weight:bold
    classDef compute fill:#0f766e,stroke:#115e59,color:#ffffff,font-weight:bold
    classDef show fill:#6d28d9,stroke:#5b21b6,color:#ffffff,font-weight:bold
    classDef gate fill:#be123c,stroke:#9f1239,color:#ffffff,font-weight:bold

    Dense["1 Dense train 25M tokens"]
    Upcycle["2 Partition drop upcycle"]
    Warm["3 MoE + three-tier bias + wu_phase"]
    Graphics["4 PRIMARY graphics"]
    Gates["5 Gates all_ok Summary"]

    Dense --> Upcycle
    Upcycle --> Warm
    Warm --> Graphics
    Graphics --> Gates

    class Dense,Upcycle stage
    class Warm compute
    class Graphics show
    class Gates gate
```

---

## 1. What this lab proves

```text
Dense CE falls (~21M TinyDenseLM)
  -> upcycle FFN to MoE (partition/drop, FP32 router, CE-aux OFF)
  -> probabilistic warm-up covers MoE phase (floor 2000; full ~6104)
  -> three-tier loss-free bias; MoE CE not worse after convert
  -> MaxVio bounded; dead experts = 0 on full
  -> 3 PRIMARY PNGs under reports/vNNN_host_gpu_budget_kind_tag
```

| Claim | Lab verb | Not this lab |
| :--- | :--- | :--- |
| Dense→MoE convert + continued loss | partition/drop upcycle | sklearn Linear |
| Loss-free balancing | three-tier bias for top-k only | CE auxiliary load loss |
| Bigger total, controlled active | N_moe_total > N_dense | Multi-GPU EP all-to-all |

### Model shape (lab defaults)

| Piece | Value |
| :--- | :--- |
| Dense LM | ~20.9M (`n_embd=384`, `n_layer=10`, vocab 8192) |
| Experts | N=8, top-k=2, SwiGLU width ≈ dense_inner / k |
| MoE total / active | ~80M total / ~27M active |
| Optimizer | AdamW |
| Router | Linear, **FP32** |
| Load balance | Loss-free **three-tier bias** (CE-aux **OFF**) |
| Colab batch / seq | batch=8, seq_len=512 → 4096 tokens/step |

```text
TOTAL params  → memory / Adam state
ACTIVE params → FLOPs / token (top-k experts only)
warmup_steps_effective = max(floor, ceil(tokens_moe / tokens_per_step))
  floor=2000; full MoE ≈ 25e6/4096 ≈ 6104  (tag wu_phase)
CE floor (synthetic uniform IDs, V=8192): ln(8192) ≈ 9.011
```

**Report folders:** `vNNN_colab_t4_50m_full_wu_phase` — budget + recipe tag. Override via `ASSGN014_VERSION_TAG`. Older `…_wu2k` folders are history.

---

## 2. Data source

**Graded default:** structured corpus (`ASSGN014_DATA=corpus`).

```text
TinyStories (HF) or data/corpus_sample.txt fallback
  -> BPE vocab=8192 (configs/assgn014_bpe_8192.json)
  -> packed (B, S+1) batches  — src/utils/corpus.py
```

| Mode | Env | Use |
| :--- | :--- | :--- |
| **corpus** | `ASSGN014_DATA=corpus` | Graded full / learning proof |
| synthetic | `ASSGN014_DATA=synthetic` | Fast wiring only (CE stays near `ln(8192)≈9.011`) |

**Full gates (non-smoke):** `final_dense_loss <= ln(V) - 0.5` and `final_moe_loss <= loss_at_convert - 0.05`, plus prior routing/PNG gates.

---

## 3. Empirical evaluation

A PASS lab writes CSV/JSON under a **versioned** `reports/vNNN_*` folder plus **three PRIMARY PNGs**. Every gate prints operands (never a naked `True`).

### 3.1 Graded full — `v020_colab_t4_50m_full_wu_phase_corpus`

```text
ASSGN014_DATA=corpus  version_tag=wu_phase_corpus
host=colab  gpu=Tesla T4  batch=8  seq_len=512
upcycle=partition_drop  n_experts=8  top_k=2
use_ce_aux=False  router_dtype=float32  use_three_tier=True
warmup_steps_effective=6104
tokens_dense≈25.0M  tokens_moe≈25.0M
data_source=cache:data/tinystories/packed_ids.pt
ce_floor_ln_v≈9.011
final_dense_loss≈1.283   (learn_dense_ok)
loss_at_convert≈1.283
final_moe_loss≈0.691     (learn_moe_ok)
dead_experts=0  max_vio≈0.858  all_ok=True
```

| Recipe | Data | Dense/convert CE | Final MoE CE | dead | Notes |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `wu2k` (v015) | synthetic | ≈9.01 | — | 6 | FAIL (warm-up) |
| `wu_phase` (v018) | synthetic | ≈9.011 | ≈9.011 | 0 | routing OK; CE at `ln(V)` floor |
| `wu_phase_corpus` (**v020**) | **corpus** | **≈1.28** | **≈0.69** | **0** | graded full PASS |

### 3.2 PRIMARY PNGs (`v020`)

![Loss trajectories](reports/v020_colab_t4_50m_full_wu_phase_corpus/assgn014_loss_trajectories.png)

![Expert load health](reports/v020_colab_t4_50m_full_wu_phase_corpus/assgn014_expert_load_health.png)

![Memory / speed dashboard](reports/v020_colab_t4_50m_full_wu_phase_corpus/assgn014_memory_speed_dashboard.png)

| Artifact | Role |
| :--- | :--- |
| `assgn014_loss_trajectories.png` | Dense then MoE; convert marker |
| `assgn014_expert_load_health.png` | MaxVio / dead / load bars |
| `assgn014_memory_speed_dashboard.png` | Peak MiB + tok/s |

---

## 4. Verification gate (`v020`)

```text
n_ok=True  moe_bigger=True  token_ok=True  loss_ok=True
learn_dense_ok=True  learn_moe_ok=True
balance_ok=True  png_ok=True  opt_ok=True  aux_off=True  router_fp32=True
n_dense=20856576  n_moe_total=79869696  dead_experts=0.0  max_vio=0.8583984375
loss_at_convert=1.2825130224227905  final_dense_loss=1.2825130224227905
final_moe_loss=0.6914553642272949  ce_floor_ln_v=9.010913347279288
data_mode=corpus  data_source=cache:data/tinystories/packed_ids.pt
smoke=False  primary_pngs=3  usd_per_hour=0.12
n_experts=8  top_k=2  use_ce_aux=False  router_dtype=float32
all_ok=True  (PASS if True; FAIL if False)
```

`all_ok` (full): dense N in range, MoE bigger, token budgets, learning gates (`learn_dense_ok`, `learn_moe_ok`), **dead_experts==0** and MaxVio≤2, 3 PRIMARY PNGs, AdamW, CE-aux OFF, router FP32.

Earlier synthetic full: `reports/v018_colab_t4_50m_full_wu_phase/`.

Smoke also requires **dead_experts==0** (soft MaxVio slack). Do not climb the ladder on `all_ok` alone if dead>0.

**pytest** (`tests/test_assgn014.py`) proves wiring in `tmp_path`. (operands + PNGs on disk).

---

## 5. From Assign_0012 / Assign_0013

| Assign | Relation |
| :--- | :--- |
| **0012** | ERA lab culture: PRIMARY PNGs, gates with operands, Domain was DP/ZeRO simulation — not shared model code. |
| **0013** | Direct sibling scaffold: ~20.9M TinyLM, **synthetic `randint` tokens**, Colab T4 ladder, versioned `reports/vNNN_*`. A14 drops Euler/midpoint reversibility and adds dense→MoE upcycle + three-tier bias + phase-scaled warm-up. Dense N matches A13 (`20856576`). |

---

## 6. Extra docs → lab mapping

Session Extra pack (`Asgn_14_NLM_Extra`, literacy beside the repo) maps as:

| Topic | In this lab? |
| :--- | :--- |
| Three-tier bias controller | **Yes** (default on) |
| Probabilistic warm-up after clone/upcycle | **Yes** (auto-extends to MoE phase; floor 2000) |
| Drop-r / top-k sweeps | Optional literacy |

---

## 7. Quick start (local smoke)

```text
cd Asgn_0014
cp .env.example .env
pip install -r requirements.txt
python tests/assgn014_moe_lab.py --smoke --tokens-dense 4096 --tokens-moe 8192 --batch 2 --seq-len 128
python -m pytest tests/test_assgn014.py -q
```

Colab T4 defaults to **batch=8**. Full budgets (25M + 25M):

```text
python tests/assgn014_moe_lab.py --full --batch 8
```

---

## 8. Colab (ladder — corpus graded path)

**Runbook:** [`tests/COLAB_CORPUS_RUNBOOK.md`](tests/COLAB_CORPUS_RUNBOOK.md).

**Phase A (local):** `python tests/_make_colab_zip.py` → upload `Assign_0014_colab.zip`.

**Phase B–D (Colab T4):** ladder notebook; `.env` must have `ASSGN014_DATA=corpus` and `ASSGN014_VERSION_TAG=wu_phase_corpus` (Cell 1 sets these). Smoke CE must be **below ~9.01** before full.

```text
Smoke   4k dense + 4k MoE     smoke=True   (corpus — verify learning)
2M–10M  optional ladder        smoke=True
Full    25M + 25M             smoke=False  RESUME=None first start
```

Graded folder: `v020_colab_t4_50m_full_wu_phase_corpus`.

**Phase E (local):** `python tests/self_evaluate.py --reports reports/v020_colab_t4_50m_full_wu_phase_corpus` — need `learn_dense_ok`, `learn_moe_ok`, `all_ok`.

Do **not** RESUME a synthetic `v018_…` folder into a corpus run. Zip extract flat under `reports/<version>/` (or `python tests/extract_zip.py`). Diagnostics cell after full.

Optional short notebook: [`assgn014_moe_train.ipynb`](notebooks/assgn014_moe_train.ipynb).

---

## 9. Repository structure

```text
Asgn_0014/
  src/  tests/  notebooks/  configs/
  reports/v020_colab_t4_50m_full_wu_phase_corpus/  # graded corpus PRIMARY
  reports/v018_colab_t4_50m_full_wu_phase/         # synthetic full (optional)
  reports/v015_colab_t4_50m_full_wu2k/             # warm-up FAIL contrast
  .env.example                                     # real .env gitignored
```

Checkpoints under `reports/**/checkpoints/` and `*.pt` / report zips are gitignored

---

### Why identical expert clones collapse

1. **Upcycle** turns one dense FFN into many experts. If those experts start **identical**, they compute the same function for every token.

2. **Loss does not care** which expert runs — any of them gives the same output — so training never “needs” all of them.

3. The **router** picks top-k. Right after convert it is nearly random, and biases often start at zero, so tiny score noise decides the winners.

4. **Hard top-k** sends tokens (and gradients) only to the chosen experts. Losers get **no weight updates**.

5. Winners improve a little → keep winning → losers stay unused → **dead experts** and load piles onto a few favorites. That is collapse.

6. **Break the symmetry early:** partition/drop so experts are not clones; use **random exploration (Gumbel warm-up)** so losers still get tokens; use **loss-free bias** to push load toward even shares before hard top-k locks in.

---

### One-line summary

**Identical (or near-identical) experts make CE indifferent among them; hard top-k + random router then locks a few winners and starves the rest.** The code mitigates that by **partition+drop (r=0.5)**, then **Gumbel warm-up for the full MoE phase**, plus **three-tier bias**—which is why pure “8 identical clones + hard top-k from step 0” would collapse, and why ending warm-up early (v015) did.
