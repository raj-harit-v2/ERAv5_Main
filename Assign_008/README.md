# Assign_008 — Attention mechanisms 

Clinical smoke package + static chronology wall for ERA V5 Assignment §18. Graded narrative surface is the launch-date timeline; teaching tour and PyTorch smoke pipeline are supporting artifacts.

**Netlify publish :**  

**Assignment Q2 (coach):** [`submission_artifacts/Q2_TIMELINE_NOTE.md`](submission_artifacts/Q2_TIMELINE_NOTE.md)

---

## 1. Core architecture & mathematical foundations

### Engineering problem

Given a user sentence, produce a pedagogical trace of tokenization → embedding → attention → logits without allocating long-context tensors. Separately, present **18 field mechanisms** ordered by **primary-source launch date** (`year_sort`), each with problem / buy / give-up / when, and optional widget iframes.

### Smoke tensor contract (`config.py`)

| Symbol | Value | Role |
| :--- | ---: | :--- |
| V | 512 | vocab |
| D | 64 | model width |
| T | 64 | padded sequence |
| L | 2 | layers (smoke) |
| H | 4 | query heads |
| d_h | 16 | D / H |
| B | 2 | batch (math helpers) |

`PEDAGOGICAL_CTX = 256000` is **board-only**; smoke never allocates T = 256K.

### Scaled dot-product attention

```text
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_h) + M ) V

M         : causal mask (−∞ on future keys)
shapes    : hidden / attn_out [1, T, D]
            logits             [1, T, V]
```

### KV-cache bytes (Full Document §10)

Do **not** apply an extra trailing ×2 beyond precision bytes P_b:

```text
bytes = 2 * L * H_KV * d_h * T * B * P_b

yardstick: L=48, H_KV=8, d_h=128, P_b=2 (bf16)
1 user @ T=32768:
  2 * 48 * 8 * 128 * 32768 * 1 * 2

GQA/MQA: fewer KV heads than query heads
  reduction ≈ H_Q / H_KV
```

### Linear / delta sketches (teaching)

```text
Linear attn: replace softmax(Q K^T) with φ(Q), φ(K)
             associative memory view → O(T · d)

Delta / Gated DeltaNet: recurrent state update
             (no full T×T score matrix in the sketch)
```

Used in widgets and chronology cards — not production sims.

### Chronology ordering (not teaching order)

| Rank | Meaning |
| :--- | :--- |
| **Sr_No** | Sort by `year_sort` (timeline / Netlify wall) |
| **Te_No** | Pedagogical sequence (`teaching_order` in JSON) |
| **assignment_cover_order** | Coverage checklist only — **not** Te_No |

---

## 2. Hardware constraints & memory footprint

| Surface | Device | Footprint |
| :--- | :--- | :--- |
| Smoke pipeline (`run_demo.py`) | CPU default (`ASSIGN008_DEVICE`) | Tiny dense tables: V·D embeds + L attention blocks at T ≤ 64 |
| Yardstick KV formula | Account only | O(L · H_KV · d_h · T · B · P_b); no tensor of that size |
| V5 dense reference (V=131072, D=8096) | Account only | Never allocated in this repo |
| Static demos | Browser | HTML/CSS/JS + copied widgets under `dist/` |

Env toggles: `ASSIGN008_SEED`, `ASSIGN008_DEVICE`, `ASSIGN008_EMBED` (`dense` \| `chrono_stub`), `ASSIGN008_POSITION` (`rope` \| `none`), `ASSIGN008_USE_HF`, `ASSIGN008_RUN_WIDGET_CHECK`.

---

## 3. Empirical evaluation & benchmarks

| Check | Command / artifact | Expected |
| :--- | :--- | :--- |
| Smoke self-diagnosis | `uv run python run_demo.py` | All checks OK; `submission_artifacts/evidence.json` |
| Unit suite | `uv run pytest tests/ -q` | All green (math + chrono + demo_Asgn_08) |
| Teaching demo diagnostics | `uv run python tests/Self_Diagionistics_02.py` | Scorecard pass; `diagnostics_02/json/` (gitignored) |
| Chronology diagnostics | `uv run python tests/chrono_self_Diagnostics.py` | Scorecard pass; `diagnostics_chrono/json/` (gitignored) |
| Chronology cards | `chronology.json` | **18** mechanisms |
| Dist build | `uv run python tests/build_dist.py` | `dist/index.html` + demos + `demo_08/.../widgets/` |

Date bibliography: [`demo_chrono/CHRONOLOGY_SOURCES_BY_DATE.md`](demo_chrono/CHRONOLOGY_SOURCES_BY_DATE.md). Sr_No vs Te_No: [`demo_chrono/CHRONOLOGY_Compare.md`](demo_chrono/CHRONOLOGY_Compare.md).

---

## 4. Repository structure & file registry

```text
Assign_008/
├── web/index.html              # landing template (editable)
├── dist/                       # Netlify publish root (built snapshot)
│   ├── index.html
│   ├── demo_chrono/
│   ├── demo_Asgn_08/
│   └── demo_08/coach_demo/widgets/   # iframe targets for timeline cards
├── demo_chrono/                # source of truth — chronology wall
├── demo_Asgn_08/               # source of truth — teaching tour
├── demo_08/coach_demo/widgets/ # widget HTML used by iframes
├── run_demo.py                 # sole root CLI (smoke → evidence)
├── config.py                   # dims, yardstick, env toggles
├── src/                        # attention, KV math, pipeline, evidence
├── tests/
│   ├── build_dist.py           # recreate dist/ for deploy
│   ├── Self_Diagionistics_02.py
│   ├── chrono_self_Diagnostics.py
│   └── test_*.py
├── netlify.toml                # publish = "dist"
├── pyproject.toml
└── README.md
```


---

## 5. Execution guide & ablation suite

```bash
cd Assign_008
uv sync --native-tls

# Smoke + evidence
uv run python run_demo.py

# Tests
uv run pytest tests/ -q

```

Local preview: open `dist/index.html` (or source trees under `demo_chrono/`, `demo_Asgn_08/`).

### Chronology data refresh

```bash
uv run python demo_chrono/scripts/import_chrono_verify.py   # if CSV batch
uv run python demo_chrono/scripts/export_chrono_sources.py  # embed + source MD + Compare
uv run python tests/build_dist.py                          # refresh publish snapshot
```

### Assignment coverage (18 mechanisms)

| id | Mechanism |
| :--- | :--- |
| standard_attention | Scaled dot-product attention |
| absolute_learned_pe | Absolute learned position embeddings |
| sinusoidal_pe | Sinusoidal position encoding |
| mqa | Multi-Query Attention |
| sparse_topk | Sparse / top-k routing |
| sliding_window | Sliding-window (Longformer) |
| linear_attention | Linear attention |
| rope | RoPE |
| alibi | ALiBi |
| gqa | GQA |
| ntk_scaling | NTK-aware RoPE scaling |
| yarn | YaRN |
| attention_sinks | Attention sinks |
| mla | Multi-head Latent Attention |
| delta_rule | Delta rule |
| gated_deltanet | Gated DeltaNet |
| deepseek_compressed_sparse | DeepSeek compressed + sparse (NSA) |
| drope | DroPE |

===================================
