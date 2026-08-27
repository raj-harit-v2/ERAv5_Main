# Session 9 — Loss Functions & Output Heads

Observable loss harness, multi-token prediction (MTP), and chunked cross-entropy memory profile for ERA V5 Session 9 (Loss Functions & Output Heads).

**Deliverables:** [`notebooks/session09_harness.ipynb`](notebooks/session09_harness.ipynb) (Colab, top-to-bottom) · short write-up [`data/evaluation/session09_writeup.txt`](data/evaluation/session09_writeup.txt).

---

## 1. Core Architecture & Mathematical Foundations

**Engineering problem.** A language-model training step reduces to: forward pass → logits → shifted targets → masked cross-entropy → scalar loss. Silent bugs in the shift or mask can yield plausible loss curves; observability is mandatory.

**Standard harness (Part 1).**

```text
h = model(x)
z = W * h
L = CE( z[:, :-1] , x[:, 1:] )
```

| Tensor | Shape (smoke run) | Dimensions |
| :--- | :--- | :--- |
| `tokens` | `[B, T]` | batch, sequence length |
| `hidden` | `[B, T, D]` | batch, seq, model width |
| `logits` | `[B, T, V]` | batch, seq, vocabulary |
| `targets` | `[B, T-1]` | next-token ids after shift |
| `mask` | `[B, T-1]` | 1 = contribute, 0 = ignore |

**Shift verification.** Compare token *strings* `(input_t, target_t)` — not integer ids — to catch off-by-one errors.

**Padding & boundary.** Masked CE excludes pad tokens; packing two documents requires masking the cross-document boundary so the model is not scored on an invalid transition.

**Perplexity at init.** For an untrained model with uniform logits over vocabulary size V:

```text
L0  ≈ ln(V)
PPL0 = exp(L0) ≈ V
```

**Tied vs untied output head.** Tied: one projection matrix reused. Untied: separate head weights → approximately 2× head parameter count.

**MTP Part 2 (k=2).** Head 1 predicts t+1; Head 2 predicts t+2 from the same trunk hidden state. Report L1, L2, and L1+L2 separately.

**Chunked CE (Part 1 item 7 / Part 3).** Materialize logits in chunks of size C instead of full sequence length T. Peak activation memory scales as O(C·V) instead of O(T·V).

---

## 2. Hardware Constraints & Memory Footprint

**NanoLM smoke configuration.** Shakespeare word tokenizer, V = 195, local CPU proxy when CUDA unavailable (`memory_backend: cpu_proxy` in artifacts).

**Peak logits memory (Part 1 item 7).**

| Mode | Peak MiB | Notes |
| :--- | ---: | :--- |
| Full CE | 3.0469 | Materialize `[T × V]` logits |
| Chunked CE (C=1024) | 0.7617 | Peak at `[C × V]` |
| Ratio full / chunked | 4.0× | ≈ T/C for proxy estimate |

**Long-context implication (Part 3).** Full CE cost grows linearly with T; chunked CE holds peak VRAM near constant in T, enabling 64K+ context on fixed GPU memory at the cost of extra forward passes.

**Optimizer state (reference).** AdamW stores ~16 bytes per trainable parameter (fp32 master + momentum + variance). Not exercised at scale in this smoke run; documented for operational planning on full models.

---

## Empirical Evaluation & Benchmarks

Colab artifacts: [`seven_numbers.json`](data/evaluation/seven_numbers.json), [`part2_losses.json`](data/evaluation/part2_losses.json).

### Part 1 — Seven observables

| # | Observable | Value |
| :--- | :--- | ---: |
| 1 | `shapes_ok` | True |
| 2 | `shift_strings_ok` | True |
| 3 | Pad count before → after | 28 → 17 |
| 4 | Boundary contrib / sum (Δ) | 13→12 / 68.549→63.276 (Δ≈−5.273) |
| 5 | PPL₀ / loss₀ (V = 195) | 195.00 / 5.2730 |
| 6 | Tied / untied params | 12480 / 24960 |
| 7 | Peak full / chunked / ratio | 3.0469 / 0.7617 / 4.0× |

Boundary: masked cross-doc next-token; CE **sum** drops; mean stays ~ln(V) at init.

### Part 2 — MTP dual heads (40-step Colab)

| Metric | Value |
| :--- | ---: |
| L1 final (t+1) | **4.0755** |
| L2 final (t+2) | **4.1171** |
| Sum | **8.1927** |

Both heads start near `ln(V) ≈ 5.273`. After training, L2 remains slightly above L1: t+2 is a harder, longer-range target from the same trunk state.

### Part 3 — Curriculum demonstration

Long-context evidence under **Curriculum_Required_stats**:

| Figure | Path |
| :--- | :--- |
| Peak VRAM | [`reports/Curriculum_Required_stats_01_peak_vram.png`](reports/Curriculum_Required_stats_01_peak_vram.png) |
| Chunk frontier | [`reports/Curriculum_Required_stats_02_chunk_frontier.png`](reports/Curriculum_Required_stats_02_chunk_frontier.png) |
| SwiGLU | [`reports/Curriculum_Required_stats_03_swiglu.png`](reports/Curriculum_Required_stats_03_swiglu.png) |

**SwiGLU vs classic FFN (curriculum).** Widget 2: [`widgets/s9_widget_2_swiglu.html`](widgets/s9_widget_2_swiglu.html).

```text
Classic:  ReLU(x W₁) W₂
SwiGLU:   (Swish(x W₁) ⊗ x V) W₂
```

Toy contrast (same `x = [0.60, −0.40]`, D=2) — intermediate before W₂:

| Path | Before W₂ | Point |
| :--- | :--- | :--- |
| Classic ReLU | `[0.40, 0.00]` | hard-zeros the negative dim |
| SwiGLU ⊗ | `[0.13, 0.03]` | gate scales value (both dims survive) |

**NanoLM still uses GELU FFN** — SwiGLU here is Llama-class curriculum context, not the smoke harness.

**Takeaway:** full CE scales with T; chunked CE flattens peak VRAM so 64K+ context is feasible. SwiGLU figure is architecture context.

<img width="1800" height="1050" alt="SwiGLU Interactive" src="https://github.com/user-attachments/assets/e824b6ad-87e3-45b6-bc4e-38558c67cea1" />


---

## Execution Guide & Ablation Suite

```text
cd Assign_009_Init
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Local Jupyter only (skip on Colab):
# pip install -r requirements-jupyter.txt

python -m src.pipeline.run_all
```

**Notebook:** `jupyter notebook notebooks\session09_harness.ipynb`

**Colab:** zip project (include `src/`, `data/`, `tests/`, `web/`, `widgets/`, `notebooks/`, `requirements.txt`; omit `.venv/`) → upload → `pip install -q -r requirements.txt` → run cells top-to-bottom.

**Widgets:** `python tests/build_dist.py` then `python -m src.pipeline.serve_dist` → http://127.0.0.1:8765/

**Design locks:** NanoLM + Shakespeare word tokenizer · hand-written chunked CE · MTP k=2 · CUDA peak memory when available, else labeled CPU proxy.
