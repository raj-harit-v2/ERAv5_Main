# Session 7: REAL Fourier Alternative of Kronecker (Problem #4)

## Synopsis

Kronecker embeddings mark a frozen UTF-8 byte × position grid and project it.
Our Fourier alternative maps each Unicode codepoint to a deterministic sin/cos wave (phase uses character index), sums the waves, normalizes, and learns only a shared Linear(code_dim → d_model).

Ablation arms: Dense vs Kronecker vs Fourier, plus collision audits, per-stage corpus audits, and PNG figures. **Problem #4 only.**

**How to run:** `uv run python run_demo.py`  
**Coach:** [`https://inspiring-frangollo-a2948b.netlify.app/`](https://inspiring-frangollo-a2948b.netlify.app/)

---

## Codec math

Dense input size (reference accounting; smoke run uses `V=512`, `D=64`):

```text
|Θ_in| = V · D = 131072 × 8096 = 1_061_158_912
AdamW GB ≈ |Θ| · 16 / 2^30 ≈ 16.98 GB
```

**Kronecker** (`KroneckerByteEmbedding`):

```text
mark UTF-8 (byte_p, p) on a 256 × pos_dim grid
flatten → scale by 1/sqrt(L) → z-normalize → Linear(256·pos_dim → D)
```

**Fourier** (`FOURIER_CODE_DIM=512`, `FOURIER_N_FREQ=32`):

```text
ω(C)  = 0.5 + ((C · 2654435761) mod 8000) / 1000
f_k   = linspace(1, n_freq, code_dim/2)
φ(C,p)= [ sin(f_k · ω(C) · (p+1) · 0.1) , cos(...) ]_k
F     = znorm( L^{-1/2} · Σ_p φ(C_p, p) )
e     = W F + b ,   W ∈ R^{code_dim × d_model}
```

One token → one length-512 code; a sentence is a **sequence** of codes.

**HRR / VSA:** FFT circular convolution at fixed `HRR_DIM=2048` for bind/unbind checks. Verification layer — not the default LM embed path.

<img width="850" height="500" alt="anim_kronecker_vs_fourier_indic" src="https://github.com/user-attachments/assets/f572b8fa-b28e-4c37-8d3c-3ad5088bf95e" />

<img width="1400" height="650" alt="Fourier_store-encode" src="https://github.com/user-attachments/assets/27c9e8ed-6798-497a-bb9e-dbef2951f4ec" />

<img width="1400" height="861" alt="Apple_chk" src="https://github.com/user-attachments/assets/afb7136f-2f6e-4269-bb4f-8129f4299f50" />

---

## Collisions and evidence

Kronecker at `pos_dim=32` shares a 32-byte UTF-8 prefix window; long Indic strings that agree on that prefix collide. Fourier uses codepoints, so it does not inherit that byte-window failure mode.

Smoke ablation (`submission_artifacts/evidence.md`):

| Arm | val_ce_hi | val_ce_en | disc | note |
|-----|-----------|-----------|------|------|
| kronecker | 1.028 | **1.062** | 0.833 | stronger English CE on smoke |
| fourier | 0.762 | 4.845 | 0.917 | Indic-facing |
| fourier_canine | **0.759** | 5.078 | **1.000** | 0 collision groups vs Kronecker 1 @ pos32 |

Primary metrics for Problem #4: collisions, discrimination, and `val_ce_hi`. English CE on this smoke run is secondary.

| Path | Stored | Encode | Output \(D \times V\) |
|------|--------|--------|------------------------|
| Dense (ref.) | vocab rows | gather | logits head (separate) |
| Kronecker | Linear(\(256\cdot\mathrm{pos\_dim}\to D\)) | byte×pos → project | logits head |
| Fourier | Linear(\(\mathrm{code\_dim}\to D\)) (+ optional CANINE) | waves → sum → z-norm → project | logits head; invertibility is Problem #5 (VQ off) |

---

## Upgrade knobs

| Stage | Env | Default |
|-------|-----|---------|
| RoPE | `ASSIGN007_POSITION_POLICY=rope` | on |
| CANINE-like Fourier | `ASSIGN007_USE_CANINE=1` | on |
| VQ (Problem #5) | `ASSIGN007_ENABLE_VQ=0` | off |

---

## Artifacts

**Run outputs** (`submission_artifacts/`): `evidence.json`, `evidence.md`, `run.log`, `embedding_policy.json`, plus figures, stage audits, metrics, and diagnosis reports.

**Coach / README figures** (`demo_07/coach_demo/figures/` — present on disk):

---

Good to add later: `fourier_hrr_bind` (HRR-bind each char wave to a position vector, then sum). Bind/unbind already exists in the VSA layer; it is not in the default ablation.
