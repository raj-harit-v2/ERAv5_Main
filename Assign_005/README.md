# ERA V5 Assignment 05 — Data Mixture & Curriculum Specification (V5 / 120B)

Defendable mixture-and-curriculum plan for a **120B** V5 pretraining run, sized against Session-4 supply and exercised on a smoke-scale pipeline (~14k tokens). Inventory honesty. Live proof: open `dist/index.html` (Verifier Panel).
https://grand-hummingbird-108f04.netlify.app/

---

## 1. Capability budgeting (token shares, not sample counts)

**Illustrative main-run scale:** 2.0T tokens (hypothesis pending cohort compute). Zero-sum: every +1% to one lane is −1% elsewhere.

| Lane | Share | Tokens @ 2.0T | Benchmarks purchased | Supply posture |
|------|-------|---------------|----------------------|----------------|
| General Web | 38% | 760B | World knowledge, language priors | Unique abundant (selective FineWeb/C4-class) |
| Code | 22% | 440B | HumanEval-class, SWE-adjacent edit | Unique strong; mild late repeat OK |
| STEM | 12% | 240B | Math/science | Unique moderate |
| Reasoning | 8% | 160B | Multi-step CoT; effort dial | Unique + synthetic traces |
| Long-context | 5% | 100B | Needle / long-doc QA | Scarce at 32K–128K; pack carefully |
| Indic | 10% | 200B | Native fluency / IndicQA | **Verified scarce → synth/translate** |
| Agentic | 5% | 100B | SWE-bench, multi-step tools | **Mostly synthesize** |

### Indic tier split (no single headline)

| Tier | % of Indic | % of total run | Tokens @ 2.0T |
|------|------------|----------------|---------------|
| Verified native | 15% | 1.5% | 30B |
| Unverified native | 25% | 2.5% | 50B |
| Translated | 25% | 2.5% | 50B |
| Synthetic | 35% | 3.5% | **70B (must generate)** |

**Asgn4 measured supply (honesty):** Path-B admits **49,852** shards / **24,744,178** tokens — ~0.012% of the 200B Indic demand. Wishful accounting forbidden; synthetic/translated fill is mandatory. 

### Slot → inventory mapping

| Slot | Datasets / shapes | Unique / Repeat / Synthesize |
|------|-------------------|------------------------------|
| Agentic | SWE-bench-format + multi-step tool trajectories; loss on plan/tool/final only | Mostly **Synthesize** |
| Reasoning | Math/code CoT; short→ultra length bands | Mix Unique + **Synthesize** |
| Long-context | Packed books/docs/repos at late ctx | Unique scarce; limited Repeat |
| Indic | Asgn4 ADMITTED (hi/te/sa); Sangraha **samples only**; translation + synth | Unique tiny; **Synthesize** dominant |
| Code | The Stack / StarCoderData-style (license filter) | Unique |
| STEM | OpenWebMath / arXiv-class | Unique + light Repeat |
| Web | FineWeb/C4-class **selective** subsets | Unique |

---

## 2. OPUS & protected floors

**OPUS** scores candidates in optimizer-induced update space (ghost + CountSketch) onto a stable proxy, then Boltzmann-samples. Smoke keep-fraction **40%**.

| Control | Spec |
|---------|------|
| Keep-fraction | 40% |
| Effective-token multiplier | **6.0×** (V4 calibration) |
| Compute overhead | **4.7%** |
| Proxy mixes | `english_heavy` (default) / `balanced` |
| Always-On floor | **12%** total — Indic 6% + Agentic 3% + Reasoning 3% |

**Why Always-On:** English-heavy proxies undervalue Indic/agentic. Smoke replay: Always-On **OFF** → Indic trained **0%**; **ON** restores floor fills. Verifier → OPUS (Next / Auto-run iterations).

---

## 3. Staged pretraining & context scaling

> **Recheck — executed smoke / short-proxy ctx :**
> ```
> seed: 64 · general: 64 · reasoning: 96 · long_context: 128 · anneal: 96
> ```


**Full-scale plan (120B):** Seed 4K → General 4K–8K → Reasoning 8K → Long-context **32K→128K** → Anneal 8K–32K.

Smoke proves the **stage machine** at tiny lengths; full-scale K targets remain the V5 plan (long-context still 5% @ 2.0T). Cite RunPod as `short_{1,3,6,12}b_synth_proxy` 

**Transitions:** never hard-cut. Smoke uses **50/50 warmup blend** for 2 steps at each stage boundary to avoid V4-style ~150× grad-norm spikes.

---

## 4. Difficulty, reasoning length, loss map

### Difficulty B0–B5 (one example each)

| Band | Example |
|------|---------|
| B0 | राम स्कूल जाता है। / The cat sat on the mat. |
| B1 | Two-sentence Hindi story + grade-school arithmetic 12+7. |
| B2 | Quadratic word problem; short Python loop. |
| B3 | Prove a discrete-math claim; implement BFS. |
| B4 | Derive attention complexity; debug a race condition. |
| B5 | Summarize a methods section; propose an ablation. |

### Reasoning lengths (same problem, four depths)

| Length | Example (91 prime?) |
|--------|---------------------|
| short | 91=7×13, not prime. |
| medium | Check primes ≤√91; find 7×13; conclude composite. |
| long | Enumerate 2,3,5,7 with self-check 7×13=91. |
| ultra | Case analysis + tool verify + recover from wrong branch. |

A hard short trace ≠ an easy ultra trace (effort dial).

### Loss map · Agentic (what we train)

**Lesson:** the model is never trained to imitate the environment. Only its own tokens are green; issue text, tool returns, and observations stay grey. Pass/fail is **violet / reward-only** — later RL, **not** Asgn5 token loss.

| Color | Role |
|-------|------|
| green | Supervised — plan / tool call / final patch  |
| grey | Masked context — issue, repo slice, user, obs |
| violet | Reward-only — hidden-test pass/fail; no token imitation (post-pretrain) |

**Sample (SWE-bench-shaped training format for Agentic 5%):**

```
[grey]  issue: TypeError when Config.foo() gets a list
[grey]  obs:   pytest failed on test_foo_list
[green] plan:  foo() calls s.strip(); guard list→str
[green] tool:  apply_patch …
[green] final: @@ def foo(self, s): …
[violet] REWARD +1  (verifier; not next-token loss)
```

Live mask dump: Verifier → Curriculum → Agentic loss map.

---

## 5. Anneal reserve

Hold **~4%** Tier-A unique docs (Indic / agentic / reasoning flagged `tier_a`) out of ordinary OPUS sampling for the low-LR cooldown. Anneal preset upweights those lanes (web → 5%). Spend early → final benchmark lift is gone.

---

## 6. Recoverability

Smoke records `dataloader_state_hash` / RNG seed on each OPUS selection. Full-scale: deterministic shuffle + pause-resume (Session 6). Asgn4 content SHA-256 / shard_id is the upstream precursor.

---

## 7. Proxy hypothesis (testable)

**Hypothesis:** Main mix Web 38 / Code 22 / STEM 12 / Reasoning 8 / Long-ctx 5 / Indic 10 / Agentic 5, Always-On 12% (6/3/3), 4% Tier-A anneal reserve, beats a web-heavy unprotected baseline on SWE-bench-lite, IndicQA-proxy, and medium+ reasoning at **3B**, without large regression on general web val loss.

| Scale | Confirm if | Refute if |
|-------|------------|-----------|
| 1B | Protected lanes move vs web-heavy baseline | Only web improves |
| 3B | SWE-bench-lite / Indic / CoT rank-order matches bets | Agentic/Indic ablations flat |

**Evidence (short):** `short_{1,3,6,12}b_synth_proxy` (\~40 steps/arm, not Chinchilla) all `smoke_confirm=true` with positive mean protected Δ (1B≈1.08, 3B≈1.27, 6B≈1.05, 12B≈1.27). Also: smoke 14,336 tokens; Always-On OFF→ON Indic 0%→\~9%; diagnostics pass. Lab Radar + `processed/proxy_*_short_metrics.json` for charts/JSON.

---

## How to run

```bash
uv sync --native-tls
python smoke_test.py
python self_diagnostics.py
# open dist/index.html
```

**Submission:** GitHub repo README for this assignment folder 