# ERA V5 — Assignment 06: Training Data Execution System

Small but complete **data execution system**. Scale is intentionally tiny. Correctness, reproducibility, and auditability are the goals.

GitHub layout (same pattern as Assign_005): `All_ERAV5/Assign_006/`.

The **runtime pipeline** :

```
Documents
run_demo.py
├── src/corpus.py
├── src/shard_builder.py
│     └── submission_artifacts/shards/*.npy
├── src/shard_manifest.py
│     └── submission_artifacts/manifests/
├── src/mixture_compiler.py
│     └── submission_artifacts/mixture_schedule.json
├── src/packing.py
│     └── submission_artifacts/packed_batch_reports/
├── src/trainer_extended.py          # batches + training loop
├── src/tiny_model.py                # training model
├── src/ledger.py
│     ├── submission_artifacts/ledgers/consumption_ledger.jsonl
│     └── submission_artifacts/ledgers/learning_ledger.jsonl
├── src/checkpoint_manager.py
│     └── submission_artifacts/checkpoints/
├── src/trainer_extended.py          # crash (SimulatedCrash)
├── src/replay_engine.py             # resume
├── src/replay_engine.py             # replay
├── src/replay_engine.py             # fork
└── src/evidence_builder.py          # audit
      ├── submission_artifacts/evidence.json
      └── submission_artifacts/evidence.md
```

Supporting gates on the same path: `src/tokenizer_wrapper.py`, `src/eval_firewall.py`, `src/opus_extended.py`. Proven by regenerating `submission_artifacts/` via `uv run python run_demo.py` (see Module map + evidence PASS).

## Quick start

```bash
cd Assign_006
uv sync --native-tls
uv run python run_demo.py
uv run pytest tests/ -q
```



## Data provenance

Training data is **not** shipped as a downloadable corpus and is **not** read from `cache/`. On every demo run, `[src/corpus.py](src/corpus.py)` synthesizes a multi-lane document set in process (`SEED=42` in `config.py`). The hermetic local tokenizer (`USE_HF_TOKENIZER=False`) needs no Hugging Face Hub access and no API keys. Cloning the repo and running the one command regenerates shards, manifests, ledgers, and evidence identically for the coach.

## Architecture



### High-level architecture

<img width="1680" height="1180" alt="High-level_arch" src="https://github.com/user-attachments/assets/b901b3c9-3f1f-4a85-bd37-22974af87890" />

The system turns synthetic multi-lane documents into immutable tokenized `.npy` shards bound to 14-field manifests (tokenizer hash, content hash, license, eval-overlap gates). A curriculum compiler emits per-step mixture quotas with Always-On floors; OPUS records accept / reject / defer / floor_override decisions before packing. Five packing policies produce loss masks, attention masks, and position IDs for microbatches. A tiny causal LM trains while appending dual append-only JSONL ledgers (consumption + learning) and tiered PPL traces. Checkpoints store model/optim/sched/RNG state plus `ledger_offset` and `expected_next_batch_id`. The demo deliberately crashes, resumes the exact next batch, replays an earlier interval with matching hashes/spans, forks a branch, and writes a generated (not hardcoded) evidence bundle under `submission_artifacts/`.

## Smoke-scale configuration


| Parameter                     | Value                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| Docs per capability lane      | 16 (+ 5 eval / 5 test holdouts)                                                                        |
| Train shards                  | ~28 (4 × 7 lanes)                                                                                      |
| Sequence length               | 128                                                                                                    |
| Microbatch size               | 2 sequences (~256 tokens/step before accum)                                                            |
| Grad accumulation (simulated) | 4 → design global batch 1024 tokens                                                                    |
| Curriculum steps              | ~26 across seed → general → reasoning → long_context → anneal                                          |
| Tiny LM                       | `d_model=64`, 2 layers, 4 heads, vocab 512                                                             |
| Crash / replay                | Crash before step 12; replay first 5 consumption events                                                |
| Packing                       | Utilization instrumented in `performance.json` (typically ~0.8 class; pad waste expected and measured) |




## Module map


| Module                                       | Role                                                      |
| -------------------------------------------- | --------------------------------------------------------- |
| `src/corpus.py`                              | Synthetic multi-lane corpus (no dataset downloads)        |
| `src/tokenizer_wrapper.py`                   | Frozen tokenizer + SHA-256 (hermetic local default)       |
| `src/shard_builder.py` / `shard_manifest.py` | Immutable `.npy` shards + manifests + admission           |
| `src/eval_firewall.py`                       | Content-hash + canary never-train registry                |
| `src/packing.py`                             | Five packing policies + loss / attention / position masks |
| `src/mixture_compiler.py`                    | Stage schedule, floors, warmup blend                      |
| `src/opus_extended.py`                       | Accept / reject / defer / floor_override audit trail      |
| `src/ledger.py`                              | Append-only consumption + learning JSONL                  |
| `src/perplexity_tracer.py`                   | Tiered per-token / sample PPL traces                      |
| `src/checkpoint_manager.py`                  | Model + optim + sched + RNG + `ledger_offset`             |
| `src/replay_engine.py`                       | Resume / replay / fork                                    |
| `src/trainer_extended.py`                    | Train loop with crash simulation                          |
| `src/evidence_builder.py`                    | Generated evidence bundle                                 |
| `deep_self_diagnosis.py`                     | Extra local diagnostics (PH0–PH15)                        |




## Design decisions

- **Hermetic by default:** no internet, no `dotenv`, no API keys for the graded path (`.env` is gitignored if present).
- **EOS-only** (no BOS), pad id `0`.
- **JSONL ledgers** with fsync on write.
- **Crash** at `CRASH_AT_STEP` before serving that batch; checkpoint stores `expected_next_batch_id`.
- **Four OPUS states** and Always-On floors are first-class design, not optional logging.



## Generated artifacts

`uv run python run_demo.py` regenerates:

```
submission_artifacts/
  run.log                 # Execution Log (required)
  evidence.json           # machine-readable evidence bundle
  evidence.md             # human-readable evidence summary
  manifests/
  ledgers/
  checkpoints/
  performance.json
```

Supporting outputs (shards, packed reports, stage audits, diagnosis) may also appear under the same tree. 

## Evidence (latest demo)

Generated (not hardcoded) by `uv run python run_demo.py`. Full bundle: `[submission_artifacts/evidence.md](submission_artifacts/evidence.md)` · `[submission_artifacts/evidence.json](submission_artifacts/evidence.json)` · `[submission_artifacts/run.log](submission_artifacts/run.log)`.

Run ID: `run_29f41fa697` · Overall: **PASS**


| Requirement         | Result | Evidence path                                |
| ------------------- | ------ | -------------------------------------------- |
| Tokenizer integrity | PASS   | `manifests/`                                 |
| Evaluation firewall | PASS   | `ledgers/firewall.json`                      |
| Packing correctness | PASS   | `packed_batch_reports/` · consumption ledger |
| Mixture compliance  | PASS   | `ledgers/consumption_ledger.jsonl`           |
| OPUS audit trail    | PASS   | `ledgers/opus_ledger.jsonl`                  |
| Crash recovery      | PASS   | `checkpoints/`                               |
| Replay              | PASS   | `ledgers/consumption_ledger.jsonl`           |
| Learning trace      | PASS   | `ledgers/learning_ledger.jsonl`              |
| Throughput          | PASS   | `performance.json`                           |




### Details

- **Tokenizer integrity:** `n_manifests=28`; hashes_ok
- **Evaluation firewall:** `blocked_events=1` (eval shard blocked)
- **Packing correctness:** `max_utilization=1.000`
- **Mixture compliance:** actual shares web≈0.307, code≈0.375, indic≈0.084, long_context≈0.088, reasoning≈0.025, stem≈0.096, agentic≈0.025; `max_abs_delta≈0.123`
- **OPUS audit trail:** statuses `accepted`, `deferred`, `floor_override`, `rejected`
- **Crash recovery:** expected next batch `run_29f41fa697:12:0` matched actual on resume
- **Replay:** compared 5 steps; original and replay batch hashes identical (`409cbabc529b…`)
- **Learning trace:** `n_learning_events=26` (loss linked to source documents)
- **Throughput:** `useful_tokens_per_second≈4043.10`

Re-running the demo regenerates these numbers; the nine requirement rows must still be PASS.

**How to re-check PASS:** open `submission_artifacts/evidence.md` (Overall: **PASS**) or `evidence.json` (`"overall_pass": true`), or re-run `uv run python run_demo.py` and confirm the console line `Overall evidence: PASS`.

## Reviewer checklist

1. Run `uv run python run_demo.py` — regenerates `submission_artifacts/`.
2. Verify `evidence.json` / `evidence.md` / `run.log` against manifests and ledgers.
3. Confirm evidence is computed from files (no hardcoded PASS values).
4. Optional: `uv run pytest tests/ -q`.

