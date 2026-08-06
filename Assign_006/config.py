"""Assignment 06 — Training Data Execution System configuration."""
from __future__ import annotations

from pathlib import Path

ROOT_06 = Path(__file__).resolve().parent
ASSIGN_005_ROOT = Path(r"D:\A1_School_ai_25\ERA_V5\Assign_005")

ARTIFACTS_DIR = ROOT_06 / "submission_artifacts"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
LEDGERS_DIR = ARTIFACTS_DIR / "ledgers"
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
SHARDS_DIR = ARTIFACTS_DIR / "shards"
STAGE_AUDITS_DIR = ARTIFACTS_DIR / "stage_audits"
PACKED_REPORTS_DIR = ARTIFACTS_DIR / "packed_batch_reports"
CACHE_DIR = ROOT_06 / "cache"

SEED = 42

# Capability lanes (from Assign_005)
MAIN_MIXTURE: dict[str, float] = {
    "web": 0.38,
    "code": 0.22,
    "stem": 0.12,
    "reasoning": 0.08,
    "long_context": 0.05,
    "indic": 0.10,
    "agentic": 0.05,
}

INDIC_TIERS: dict[str, float] = {
    "verified": 0.15,
    "unverified": 0.25,
    "translated": 0.25,
    "synthetic": 0.35,
}

ANNEAL_MIXTURE: dict[str, float] = {
    "agentic": 0.30,
    "indic": 0.25,
    "reasoning": 0.25,
    "code": 0.15,
    "web": 0.05,
}

ALWAYS_ON_FLOOR = 0.12
ALWAYS_ON_SPLIT: dict[str, float] = {
    "indic": 0.06,
    "agentic": 0.03,
    "reasoning": 0.03,
}

OPUS_KEEP_FRACTION = 0.40
OPUS_TEMPERATURE = 0.7
ANNEAL_RESERVE_FRACTION = 0.04
WARMUP_BLEND_STEPS = 2
COUNTSKETCH_DIM = 32

STAGES = ("seed", "general", "reasoning", "long_context", "anneal")

STAGE_CTX: dict[str, int] = {
    "seed": 64,
    "general": 64,
    "reasoning": 96,
    "long_context": 128,
    "anneal": 96,
}

STAGE_MIXTURE: dict[str, dict[str, float]] = {
    "seed": {
        "web": 0.55,
        "indic": 0.15,
        "code": 0.10,
        "stem": 0.08,
        "reasoning": 0.05,
        "long_context": 0.02,
        "agentic": 0.05,
    },
    "general": {
        "web": 0.45,
        "code": 0.18,
        "stem": 0.12,
        "indic": 0.12,
        "reasoning": 0.05,
        "long_context": 0.03,
        "agentic": 0.05,
    },
    "reasoning": {
        "web": 0.22,
        "code": 0.28,
        "stem": 0.18,
        "reasoning": 0.14,
        "indic": 0.10,
        "long_context": 0.03,
        "agentic": 0.05,
    },
    "long_context": {
        "web": 0.20,
        "code": 0.22,
        "stem": 0.10,
        "reasoning": 0.10,
        "long_context": 0.18,
        "indic": 0.10,
        "agentic": 0.10,
    },
    "anneal": dict(ANNEAL_MIXTURE),
}

STAGE_STEPS: dict[str, int] = {
    "seed": 6,
    "general": 6,
    "reasoning": 6,
    "long_context": 4,
    "anneal": 4,
}

CAPABILITY_LANES = list(MAIN_MIXTURE.keys())
PROTECTED_LANES = frozenset(ALWAYS_ON_SPLIT.keys())
TIER_A_LANES = frozenset({"indic", "agentic", "reasoning"})

# Tokenizer — prefer HF tiny-gpt2; offline fallback uses local hash tokenizer
TOKENIZER_HF_NAME = "sshleifer/tiny-gpt2"
USE_HF_TOKENIZER = False  # False = hermetic local tokenizer (no internet)
LOCAL_VOCAB_SIZE = 512
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 50256  # GPT-2 convention; remapped into local vocab when offline
LOCAL_EOS_ID = 511

# Shard / batch smoke scale
SHARD_TARGET_TOKENS = 4096
SEQUENCE_LENGTH = 128
GLOBAL_BATCH_TOKENS = 1024
MICROBATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
N_GPUS_SIMULATED = 1

PPL_THRESHOLD_SKIP = 1.2
CKPT_EVERY_N_STEPS = 10
CRASH_AT_STEP = 12
POST_CRASH_STEPS = 8
REPLAY_STEPS = 5
FORK_FROM_STEP = 10

OPUS_DECISION_STATES = ("accepted", "rejected", "deferred", "floor_override")
PPL_FULL_TRACE_SHARDS = 3
LEDGER_BACKEND = "jsonl"

EVAL_LANE_TAG = "eval"
TEST_LANE_TAG = "test"

ADMITTED_LICENSES = frozenset({"cc0", "mit", "apache2", "open_rail"})

# Tiny model (smoke)
SMOKE_VOCAB_SIZE = LOCAL_VOCAB_SIZE
SMOKE_D_MODEL = 64
SMOKE_N_LAYERS = 2
SMOKE_N_HEADS = 4
SMOKE_LR = 3e-3

DOCS_PER_LANE = 16
EVAL_DOCS = 5
TEST_DOCS = 5

SCHEMA_VERSION = "a06.1.0"
