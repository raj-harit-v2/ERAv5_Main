"""Smoke-scale configuration for Assign_007 Problem #4 (Fourier)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

PROBLEM_ID = 4
SCHEMA_VERSION = "a07.4.2"
SEED = int(os.getenv("ASSIGN007_SEED", "42"))
USE_HF = os.getenv("ASSIGN007_USE_HF", "0") == "1"
DEVICE = os.getenv("ASSIGN007_DEVICE", "cpu")

# Position: RoPE is default (industry upgrade #1)
POSITION_POLICY = os.getenv("ASSIGN007_POSITION_POLICY", "rope")  # rope | absolute

# Fourier path: keep baseline + CANINE upgrade in ablation
USE_FOURIER_CANINE = os.getenv("ASSIGN007_USE_CANINE", "1") == "1"
CANINE_STRIDE = int(os.getenv("ASSIGN007_CANINE_STRIDE", "2"))

# Problem #5 VQ — gated OFF by default ("only then")
ENABLE_VQ_PROBLEM5 = os.getenv("ASSIGN007_ENABLE_VQ", "0") == "1"
VQ_NUM_CODES = int(os.getenv("ASSIGN007_VQ_CODES", "256"))
VQ_LOSS_WEIGHT = float(os.getenv("ASSIGN007_VQ_LOSS_WEIGHT", "0.25"))

# VSA / HRR verification layer (Problem #4 deep tests)
HRR_DIM = int(os.getenv("ASSIGN007_HRR_DIM", "2048"))
VSA_NOISE_FRAC = float(os.getenv("ASSIGN007_VSA_NOISE_FRAC", "0.30"))
VSA_CAPACITY_TARGET_COS = float(os.getenv("ASSIGN007_VSA_CAPACITY_COS", "0.70"))
RUN_VSA_DEEP = os.getenv("ASSIGN007_RUN_VSA_DEEP", "1") == "1"

ARTIFACTS_DIR = ROOT / os.getenv("ASSIGN007_ARTIFACTS_DIR", "submission_artifacts")
FIGURES_DIR = ARTIFACTS_DIR / "figures"
STAGE_AUDITS_DIR = ARTIFACTS_DIR / "stage_audits"
LEDGERS_DIR = ARTIFACTS_DIR / "ledgers"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

VOCAB_SIZE = 512
SEQ_LEN = 64
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
BATCH_SIZE = 8
STEPS_PER_STAGE = 24
PAD_ID = 0

POS_DIM_KRON = 8  # smoke training capacity; collision audit still sweeps 32/48/64
POS_DIM_SWEEP = (32, 48, 64)
FOURIER_CODE_DIM = 512  # ~ matches Kronecker smoke code_dim 256*8
FOURIER_N_FREQ = 32
MAX_CHARS_FOURIER = 64
# Soft EN gate: allow larger smoke gap when Fourier wins an Indic metric
EN_CE_REGRESSION_TOL = 1.5

STAGES = ("seed", "general", "indic_focus", "anneal")
STAGE_INDIC_FLOOR = {
    "seed": 0.15,
    "general": 0.20,
    "indic_focus": 0.45,
    "anneal": 0.25,
}

