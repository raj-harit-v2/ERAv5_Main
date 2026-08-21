"""Session 8 Phase 1 smoke config.

Assumption A1: PEDAGOGICAL_CTX=256K is for boards only, not local training.
Assumption A2: smoke tensors stay tiny (V=512, D=64, T<=64).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "submission_artifacts"
DEMO_FIGURES = ROOT / "demo_08" / "coach_demo" / "figures"

SCHEMA_VERSION = "a08.0.1"
VOCAB_SIZE = 512
D_MODEL = 64
SEQ_LEN = 64
N_LAYERS = 2
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
BATCH_SIZE = 2
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Boards only — do not allocate tensors of this length locally
PEDAGOGICAL_CTX = 256_000

# Full Document §10 yardstick
YARDSTICK_L = 48
YARDSTICK_H_KV = 8
YARDSTICK_D_HEAD = 128
P_BYTES = 2  # bf16

# V5 reference dense table — account only, never allocate
V5_REF_V = 131_072
V5_REF_D = 8_096

SEED = int(os.getenv("ASSIGN008_SEED", "42"))
DEVICE = os.getenv("ASSIGN008_DEVICE", "cpu")
USE_HF = os.getenv("ASSIGN008_USE_HF", "0") == "1"
EMBED_MODE = os.getenv("ASSIGN008_EMBED", "dense")  # dense | chrono_stub
POSITION_POLICY = os.getenv("ASSIGN008_POSITION", "rope")  # rope | none
RUN_WIDGET_CHECK = os.getenv("ASSIGN008_RUN_WIDGET_CHECK", "0") == "1"

EXAMPLE_SENTENCE = "The cat sat on the mat"
