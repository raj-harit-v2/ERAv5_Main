"""Session 9 harness tests — fail-closed gates."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.chunked_ce import ChunkedCrossEntropy, full_cross_entropy, shift_logits_and_targets
from src.llm.nano_lm import NanoLM
from src.llm.tokenizer import WordTokenizer
from src.pipeline.harness import build_loss_mask, load_tokenizer, pad_batch, run_harness
from src.utils.decode_strings import print_shift_table, verify_shift_strings


@pytest.fixture(scope="module")
def tok() -> WordTokenizer:
    return load_tokenizer()


def test_shift_string_pairs(tok: WordTokenizer):
    ids = tok.encode("First Citizen: Speak.", add_bos=True, add_eos=True)
    assert verify_shift_strings(tok, ids)
    pairs = print_shift_table(tok, ids, max_rows=5)
    assert len(pairs) >= 1
    # Consecutive in decoded stream
    decoded = tok.decode(ids)
    assert pairs[0] == (decoded[0], decoded[1])


def test_pad_count_decreases(tok: WordTokenizer):
    short = tok.encode("Come", add_bos=True, add_eos=True)
    long = tok.encode("First Citizen: Speak speak.", add_bos=True, add_eos=True)
    batch = pad_batch([short, long], tok.pad_id)
    targets = batch[:, 1:]
    before = int(torch.ones_like(targets).sum().item())
    mask = build_loss_mask(batch, tok.pad_id)
    after = int(mask.sum().item())
    assert after < before


def test_chunked_ce_parity():
    torch.manual_seed(0)
    n, v = 200, 50
    logits = torch.randn(n, v)
    targets = torch.randint(0, v, (n,))
    full = full_cross_entropy(logits, targets)
    chunked = ChunkedCrossEntropy(chunk_size=32)(logits, targets)
    assert abs(float(full - chunked)) < 1e-5


def test_untrained_ppl_near_v(tok: WordTokenizer):
    torch.manual_seed(0)
    model = NanoLM(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, max_seq=64)
    model.eval()
    ids = tok.encode("Let us kill him and have corn.", add_bos=True, add_eos=True)
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)
    sl, st = shift_logits_and_targets(logits, tokens)
    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1))
    ppl = math.exp(float(loss))
    v = tok.vocab_size
    # Within 35% relative of V on log scale via loss≈ln(V)
    assert abs(float(loss) - math.log(v)) / math.log(v) < 0.35
    assert abs(ppl - v) / v < 0.5


def test_harness_writes_seven_numbers(tmp_path, monkeypatch, tok):
    # Run harness; should write evaluation JSON
    out = run_harness(device="cpu", seed=0)
    assert out["shift_strings_ok"] is True
    assert out["pad_count_after"] < out["pad_count_before"]
    assert out["ratio_full_over_chunked"] > 0
    path = ROOT / "data" / "evaluation" / "seven_numbers.json"
    assert path.exists()
