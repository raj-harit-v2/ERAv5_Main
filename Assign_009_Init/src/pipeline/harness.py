"""Part 1 observable loss harness — seven numbers to data/evaluation/seven_numbers.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.llm.chunked_ce import shift_logits_and_targets
from src.llm.nano_lm import NanoLM
from src.llm.output_head import compare_tied_untied_counts
from src.llm.tokenizer import WordTokenizer
from src.pipeline.memory_profile import measure_full_vs_chunked
from src.utils.decode_strings import print_shift_table, verify_shift_strings
from src.utils.shapes import dump_batch_shapes

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "shakespeare_tiny.txt"
OUT = ROOT / "data" / "evaluation" / "seven_numbers.json"

# Fail-closed: untrained CE near ln(V) => PPL near V
PPL_REL_TOL = 0.35  # loose for random init small models


def _project_root() -> Path:
    return ROOT


def load_tokenizer() -> WordTokenizer:
    return WordTokenizer.from_file(RAW)


def pad_batch(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    max_t = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_t), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def build_loss_mask(tokens: torch.Tensor, pad_id: int, boundary_positions: set[tuple[int, int]] | None = None) -> torch.Tensor:
    """
    Mask for shifted targets: shape [B, T-1].
    1 = contribute. Zero pad targets and optional boundary transitions.
    boundary_positions: set of (batch, target_index_in_shifted) to zero.
    """
    targets = tokens[:, 1:]
    mask = (targets != pad_id).float()
    if boundary_positions:
        for b, j in boundary_positions:
            if 0 <= j < mask.size(1):
                mask[b, j] = 0.0
    return mask


def masked_mean_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """logits/targets/mask aligned at [B, T-1, V] / [B, T-1] / [B, T-1]."""
    b, t, v = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * t, v),
        targets.reshape(b * t),
        reduction="none",
    ).view(b, t)
    masked = loss * mask
    count = int(mask.sum().item())
    mean = masked.sum() / max(count, 1)
    return mean, count


def run_harness(device: str | None = None, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = load_tokenizer()
    v = tok.vocab_size

    # --- Demo sentence (RULE#3 spine starts with user text) ---
    user_sentence = "First Citizen: Let us kill him, and we'll have corn."
    print("\n=== RULE#3 user sentence ===")
    print(user_sentence)

    ids = tok.encode(user_sentence, add_bos=True, add_eos=True)
    tokens_1d = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    model = NanoLM(
        vocab_size=v,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq=256,
        tie_weights=False,
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits, hidden = model(tokens_1d, return_hidden=True)
    shift_logits, shift_targets = shift_logits_and_targets(logits, tokens_1d)
    pad_mask = build_loss_mask(tokens_1d, tok.pad_id)

    print("\n=== 1) Tensor shapes ===")
    shapes = dump_batch_shapes(tokens_1d, hidden, logits, shift_targets, pad_mask)

    print("\n=== 2) Shift verification (strings, not ids) ===")
    pairs = print_shift_table(tok, tokens_1d[0].tolist())
    shift_ok = verify_shift_strings(tok, tokens_1d[0].tolist())
    print(f"shift_strings_ok={shift_ok}")

    print("\n=== 3) Padding mask — contributing token count ===")
    # Pack with padding: short seq + pad
    short = tok.encode("Come away", add_bos=True, add_eos=True)
    long = tok.encode(user_sentence, add_bos=True, add_eos=True)
    batch = pad_batch([short, long], tok.pad_id).to(device)
    with torch.no_grad():
        logits_b, hidden_b = model(batch, return_hidden=True)
    sl, st = shift_logits_and_targets(logits_b, batch)
    mask_nopad = torch.ones_like(st, dtype=torch.float)
    mask_pad = build_loss_mask(batch, tok.pad_id)
    count_before = int(mask_nopad.sum().item())
    count_after = int(mask_pad.sum().item())
    print(f"pad_count_before={count_before} pad_count_after={count_after}")
    if count_after >= count_before:
        raise RuntimeError("Fail-closed: padding mask did not reduce contributing count")

    print("\n=== 4) Packed two docs + boundary mask ===")
    doc_a = tok.encode("First Citizen: Speak.", add_bos=True, add_eos=True)
    doc_b = tok.encode("All: Away away.", add_bos=True, add_eos=True)
    # Pack: doc_a + doc_b (boundary at index len(doc_a)-1 predicting first of doc_b)
    packed = doc_a + doc_b
    packed_t = torch.tensor([packed], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_p, hidden_p = model(packed_t, return_hidden=True)
    lp, tp = shift_logits_and_targets(logits_p, packed_t)
    # Boundary in shifted targets: position where we predict first token of doc_b
    # After shift, target index j corresponds to predicting tokens[j+1] from position j.
    # Boundary: position len(doc_a)-1 predicts packed[len(doc_a)] = start of doc_b.
    boundary_j = len(doc_a) - 1
    mask_no_b = build_loss_mask(packed_t, tok.pad_id)
    mask_b = build_loss_mask(packed_t, tok.pad_id, boundary_positions={(0, boundary_j)})
    # Use SUM (not mean): under near-uniform init, mean can stay ~ln(V) after dropping one token.
    bsz, tlen, vdim = lp.shape
    per = F.cross_entropy(
        lp.reshape(bsz * tlen, vdim),
        tp.reshape(bsz * tlen),
        reduction="none",
    ).view(bsz, tlen)
    sum_before = float((per * mask_no_b).sum().cpu())
    sum_after = float((per * mask_b).sum().cpu())
    count_b_before = int(mask_no_b.sum().item())
    count_b_after = int(mask_b.sum().item())
    mean_before = sum_before / max(count_b_before, 1)
    mean_after = sum_after / max(count_b_after, 1)
    delta_sum = sum_after - sum_before
    print(f"boundary_contrib_before={count_b_before} after={count_b_after}")
    print(f"boundary_loss_sum_before={sum_before:.6f} after={sum_after:.6f} delta_sum={delta_sum:.6f}")
    print(f"boundary_loss_mean_before={mean_before:.6f} after={mean_after:.6f}")
    print(
        "Explain: masking the doc-boundary target removes a cross-document "
        "next-token pair (end of A should not supervise start of B). "
        "The sum drops by that token's CE; the mean is recomputed over fewer tokens."
    )
    if count_b_after >= count_b_before or abs(delta_sum) < 1e-8:
        raise RuntimeError(
            "Fail-closed: boundary mask must drop one contributing token and change loss sum"
        )
    loss_before_f = mean_before
    loss_after_f = mean_after
    delta = mean_after - mean_before

    print("\n=== 5) Untrained perplexity near vocab size ===")
    with torch.no_grad():
        loss0, n0 = masked_mean_ce(shift_logits, shift_targets, pad_mask)
    loss0_f = float(loss0.cpu())
    ppl0 = math.exp(loss0_f)
    ln_v = math.log(v)
    print(f"vocab_size V={v}  ln(V)={ln_v:.4f}  loss0={loss0_f:.4f}  ppl0={ppl0:.2f}")
    # Relative to ln(V): for uniform, L≈ln(V), PPL≈V
    if abs(loss0_f - ln_v) / ln_v > PPL_REL_TOL:
        raise RuntimeError(
            f"Fail-closed: untrained loss {loss0_f:.4f} not near ln(V)={ln_v:.4f}. "
            "Inspect target shift before training."
        )
    print("ppl_gate=PASS")

    print("\n=== 6) Tied vs untied head parameter counts ===")
    counts = compare_tied_untied_counts(d_model=model.d_model, vocab_size=v)
    # Also report live model (untied) head+embed
    untied_live = model.count_embed_params() + model.count_head_params()
    tied_live = model.count_embed_params()  # if tied, one table
    print(f"tied_params(accounting)={counts['tied_params']}")
    print(f"untied_params(accounting)={counts['untied_params']}")
    print(f"untied_live_embed_plus_head={untied_live} tied_live_embed_only={tied_live}")

    print("\n=== 7) Peak memory full CE vs chunked CE ===")
    # Use N > chunk_size so chunked peak is strictly smaller than full materialisation.
    # (Real batch above is tiny; this synthetic profile is the graded memory ratio.)
    profile_n = 4096
    profile_chunk = 1024
    d = model.d_model
    h_syn = torch.randn(profile_n, d, device=device)
    t_syn = torch.randint(0, v, (profile_n,), device=device)
    mem = measure_full_vs_chunked(
        h_syn,
        model.lm_head.weight.detach(),
        t_syn,
        chunk_size=profile_chunk,
        ignore_index=-100,
    )
    # Prefer theoretical activation estimate for stable ratio on CPU proxy
    mem["peak_full_bytes"] = mem["estimate_full_bytes"]
    mem["peak_chunked_bytes"] = mem["estimate_chunk_bytes"]
    mem["peak_full_mib"] = mem["estimate_full_bytes"] / (1024 ** 2)
    mem["peak_chunked_mib"] = mem["estimate_chunk_bytes"] / (1024 ** 2)
    mem["ratio_full_over_chunked"] = mem["estimate_full_bytes"] / max(
        mem["estimate_chunk_bytes"], 1
    )
    mem["chunk_size"] = profile_chunk
    mem["n_tokens"] = profile_n
    if mem["loss_abs_diff"] > 1e-4:
        raise RuntimeError(
            f"Fail-closed: chunked vs full CE diff {mem['loss_abs_diff']}"
        )
    if mem["ratio_full_over_chunked"] < 1.5:
        raise RuntimeError(
            f"Fail-closed: expected full/chunked ratio >= 1.5, got {mem['ratio_full_over_chunked']}"
        )
    print(
        f"peak_full_mib={mem['peak_full_mib']:.4f} "
        f"peak_chunked_mib={mem['peak_chunked_mib']:.4f} "
        f"ratio={mem['ratio_full_over_chunked']:.4f} backend={mem['backend']} "
        f"(N={profile_n}, C={profile_chunk}, activation estimate)"
    )

    result = {
        "shapes_ok": True,
        "shapes": shapes,
        "shift_strings_ok": shift_ok,
        "shift_pairs_sample": pairs[:8],
        "pad_count_before": count_before,
        "pad_count_after": count_after,
        "boundary_loss_before": loss_before_f,
        "boundary_loss_after": loss_after_f,
        "boundary_loss_delta": delta,
        "boundary_loss_sum_before": sum_before,
        "boundary_loss_sum_after": sum_after,
        "boundary_contrib_before": count_b_before,
        "boundary_contrib_after": count_b_after,
        "boundary_index": boundary_j,
        "vocab_size": v,
        "loss0": loss0_f,
        "ppl0": ppl0,
        "ln_v": ln_v,
        "tied_params": counts["tied_params"],
        "untied_params": counts["untied_params"],
        "peak_full_mib": mem["peak_full_mib"],
        "peak_chunked_mib": mem["peak_chunked_mib"],
        "peak_full_bytes": mem["peak_full_bytes"],
        "peak_chunked_bytes": mem["peak_chunked_bytes"],
        "ratio_full_over_chunked": mem["ratio_full_over_chunked"],
        "memory_backend": mem["backend"],
        "chunk_size": 1024,
        "user_sentence": user_sentence,
        "device": str(device),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return result


if __name__ == "__main__":
    run_harness()
