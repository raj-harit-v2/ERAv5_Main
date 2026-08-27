"""Part 2: short MTP smoke train (k=2) → data/evaluation/part2_losses.json."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.llm.mtp_heads import MTPDualHeads
from src.llm.nano_lm import NanoLM
from src.llm.tokenizer import WordTokenizer
from src.pipeline.harness import pad_batch

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "shakespeare_tiny.txt"
OUT = ROOT / "data" / "evaluation" / "part2_losses.json"


def _batches(tok: WordTokenizer, device: torch.device, seq_len: int = 48, batch_size: int = 4):
    text = RAW.read_text(encoding="utf-8")
    ids = tok.encode(text, add_bos=True, add_eos=True)
    # Sliding windows
    windows: list[list[int]] = []
    step = seq_len // 2
    for start in range(0, max(1, len(ids) - seq_len), step):
        windows.append(ids[start : start + seq_len])
        if len(windows) >= 32:
            break
    if not windows:
        windows = [ids[:seq_len]]
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        yield pad_batch(chunk, tok.pad_id).to(device)


def run_mtp_smoke(
    steps: int = 40,
    lr: float = 3e-3,
    device: str | None = None,
    seed: int = 1,
) -> dict:
    torch.manual_seed(seed)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = WordTokenizer.from_file(RAW)
    model = NanoLM(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq=128,
        tie_weights=False,
    ).to(device_t)
    mtp = MTPDualHeads(d_model=64, vocab_size=tok.vocab_size).to(device_t)
    # Trunk without its own lm_head for MTP: use hidden from model
    opt = torch.optim.Adam(
        list(model.parameters()) + list(mtp.parameters()),
        lr=lr,
    )

    history: list[dict] = []
    data = list(_batches(tok, device_t))
    step = 0
    model.train()
    mtp.train()
    while step < steps:
        for batch in data:
            if batch.size(1) < 4:
                continue
            opt.zero_grad(set_to_none=True)
            _, hidden = model(batch, return_hidden=True)
            # Mask pad in MTP via ignore_index
            losses = mtp.forward_losses(hidden, batch, ignore_index=tok.pad_id)
            losses["loss_sum"].backward()
            opt.step()
            row = {
                "step": step,
                "L1": float(losses["loss1"].detach().cpu()),
                "L2": float(losses["loss2"].detach().cpu()),
                "sum": float(losses["loss_sum"].detach().cpu()),
            }
            history.append(row)
            if step % 10 == 0:
                print(
                    f"step={step} L1={row['L1']:.4f} L2={row['L2']:.4f} sum={row['sum']:.4f}"
                )
            step += 1
            if step >= steps:
                break

    first, last = history[0], history[-1]
    explanation = (
        f"Head1 (t+1) started at L1={first['L1']:.3f} and ended at {last['L1']:.3f}; "
        f"Head2 (t+2) started at L2={first['L2']:.3f} and ended at {last['L2']:.3f}. "
        "Typically L2 stays higher / falls slower than L1 because t+2 is a harder "
        "longer-range target from the same trunk state (less local evidence)."
    )
    result = {
        "steps": steps,
        "history": history,
        "L1_final": last["L1"],
        "L2_final": last["L2"],
        "sum_final": last["sum"],
        "L1_initial": first["L1"],
        "L2_initial": first["L2"],
        "explanation": explanation,
        "device": str(device_t),
        "vocab_size": tok.vocab_size,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(explanation)
    print(f"Wrote {OUT}")
    return result


if __name__ == "__main__":
    run_mtp_smoke()
