"""Export NanoLM weights + vocab for browser widgets."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.llm.nano_lm import NanoLM
from src.llm.output_head import compare_tied_untied_counts
from src.llm.tokenizer import BOS, EOS, PAD, UNK
from src.pipeline.harness import load_tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "evaluation" / "widget_weights.json"


def _tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()


def export_widget_weights(seed: int = 0, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    tok = load_tokenizer()
    v = tok.vocab_size
    model = NanoLM(
        vocab_size=v,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq=256,
        tie_weights=False,
    ).to(device)
    model.eval()

    id_to_token = [tok.id_to_token[i] for i in range(v)]
    counts = compare_tied_untied_counts(d_model=model.d_model, vocab_size=v)

    state: dict[str, list] = {}
    for name, param in model.state_dict().items():
        state[name] = _tensor_to_list(param)

    bundle = {
        "version": 1,
        "seed": seed,
        "special": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "UNK": UNK},
        "pad_id": tok.pad_id,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "unk_id": tok.unk_id,
        "vocab_size": v,
        "id_to_token": id_to_token,
        "token_to_id": tok.token_to_id,
        "hyperparams": {
            "d_model": model.d_model,
            "n_layers": len(model.blocks),
            "n_heads": model.blocks[0].attn.n_heads,
            "max_seq": model.max_seq,
            "d_ff_mult": 4,
        },
        "tied_params": counts["tied_params"],
        "untied_params": counts["untied_params"],
        "state_dict": state,
    }
    return bundle


def run_export(path: Path | None = None) -> Path:
    path = path or OUT
    bundle = export_widget_weights(seed=0, device="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size // 1024} KiB)")
    return path


if __name__ == "__main__":
    run_export()
