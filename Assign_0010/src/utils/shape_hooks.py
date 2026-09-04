"""Forward-hook tensor shape logging for one training step."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.llm.mini_gpt import MiniGPT, MiniGPTBlock, MiniGPTConfig

SHAPE_HEADER = """# Tensor shape log — Session 10 Assignment Task 1
# Dimension legend:
#   B = batch size (number of sequences)
#   T = time / sequence length (context tokens)
#   D / C = model hidden dimension (n_embd)
#   V = vocabulary size (logits last dim)
"""


def _interpret_shape(name: str, shape: tuple[int, ...], cfg: MiniGPTConfig) -> str:
    if len(shape) == 2 and shape[1] == cfg.n_embd:
        return f"B={shape[0]} T={shape[1] // cfg.n_embd if shape[1] != cfg.n_embd else '?'} D={cfg.n_embd}"
    if len(shape) == 3:
        return f"B={shape[0]} T={shape[1]} D={shape[2]}"
    if len(shape) == 3 and shape[2] == cfg.vocab_size:
        return f"B={shape[0]} T={shape[1]} V={shape[2]}"
    if len(shape) == 2 and shape[0] == cfg.vocab_size:
        return f"V={shape[0]} D={shape[1]}"
    dims = ", ".join(str(s) for s in shape)
    return f"dims=({dims})"


def collect_shape_log(
    model: MiniGPT,
    batch: torch.Tensor,
    out_path: Path,
) -> list[str]:
    """Run one forward pass; register hooks; write shape lines to out_path."""
    cfg = model.config
    lines: list[str] = [SHAPE_HEADER.strip()]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str):
        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if not isinstance(tensor, torch.Tensor):
                return
            interp = _interpret_shape(module_name, tuple(tensor.shape), cfg)
            line = f"{module_name}: shape={tuple(tensor.shape)} | {interp}"
            lines.append(line)

        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.Linear, MiniGPTBlock, nn.MultiheadAttention)):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        logits = model(batch)
    lines.append(
        f"lm_head_output: shape={tuple(logits.shape)} | "
        f"B={logits.shape[0]} T={logits.shape[1]} V={logits.shape[2]}"
    )

    for h in handles:
        h.remove()

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return lines[1:]  # exclude header for count


def run_tensor_shapes(out_path: Path, batch_size: int = 2, block_size: int = 128) -> dict:
    cfg = MiniGPTConfig(block_size=block_size)
    model = MiniGPT(cfg)
    batch = torch.randint(0, cfg.vocab_size, (batch_size, block_size))
    shape_lines = collect_shape_log(model, batch, out_path)
    return {
        "path": str(out_path),
        "n_lines": len(shape_lines),
        "ok": len(shape_lines) >= 10,
    }
