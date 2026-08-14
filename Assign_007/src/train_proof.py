"""Train dense / kronecker / fourier / fourier_canine (+ optional VQ) with RoPE LM."""
from __future__ import annotations

import math
import random
from typing import Any

import torch

import config as cfg
from src.batching import encode_docs
from src.corpus_indic_toy import Doc
from src.embeddings import adamw_train_state_gb, build_embedding, count_trainable
from src.mixture_stages import docs_for_stage, iter_stage_steps
from src.tiny_transformer import TinyTransformerLM
from src.tokenizer_wrapper import LocalHashTokenizer


def ablation_kinds() -> tuple[str, ...]:
    kinds = ["dense", "kronecker", "fourier"]
    if cfg.USE_FOURIER_CANINE:
        kinds.append("fourier_canine")
    if cfg.ENABLE_VQ_PROBLEM5:
        kinds.append("fourier_vq")
    return tuple(kinds)


def _eval_ce(
    model: TinyTransformerLM,
    docs: list[Doc],
    tok: LocalHashTokenizer,
    device: str,
    script_filter: str | None = None,
) -> float:
    model.eval()
    subset = [d for d in docs if d.split == "eval"]
    if script_filter:
        subset = [d for d in subset if d.script == script_filter]
    if not subset:
        subset = [d for d in docs if d.split == "train"][:8]
        if script_filter:
            subset = [d for d in subset if d.script == script_filter] or subset
    ids, strs = encode_docs(subset[:16], tok)
    ids = ids.to(device)
    with torch.no_grad():
        loss = model.loss(ids, strs if getattr(model.embed, "requires_token_text", False) else None)
    return float(loss.item())


def discrimination_accuracy(
    model: TinyTransformerLM,
    tok: LocalHashTokenizer,
    device: str,
) -> float:
    model.eval()
    pairs = [
        ("अंतर्राष्ट्रीयकरण", "महत्वपूर्ण"),
        ("अंतर्राष्ट्रीयता", "भी"),
        ("भारत", "देश"),
        ("राम", "स्कूल"),
    ]
    correct = 0
    total = 0
    for left, right in pairs:
        for wrong in ("xyz", "apple", "तेलुगू"):
            a_ids = tok.encode(left)[: cfg.SEQ_LEN - 1]
            rid = tok.token_to_id.get(right, tok.unk_id) if tok.token_to_id else tok.unk_id
            wid = tok.token_to_id.get(wrong, tok.unk_id) if tok.token_to_id else tok.unk_id
            seq = a_ids + [cfg.PAD_ID] * (cfg.SEQ_LEN - len(a_ids))
            strs = [tok.decode_id(i) for i in seq]
            x = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(
                    x,
                    [strs] if getattr(model.embed, "requires_token_text", False) else None,
                )
            pos = max(len(a_ids) - 1, 0)
            correct += int(float(logits[0, pos, rid]) > float(logits[0, pos, wid]))
            total += 1
    return correct / max(total, 1)


def train_arm(
    kind: str,
    docs: list[Doc],
    tok: LocalHashTokenizer,
    device: str | None = None,
) -> dict[str, Any]:
    device = device or cfg.DEVICE
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    embed = build_embedding(
        kind=kind,
        vocab_size=cfg.VOCAB_SIZE,
        d_model=cfg.D_MODEL,
        pad_id=cfg.PAD_ID,
        pos_dim=cfg.POS_DIM_KRON,
        fourier_code_dim=cfg.FOURIER_CODE_DIM,
        fourier_n_freq=cfg.FOURIER_N_FREQ,
        max_chars=cfg.MAX_CHARS_FOURIER,
        canine_stride=cfg.CANINE_STRIDE,
        vq_num_codes=cfg.VQ_NUM_CODES,
    )
    model = TinyTransformerLM(
        embed=embed,
        vocab_size=cfg.VOCAB_SIZE,
        d_model=cfg.D_MODEL,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        seq_len=cfg.SEQ_LEN,
        pad_id=cfg.PAD_ID,
        position_policy=cfg.POSITION_POLICY,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    history: list[dict[str, Any]] = []
    grad_proj: list[float] = []
    grad_l1: list[float] = []
    vq_losses: list[float] = []
    step_global = 0

    for stage_i, stage in enumerate(cfg.STAGES):
        for batch_docs in iter_stage_steps(
            docs, stage, cfg.STEPS_PER_STAGE, cfg.BATCH_SIZE, cfg.SEED + stage_i
        ):
            model.train()
            ids, strs = encode_docs(batch_docs, tok)
            ids = ids.to(device)
            opt.zero_grad(set_to_none=True)
            need_str = getattr(model.embed, "requires_token_text", False)
            loss = model.loss(ids, strs if need_str else None)
            if kind == "fourier_vq" and getattr(model.embed, "vq_loss", None) is not None:
                loss = loss + cfg.VQ_LOSS_WEIGHT * model.embed.vq_loss
                vq_losses.append(float(model.embed.vq_loss.detach().item()))
            loss.backward()

            if hasattr(model.embed, "projection_parameters"):
                pn = 0.0
                for p in model.embed.projection_parameters():
                    if p.grad is not None:
                        pn += float(p.grad.detach().norm().item() ** 2)
                grad_proj.append(math.sqrt(pn))
            l1 = 0.0
            for p in model.blocks[0].parameters():
                if p.grad is not None:
                    l1 += float(p.grad.detach().norm().item() ** 2)
            grad_l1.append(math.sqrt(l1))

            opt.step()
            history.append({"stage": stage, "step": step_global, "loss": float(loss.item())})
            step_global += 1

    # freeze-shift diagnostic
    if hasattr(model.embed, "projection_parameters"):
        model.train()
        ids, strs = encode_docs(docs_for_stage(docs, "indic_focus")[: cfg.BATCH_SIZE], tok)
        ids = ids.to(device)
        for p in model.embed.projection_parameters():
            p.requires_grad_(False)
        opt.zero_grad(set_to_none=True)
        need_str = getattr(model.embed, "requires_token_text", False)
        loss_f = model.loss(ids, strs if need_str else None)
        loss_f.backward()
        pn = 0.0
        for p in model.embed.projection_parameters():
            if p.grad is not None:
                pn += float(p.grad.detach().norm().item() ** 2)
        l1 = 0.0
        for p in model.blocks[0].parameters():
            if p.grad is not None:
                l1 += float(p.grad.detach().norm().item() ** 2)
        grad_proj.append(math.sqrt(pn) if pn > 0 else 0.0)
        grad_l1.append(math.sqrt(l1))
        for p in model.embed.projection_parameters():
            p.requires_grad_(True)

    n_train = count_trainable(model)
    metrics = {
        "kind": kind,
        "position_policy": cfg.POSITION_POLICY,
        "trainable_params": n_train,
        "adamw_train_state_gb": adamw_train_state_gb(n_train),
        "final_train_loss": history[-1]["loss"] if history else None,
        "val_ce_all": _eval_ce(model, docs, tok, device, None),
        "val_ce_en": _eval_ce(model, docs, tok, device, "en"),
        "val_ce_hi": _eval_ce(model, docs, tok, device, "hi"),
        "discrimination_acc": discrimination_accuracy(model, tok, device),
        "grad_proj_trace": grad_proj,
        "grad_l1_trace": grad_l1,
        "history": history,
        "embed_params": count_trainable(model.embed),
        "mean_vq_loss": (sum(vq_losses) / len(vq_losses)) if vq_losses else None,
    }
    return {"model": model, "metrics": metrics}


def run_ablation(docs: list[Doc], tok: LocalHashTokenizer, device: str | None = None) -> dict[str, Any]:
    out = {}
    for kind in ablation_kinds():
        result = train_arm(kind, docs, tok, device=device)
        out[kind] = result["metrics"]
        del result
    return out
