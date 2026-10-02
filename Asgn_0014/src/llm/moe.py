"""Mixture-of-Experts FFN, FP32 router, three-tier loss-free bias, dense->MoE upcycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.llm.tiny_lm import CausalSelfAttention, LMConfig, TinyDenseLM, count_parameters


@dataclass(frozen=True)
class MoEConfig:
    """MoE shape. Default N=8, k=2 with width identity vs dense n_inner."""

    vocab_size: int = 8192
    n_embd: int = 384
    n_layer: int = 10
    n_head: int = 6
    n_experts: int = 8
    top_k: int = 2
    d_ff_expert: int = 192
    dropout: float = 0.0
    bias_gamma: float = 0.001  # used when use_three_tier=False (proportional scale)
    drop_redraw_frac: float = 0.5
    use_ce_aux: bool = False  # locked OFF for graded path
    use_three_tier: bool = True
    # Stronger than Extra micro-steps: Colab 1M+ MoE collapsed with γ≈1e-4 / 200 steps.
    gamma1: float = 0.0005
    gamma2: float = 0.002
    gamma3: float = 0.01
    thresh1: float = 0.02
    thresh2: float = 0.05
    warmup_steps: int = 2000  # probabilistic selection after upcycle (~8M tok @ B=8,S=512)


class SwiGLUExpert(nn.Module):
    def __init__(self, n_embd: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(n_embd, d_ff, bias=False)
        self.w_up = nn.Linear(n_embd, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MoELayer(nn.Module):
    """Dropless MoE with loss-free bias (choose only) + optional probabilistic warm-up."""

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.bias_gamma = cfg.bias_gamma
        self.use_three_tier = cfg.use_three_tier
        self.gamma1 = cfg.gamma1
        self.gamma2 = cfg.gamma2
        self.gamma3 = cfg.gamma3
        self.thresh1 = cfg.thresh1
        self.thresh2 = cfg.thresh2
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.router = self.router.to(dtype=torch.float32)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(cfg.n_embd, cfg.d_ff_expert, cfg.dropout) for _ in range(cfg.n_experts)]
        )
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts, dtype=torch.float32))
        self._last_load: Optional[Tensor] = None
        self._last_max_vio: float = 0.0
        self._last_dead: int = 0
        self.probabilistic: bool = False  # set True during warm-up

    def _select_experts(self, choose_scores: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (top_idx, gate_logits_from_raw) shapes (T, k)."""
        if self.probabilistic:
            # Soft multinomial without replacement for top_k (clone warm-up fix)
            probs = torch.softmax(choose_scores, dim=-1)
            # Gumbel-top-k for differentiable-free sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(probs).clamp_min(1e-9) + 1e-9) + 1e-9)
            noisy = torch.log(probs.clamp_min(1e-9)) + gumbel
            _top_val, top_idx = torch.topk(noisy, k=self.top_k, dim=-1)
            return top_idx, choose_scores
        _top_val, top_idx = torch.topk(choose_scores, k=self.top_k, dim=-1)
        return top_idx, choose_scores

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat.float())  # (T, E) FP32
        choose_scores = logits + self.expert_bias
        top_idx, _ = self._select_experts(choose_scores)
        # Gate weights from raw logits (not bias), softmax over chosen
        raw_chosen = torch.gather(logits, dim=-1, index=top_idx)
        gates = torch.softmax(raw_chosen, dim=-1)

        out = torch.zeros_like(flat, dtype=flat.dtype)
        for e in range(self.n_experts):
            mask = top_idx == e
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            expert_in = flat[token_ids]
            expert_out = self.experts[e](expert_in)
            w = gates[token_ids, slot_ids].unsqueeze(-1).to(expert_out.dtype)
            out.index_add_(0, token_ids, expert_out * w)

        with torch.no_grad():
            load = torch.zeros(self.n_experts, device=flat.device, dtype=torch.float32)
            ones = torch.ones(top_idx.numel(), device=flat.device, dtype=torch.float32)
            load.scatter_add_(0, top_idx.reshape(-1), ones)
            load = load / max(1.0, float(top_idx.numel()))
            mean = load.mean().clamp_min(1e-8)
            max_vio = float(((load.max() - mean) / mean).item())
            dead = int((load < 1e-12).sum().item())
            self._last_load = load.detach().cpu()
            self._last_max_vio = max_vio
            self._last_dead = dead

        return out.view(b, s, d)

    @torch.no_grad()
    def update_bias(self) -> None:
        """Loss-free balancer. Default: Extra three-tier; else proportional γ*(target−f)."""
        if self._last_load is None:
            return
        load = self._last_load.to(self.expert_bias.device)
        target = 1.0 / float(self.n_experts)
        err = load - target  # positive = overloaded

        if self.use_three_tier:
            abs_err = err.abs()
            step = torch.zeros_like(err)
            t1 = abs_err <= self.thresh1
            t2 = (abs_err > self.thresh1) & (abs_err <= self.thresh2)
            t3 = abs_err > self.thresh2
            step[t1] = -self.gamma1 * torch.sign(err[t1])
            step[t2] = -self.gamma2 * err[t2]
            step[t3] = -self.gamma3 * torch.sign(err[t3]) * (err[t3] ** 2)
            # exact-zero err: no update
            step = torch.where(err == 0, torch.zeros_like(step), step)
            self.expert_bias.add_(step)
        else:
            # DeepSeek-style proportional: b ← b + γ · (f_target − f_i)
            self.expert_bias.add_(self.bias_gamma * (target - load))
        self.expert_bias.clamp_(-5.0, 5.0)


class MoEBlock(nn.Module):
    def __init__(self, dense_cfg: LMConfig, moe_cfg: MoEConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dense_cfg.n_embd)
        self.attn = CausalSelfAttention(dense_cfg)
        self.ln2 = nn.LayerNorm(dense_cfg.n_embd)
        self.moe = MoELayer(moe_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class TinyMoELM(nn.Module):
    def __init__(self, moe_cfg: MoEConfig | None = None, dense_cfg: LMConfig | None = None) -> None:
        super().__init__()
        self.moe_cfg = moe_cfg or MoEConfig()
        self.dense_cfg = dense_cfg or LMConfig(
            vocab_size=self.moe_cfg.vocab_size,
            n_embd=self.moe_cfg.n_embd,
            n_layer=self.moe_cfg.n_layer,
            n_head=self.moe_cfg.n_head,
            dropout=self.moe_cfg.dropout,
        )
        c = self.dense_cfg
        m = self.moe_cfg
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList([MoEBlock(c, m) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.moe.router = blk.moe.router.to(dtype=torch.float32)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_probabilistic(self, enabled: bool) -> None:
        for blk in self.blocks:
            blk.moe.probabilistic = bool(enabled)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def count_parameters(self) -> int:
        return count_parameters(self)

    def count_active_parameters(self) -> int:
        c = self.dense_cfg
        m = self.moe_cfg
        embed = c.vocab_size * c.n_embd
        per_attn = (
            3 * c.n_embd * c.n_embd
            + c.n_embd * c.n_embd
            + 2 * c.n_embd
            + 2 * c.n_embd
        )
        expert = 3 * c.n_embd * m.d_ff_expert
        router = c.n_embd * m.n_experts
        per_layer_active = per_attn + m.top_k * expert + router
        ln_f = 2 * c.n_embd
        return int(embed + c.n_layer * per_layer_active + ln_f)

    def update_all_biases(self) -> None:
        for blk in self.blocks:
            blk.moe.update_bias()

    def routing_stats(self) -> Dict[str, float]:
        max_vios: List[float] = []
        deads: List[int] = []
        for blk in self.blocks:
            max_vios.append(blk.moe._last_max_vio)
            deads.append(blk.moe._last_dead)
        return {
            "max_vio": float(max(max_vios) if max_vios else 0.0),
            "dead_experts": float(max(deads) if deads else 0.0),
            "mean_max_vio": float(sum(max_vios) / len(max_vios) if max_vios else 0.0),
        }

    def load_histogram(self) -> Optional[Tensor]:
        loads = [blk.moe._last_load for blk in self.blocks if blk.moe._last_load is not None]
        if not loads:
            return None
        stacked = torch.stack(loads, dim=0)
        return stacked.mean(dim=0)


def _partition_drop_init_expert(
    expert: SwiGLUExpert,
    dense_fc: nn.Linear,
    dense_proj: nn.Linear,
    expert_idx: int,
    n_experts: int,
    drop_frac: float,
    generator: torch.Generator,
) -> None:
    d_inner = dense_fc.out_features
    d_ff = expert.w_gate.out_features
    start = (expert_idx * d_ff) % max(1, d_inner - d_ff + 1) if d_inner >= d_ff else 0
    end = start + d_ff
    if end <= d_inner:
        slice_fc = dense_fc.weight.data[start:end, :].clone()
        slice_proj = dense_proj.weight.data[:, start:end].clone()
    else:
        part1 = dense_fc.weight.data[start:, :].clone()
        part2 = dense_fc.weight.data[: end - d_inner, :].clone()
        slice_fc = torch.cat([part1, part2], dim=0)
        p1 = dense_proj.weight.data[:, start:].clone()
        p2 = dense_proj.weight.data[:, : end - d_inner].clone()
        slice_proj = torch.cat([p1, p2], dim=1)

    expert.w_gate.weight.data.copy_(slice_fc)
    expert.w_up.weight.data.copy_(slice_fc)
    expert.w_down.weight.data.copy_(slice_proj)

    n_redraw = max(1, int(d_ff * drop_frac))
    idx = torch.randperm(d_ff, generator=generator)[:n_redraw]
    expert.w_gate.weight.data[idx].normal_(0.0, 0.02)
    expert.w_up.weight.data[idx].normal_(0.0, 0.02)
    expert.w_down.weight.data[:, idx].normal_(0.0, 0.02)


def upcycle_dense_to_moe(
    dense: TinyDenseLM,
    *,
    n_experts: int = 8,
    top_k: int = 2,
    bias_gamma: float = 0.001,
    drop_redraw_frac: float = 0.5,
    seed: int = 14,
    warmup_steps: int = 2000,
    use_three_tier: bool = True,
) -> TinyMoELM:
    """Convert dense TinyDenseLM into TinyMoELM (partition + drop)."""
    c = dense.cfg
    d_ff_expert = max(32, c.n_inner // top_k)
    moe_cfg = MoEConfig(
        vocab_size=c.vocab_size,
        n_embd=c.n_embd,
        n_layer=c.n_layer,
        n_head=c.n_head,
        n_experts=n_experts,
        top_k=top_k,
        d_ff_expert=d_ff_expert,
        dropout=c.dropout,
        bias_gamma=bias_gamma,
        drop_redraw_frac=drop_redraw_frac,
        use_ce_aux=False,
        use_three_tier=use_three_tier,
        warmup_steps=warmup_steps,
    )
    moe = TinyMoELM(moe_cfg=moe_cfg, dense_cfg=c)
    moe.embed.load_state_dict(dense.embed.state_dict())
    moe.ln_f.load_state_dict(dense.ln_f.state_dict())
    gen = torch.Generator()
    gen.manual_seed(seed)
    for d_blk, m_blk in zip(dense.blocks, moe.blocks):
        m_blk.ln1.load_state_dict(d_blk.ln1.state_dict())
        m_blk.attn.load_state_dict(d_blk.attn.state_dict())
        m_blk.ln2.load_state_dict(d_blk.ln2.state_dict())
        for e_idx, expert in enumerate(m_blk.moe.experts):
            _partition_drop_init_expert(
                expert,
                d_blk.mlp.fc,
                d_blk.mlp.proj,
                e_idx,
                n_experts,
                drop_redraw_frac,
                gen,
            )
        nn.init.normal_(m_blk.moe.router.weight, mean=0.0, std=0.02)
        m_blk.moe.router = m_blk.moe.router.to(dtype=torch.float32)
    return moe


def active_vs_total(moe: TinyMoELM) -> Tuple[int, int]:
    return moe.count_active_parameters(), moe.count_parameters()
