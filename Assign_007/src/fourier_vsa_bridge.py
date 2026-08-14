"""Lift Problem #4 Fourier codes into fixed HRR_DIM for VSA verification."""
from __future__ import annotations

import torch

import config as cfg
from src.embeddings.fourier_baseline import FourierCodepointEmbedding
from src.hrr import normalize
from src.vsa_symbols import SymbolSpec, hv_for_spec, synthetic_image_patch_hv, synthetic_spectrogram_hv


class FourierVSABridge:
    def __init__(self, hrr_dim: int | None = None, code_dim: int | None = None, n_freq: int | None = None):
        self.hrr_dim = hrr_dim or cfg.HRR_DIM
        self.code_dim = code_dim or cfg.FOURIER_CODE_DIM
        self.codec = FourierCodepointEmbedding(
            d_model=8,
            code_dim=self.code_dim,
            n_freq=n_freq or cfg.FOURIER_N_FREQ,
            max_chars=cfg.MAX_CHARS_FOURIER,
        )

    def lift_vector(self, code: torch.Tensor) -> torch.Tensor:
        """Pad/truncate Fourier code to HRR_DIM, unitary-project, normalize."""
        from src.hrr import to_unitary

        flat = code.detach().float().flatten()
        out = torch.zeros(self.hrr_dim)
        n = min(flat.numel(), self.hrr_dim)
        out[:n] = flat[:n]
        if flat.numel() < self.hrr_dim:
            rep = flat
            i = n
            while i < self.hrr_dim:
                take = min(rep.numel(), self.hrr_dim - i)
                out[i : i + take] = rep[:take] * 0.01
                i += take
        return normalize(to_unitary(out))

    def from_text(self, text: str) -> torch.Tensor:
        return self.lift_vector(self.codec.encode_string(text))

    def from_text_truncated(self, text: str, max_chars: int) -> torch.Tensor:
        old = self.codec.max_chars
        self.codec.max_chars = max_chars
        try:
            return self.lift_vector(self.codec.encode_string(text))
        finally:
            self.codec.max_chars = old

    def from_spec(self, spec: SymbolSpec, prefer_fourier_text: bool = True) -> torch.Tensor:
        if spec.kind == "patch":
            return synthetic_image_patch_hv(self.hrr_dim, spec.seed)
        if spec.kind == "audio":
            return synthetic_spectrogram_hv(self.hrr_dim, spec.seed)
        if prefer_fourier_text and spec.text:
            return self.from_text(spec.text)
        return hv_for_spec(spec, self.hrr_dim)

    def random_pair_codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.from_text("alpha")
        b = self.from_text("beta")
        return a, b
