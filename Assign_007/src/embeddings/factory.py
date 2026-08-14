"""Factory wrapper for embedding kinds."""
from __future__ import annotations

from torch import nn

from src.embeddings import build_embedding as _build


def build_embedding(**kwargs) -> nn.Module:
    return _build(**kwargs)
