"""Deterministic VSA symbol bank (digits, ops, roles, Abugida, emoji, media stand-ins)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.hrr import normalize, random_hv, to_unitary


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    kind: str  # role | digit | op | lexical | emoji | modality | patch | audio
    seed: int
    text: str | None = None  # optional Unicode for Fourier lift


# Fixed seeds so reports are reproducible
ROLES = {
    "LEFT": SymbolSpec("LEFT", "role", 1001),
    "RIGHT": SymbolSpec("RIGHT", "role", 1002),
    "OP": SymbolSpec("OP", "role", 1003),
    "RESULT": SymbolSpec("RESULT", "role", 1004),
    "LEMMA": SymbolSpec("LEMMA", "role", 1005),
    "SCRIPT_HI": SymbolSpec("SCRIPT_HI", "role", 1006, text="HI"),
    "SCRIPT_EN": SymbolSpec("SCRIPT_EN", "role", 1007, text="EN"),
    "ICON": SymbolSpec("ICON", "role", 1008),
    "MODALITY_IMAGE": SymbolSpec("MODALITY_IMAGE", "modality", 1009),
    "MODALITY_AUDIO": SymbolSpec("MODALITY_AUDIO", "modality", 1010),
    "FRENCH": SymbolSpec("FRENCH", "role", 1011),
    "ENGLISH": SymbolSpec("ENGLISH", "role", 1012),
}

OPS = {
    "ADD": SymbolSpec("ADD", "op", 2001, text="+"),
    "MUL": SymbolSpec("MUL", "op", 2002, text="*"),
    "DIV": SymbolSpec("DIV", "op", 2003, text="/"),
    "SUB": SymbolSpec("SUB", "op", 2004, text="-"),
}

DIGITS = {str(i): SymbolSpec(str(i), "digit", 3000 + i, text=str(i)) for i in range(0, 65)}

LEXICAL = {
    "a": SymbolSpec("a", "lexical", 4001, text="a"),
    "b": SymbolSpec("b", "lexical", 4002, text="b"),
    "c": SymbolSpec("c", "lexical", 4003, text="c"),
    "Fruit": SymbolSpec("Fruit", "lexical", 4004, text="Fruit"),
    "Mango": SymbolSpec("Mango", "lexical", 4005, text="Mango"),
    "Apple": SymbolSpec("Apple", "lexical", 4006, text="Apple"),
    "Pomme": SymbolSpec("Pomme", "lexical", 4007, text="Pomme"),
    "Ram": SymbolSpec("Ram", "lexical", 4008, text="Ram"),
    "राम": SymbolSpec("राम", "lexical", 4009, text="राम"),
    "क": SymbolSpec("क", "lexical", 4010, text="क"),
    "का": SymbolSpec("का", "lexical", 4011, text="का"),
    "क्ष": SymbolSpec("क्ष", "lexical", 4012, text="क्ष"),
    "अंतर्राष्ट्रीयकरण": SymbolSpec("अंतर्राष्ट्रीयकरण", "lexical", 4013, text="अंतर्राष्ट्रीयकरण"),
    "अंतर्राष्ट्रीयता": SymbolSpec("अंतर्राष्ट्रीयता", "lexical", 4014, text="अंतर्राष्ट्रीयता"),
}

EMOJI = {
    "apple_icon": SymbolSpec("apple_icon", "emoji", 5001, text="🍎"),
    "mango_icon": SymbolSpec("mango_icon", "emoji", 5002, text="🥭"),
    "person": SymbolSpec("person", "emoji", 5003, text="🧑"),
}

MEDIA = {
    "patch_A": SymbolSpec("patch_A", "patch", 6001),
    "patch_B": SymbolSpec("patch_B", "patch", 6002),
    "clip_A": SymbolSpec("clip_A", "audio", 6003),
    "clip_B": SymbolSpec("clip_B", "audio", 6004),
    "tree_A": SymbolSpec("tree_A", "lexical", 6005, text="A"),
    "tree_B": SymbolSpec("tree_B", "lexical", 6006, text="B"),
    "tree_C": SymbolSpec("tree_C", "lexical", 6007, text="C"),
    "tree_D": SymbolSpec("tree_D", "lexical", 6008, text="D"),
}


def hv_for_spec(spec: SymbolSpec, dim: int) -> torch.Tensor:
    return normalize(to_unitary(random_hv(dim, seed=spec.seed, unitary=True)))


def synthetic_spectrogram_hv(dim: int, seed: int, n_bands: int = 32) -> torch.Tensor:
    """Hermetic audio stand-in: log-spaced sinusoid energy vector padded to dim."""
    g = torch.Generator()
    g.manual_seed(seed)
    bands = torch.linspace(0.5, 8.0, n_bands)
    phase = torch.rand(n_bands, generator=g) * 6.28
    energies = torch.sin(bands * 3.0 + phase).abs()
    v = torch.zeros(dim)
    v[:n_bands] = energies
    v[n_bands:] = torch.randn(dim - n_bands, generator=g) * 0.01
    return normalize(to_unitary(v))


def synthetic_image_patch_hv(dim: int, seed: int) -> torch.Tensor:
    """Hermetic image stand-in: seeded random patch embedding."""
    return normalize(random_hv(dim, seed=seed + 777, unitary=True))
