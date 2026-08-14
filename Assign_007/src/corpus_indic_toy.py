"""Synthetic EN + Indic corpus (hermetic, no downloads)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import config as cfg


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    script: str  # en | hi | te
    split: str  # train | eval
    stage_affinity: str


def _en_sentences(rng: random.Random, n: int) -> list[str]:
    stems = [
        "the cat sat on the mat",
        "india is a diverse country",
        "neural networks learn from data",
        "apple grows on trees and companies",
        "train and trainer share spelling",
        "the sun rises in the east",
        "code must be reproducible locally",
        "embedding tables cost memory",
    ]
    out = []
    for i in range(n):
        a = rng.choice(stems)
        b = rng.choice(stems)
        out.append(f"{a}. {b}. id={i}")
    return out


def _hi_sentences(rng: random.Random, n: int) -> list[str]:
    stems = [
        "राम स्कूल जाता है",
        "भारत एक विविध देश है",
        "अंतर्राष्ट्रीयकरण महत्वपूर्ण है",
        "अंतर्राष्ट्रीयता भी महत्वपूर्ण है",
        "तेलुगू और हिंदी दोनों भारत में बोली जाती हैं",
        "रेलगाड़ी स्टेशन पर खड़ी है",
        "शिक्षा सबके लिए आवश्यक है",
        "गणित और विज्ञान साथ चलते हैं",
    ]
    out = []
    for i in range(n):
        a = rng.choice(stems)
        b = rng.choice(stems)
        out.append(f"{a}। {b}। क्रमांक={i}")
    return out


def _te_sentences(rng: random.Random, n: int) -> list[str]:
    stems = [
        "తెలుగు ఒక భారతీయ భాష",
        "హైదరాబాద్ ఒక పెద్ద నగరం",
        "విద్య అందరికీ అవసరం",
        "రైలు స్టేషన్ వద్ద ఉంది",
    ]
    out = []
    for i in range(n):
        a = rng.choice(stems)
        b = rng.choice(stems)
        out.append(f"{a}. {b}. id={i}")
    return out


def build_corpus(seed: int | None = None) -> list[Doc]:
    seed = cfg.SEED if seed is None else seed
    rng = random.Random(seed)
    docs: list[Doc] = []

    def add(lines: list[str], script: str, split: str, stage: str, prefix: str) -> None:
        for i, text in enumerate(lines):
            docs.append(
                Doc(
                    doc_id=f"{prefix}_{split}_{i}",
                    text=text,
                    script=script,
                    split=split,
                    stage_affinity=stage,
                )
            )

    add(_en_sentences(rng, 40), "en", "train", "general", "en")
    add(_hi_sentences(rng, 40), "hi", "train", "indic_focus", "hi")
    add(_te_sentences(rng, 16), "te", "train", "indic_focus", "te")
    add(_en_sentences(rng, 12), "en", "train", "seed", "en_seed")
    add(_hi_sentences(rng, 12), "hi", "train", "anneal", "hi_an")
    add(_en_sentences(rng, 8), "en", "eval", "general", "en_ev")
    add(_hi_sentences(rng, 8), "hi", "eval", "indic_focus", "hi_ev")
    return docs
