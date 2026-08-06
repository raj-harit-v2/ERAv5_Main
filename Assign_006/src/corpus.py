"""Synthetic multi-lane document corpus for hermetic smoke demos."""
from __future__ import annotations

import hashlib
import random
from typing import Any

import config as cfg

LANES = list(cfg.CAPABILITY_LANES)
EVAL_LANE = cfg.EVAL_LANE_TAG
TEST_LANE = cfg.TEST_LANE_TAG

HINDI_SNIPPETS = [
    "भारत एक विशाल देश है जहाँ कई भाषाएँ बोली जाती हैं।",
    "राम स्कूल जाता है और गणित सीखता है।",
    "विज्ञान और तकनीक से समाज बदल रहा है।",
]
TELUGU_SNIPPETS = [
    "తెలుగు భాష చాలా అందమైనది మరియు పురాతనమైనది.",
    "విద్య అనేది అందరికీ అవసరం.",
]

CANARY_EVAL = "EVAL_CANARY_BENCHMARK_NEEDLE_X9F2"
CANARY_TEST = "TEST_CANARY_HOLD_OUT_NEEDLE_Z7Q1"


def _doc(
    doc_id: str,
    lane: str,
    text: str,
    *,
    tier: str = "b",
    difficulty: str = "B1",
    reasoning_length: str = "short",
    license: str = "cc0",
    lang: str = "en",
    tier_a: bool = False,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "lane": lane,
        "text": text,
        "tier": tier,
        "difficulty": difficulty,
        "reasoning_length": reasoning_length,
        "license": license,
        "source": "synthetic_a06",
        "lang": lang,
        "n_chars": len(text),
        "tier_a": tier_a,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _web_text(i: int) -> str:
    return (
        f"Web article {i}: Large language models learn from diverse internet text. "
        f"Quality filtering and deduplication remain critical. Paragraph {i} expands "
        f"on retrieval-augmented generation and citation habits. "
        + ("More context. " * (8 + i % 5))
    )


def _code_text(i: int) -> str:
    return (
        f"# module_utils_{i}.py\n"
        f"def fib(n):\n"
        f"    if n < 2:\n"
        f"        return n\n"
        f"    a, b = 0, 1\n"
        f"    for _ in range(n - 1):\n"
        f"        a, b = b, a + b\n"
        f"    return b\n\n"
        f"def main():\n"
        f"    print(fib({i + 3}))\n"
        f"\nif __name__ == '__main__':\n"
        f"    main()\n"
    )


def _stem_text(i: int) -> str:
    return (
        f"STEM note {i}: The derivative of x^{i + 2} is {(i + 2)}*x^{i + 1}. "
        f"Entropy of a fair coin is 1 bit. Attention is O(n^2) in sequence length."
    )


def _reasoning_text(i: int) -> str:
    return (
        f"Q: Is {90 + i} prime? Reason step by step. "
        f"Check divisibility by primes up to sqrt. Conclude with yes/no. "
        f"Self-check arithmetic carefully."
    )


def _indic_text(i: int) -> str:
    hi = HINDI_SNIPPETS[i % len(HINDI_SNIPPETS)]
    te = TELUGU_SNIPPETS[i % len(TELUGU_SNIPPETS)]
    return f"{hi} {te} Indic sample {i} for fertility stress-test."


def _agentic_text(i: int) -> str:
    return (
        f"<user>[ISSUE] Fix flaky test #{i} in CI.</user>"
        f"<obs>[OBS] pytest failed on test_timeout line 42.</obs>"
        f"<plan>[PLAN] Reproduce locally, inspect race, add lock.</plan>"
        f"<tool>[TOOL] run pytest -k timeout -vv</tool>"
        f"<final>[FINAL] Added lock around shared counter; tests green.</final>"
    )


def _long_text(i: int) -> str:
    body = (
        f"Long document {i}. "
        + "Sentence expands narrative context for packing stress. " * (20 + i % 10)
    )
    return body


_GENERATORS = {
    "web": _web_text,
    "code": _code_text,
    "stem": _stem_text,
    "reasoning": _reasoning_text,
    "indic": _indic_text,
    "agentic": _agentic_text,
    "long_context": _long_text,
}


def generate_corpus(
    docs_per_lane: int = cfg.DOCS_PER_LANE,
    eval_docs: int = cfg.EVAL_DOCS,
    test_docs: int = cfg.TEST_DOCS,
    seed: int = cfg.SEED,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    out[EVAL_LANE] = []
    out[TEST_LANE] = []

    for lane in LANES:
        gen = _GENERATORS[lane]
        for i in range(docs_per_lane):
            tier_a = lane in cfg.TIER_A_LANES and i % 3 == 0
            lang = "hi" if lane == "indic" else ("py" if lane == "code" else "en")
            text = gen(i)
            # shuffle-stable fluff
            if rng.random() < 0.3:
                text = text + f" Extra note {rng.randint(0, 999)}."
            out[lane].append(
                _doc(
                    f"{lane}_{i:04d}",
                    lane,
                    text,
                    tier="a" if tier_a else "b",
                    difficulty=f"B{(i % 6)}",
                    reasoning_length=("short", "medium", "long", "ultra")[i % 4],
                    license="mit" if lane == "code" else "cc0",
                    lang=lang,
                    tier_a=tier_a,
                )
            )

    for i in range(eval_docs):
        text = f"{CANARY_EVAL} held-out eval document {i}. Do not train on this."
        out[EVAL_LANE].append(
            _doc(f"eval_{i:04d}", EVAL_LANE, text, license="cc0", lang="en")
        )
    for i in range(test_docs):
        text = f"{CANARY_TEST} held-out test document {i}. Never gradient."
        out[TEST_LANE].append(
            _doc(f"test_{i:04d}", TEST_LANE, text, license="cc0", lang="en")
        )
    return out
