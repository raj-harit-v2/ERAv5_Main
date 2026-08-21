"""Phase 1 entrypoint: user sentence → attention → evidence."""

from __future__ import annotations

import config
from src.evidence_builder import write_evidence
from src.pipeline_user_sentence import UserSentencePipeline
from src.self_diagnosis import run_checks
from src.utils import ensure_dir, set_seed


def main() -> None:
    set_seed(config.SEED)
    ensure_dir(config.ARTIFACTS)
    print("=== Assign_008 Phase 1: when a user gives the AI a sentence ===")
    print(f"Example: {config.EXAMPLE_SENTENCE!r}")
    print(f"device={config.DEVICE} embed={config.EMBED_MODE} position={config.POSITION_POLICY}")
    print()

    pipe = UserSentencePipeline().to(config.DEVICE)
    pipe.eval()
    result = pipe.forward_sentence(config.EXAMPLE_SENTENCE)
    for step in result.steps:
        print(step)
    print()
    print(f"token_strings (trim): {result.token_strings[:12]} ...")
    print(f"hidden {tuple(result.hidden.shape)}  attn_out {tuple(result.attn_out.shape)}")
    print(f"logits {tuple(result.logits.shape)}  reply_sketch={result.reply_sketch!r}")
    print()

    checks = run_checks()
    for c in checks:
        print(f"[{'OK' if c.ok else 'FAIL'}] {c.id}: {c.detail}")
    if not all(c.ok for c in checks):
        raise SystemExit("self-diagnosis failed")

    write_evidence(result, checks)
    print()
    print(f"Wrote {config.ARTIFACTS / 'evidence.md'}")
    print("Widgets: Phase 2 embedded — open demo_08/coach_demo/index.html")


if __name__ == "__main__":
    main()
