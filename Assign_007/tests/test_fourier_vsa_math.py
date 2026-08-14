"""Math role-filler VSA tests (8+8 / 8*8 / 8/8) — Problem #4 verifier, not Problem #1."""
from __future__ import annotations

from src.fourier_vsa_bridge import FourierVSABridge
from src.fourier_vsa_tests import run_s3_math


def test_math_role_suite_critical_passes():
    bridge = FourierVSABridge()
    cases = run_s3_math(bridge)
    critical = [c for c in cases if c.get("critical", True)]
    failed = [c["name"] for c in critical if not c["pass"]]
    assert not failed, f"critical math cases failed: {failed}"
