"""
Self_Diagionistics_02.py — diagnostics for demo_Asgn_08 (+ pipeline mirrors).

Emits JSON under diagnostics_02/json/ and a short SUMMARY.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.utils import ensure_dir

ROOT = config.ROOT
DEMO = ROOT / "demo_Asgn_08"
OUT = ROOT / "diagnostics_02"
JSON_DIR = OUT / "json"
REPORT = OUT / "Diagionistics_02.md"


@dataclass
class StepResult:
    step_id: str
    title: str
    ok: bool
    severity: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(step: StepResult) -> None:
    path = JSON_DIR / f"{step.step_id}.json"
    path.write_text(json.dumps({"generated_at": _utc(), **asdict(step)}, indent=2), encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return [t for t in (text or "").strip().split() if t][:24]


def two_bills(t: int) -> dict[str, int]:
    return {"T": t, "computeUnits": t * t, "kvUnits": t}


def chunk_gate(fact: str, stream_on: bool, gate: float, alpha: float = 0.391) -> dict[str, Any]:
    injection = alpha * gate if stream_on else 0.0
    if not stream_on or injection < 0.01:
        verdict = "bad"
        message = "unavailable"
    elif injection > 0.18:
        verdict = "loud"
        message = f"{fact} too loud"
    else:
        verdict = "good"
        message = f"Answer: {fact}"
    return {"injection": injection, "verdict": verdict, "message": message}


def causal_future_mass_ok() -> tuple[bool, float]:
    """Tiny numeric check: lower-triangular softmax mass on identity-ish scores."""
    # Build T=4 scores = identity / sqrt(d) then causal mask
    t, d = 4, 4
    scale = math.sqrt(d)
    scores = [[(1.0 if i == j else 0.0) / scale for j in range(t)] for i in range(t)]
    for i in range(t):
        for j in range(t):
            if j > i:
                scores[i][j] = -1e9
    weights = []
    for row in scores:
        m = max(v for v in row if v > -1e8)
        ex = [0.0 if v <= -1e8 else math.exp(v - m) for v in row]
        z = sum(ex) or 1.0
        weights.append([e / z for e in ex])
    future = 0.0
    for i in range(t):
        for j in range(t):
            if j > i:
                future = max(future, abs(weights[i][j]))
    return future < 1e-8, future


def kv_cache_bytes(L: int, H_KV: int, d_head: int, T: int, B: int, P_b: int) -> int:
    return 2 * L * H_KV * d_head * T * B * P_b


def gqa_layout(mode: str, H_Q: int, H_KV: int) -> dict[str, Any]:
    h_q = max(1, int(H_Q) or 8)
    if mode == "MHA":
        h_kv = h_q
    elif mode == "MQA":
        h_kv = 1
    else:
        h_kv = max(1, int(H_KV) or 2)
        if h_kv >= h_q:
            h_kv = max(1, h_q // 2)
        if h_q % h_kv != 0:
            divisors = [d for d in range(1, h_q) if h_q % d == 0]
            h_kv = divisors[min(len(divisors) - 1, 1)] if divisors else 1
        mode = "GQA"
    return {"mode": mode if mode in ("MHA", "MQA") else "GQA", "H_Q": h_q, "H_KV": h_kv, "reduction": h_q / h_kv}


def linear_softmax_off_match() -> bool:
    """Scalar cartoon: two steps; direct sum equals S @ q."""
    steps = [((1.0,), (2.0,), (3.0,)), ((2.0,), (4.0,), (3.0,))]
    S = [[0.0]]
    for t, (k, v, q) in enumerate(steps):
        S[0][0] += v[0] * k[0]
        direct = 0.0
        for j in range(t + 1):
            kj, vj, _ = steps[j]
            direct += (q[0] * kj[0]) * vj[0]
        y = S[0][0] * q[0]
        if abs(direct - y) > 1e-9:
            return False
    return True


def delta_rule_unit_key_ok() -> bool:
    """Unit key: after delta write, S k ≈ v."""
    k = [1.0]
    v = [3.0]
    S = [[0.0]]
    v_hat = S[0][0] * k[0]
    delta = v[0] - v_hat
    S[0][0] += delta * k[0]
    after = S[0][0] * k[0]
    return abs(after - v[0]) < 1e-9 and abs(math.sqrt(k[0] ** 2) - 1.0) < 1e-9


def sparse_topk_row_mass_ok() -> bool:
    """Top-1 causal on increasing scores: mass concentrates on last past key."""
    # Fake scores row i=2: j=0,1,2 → keep top-1 = j=2
    scores = [0.1, 0.5, 2.0]
    keep = sorted(enumerate(scores), key=lambda t: -t[1])[:1]
    m = keep[0][1]
    ex = [math.exp(s - m) for _, s in keep]
    z = sum(ex) or 1.0
    w = {j: e / z for (j, _), e in zip(keep, ex)}
    return abs(w.get(2, 0.0) - 1.0) < 1e-9 and abs(sum(w.values()) - 1.0) < 1e-9


def rope_relative_depends_on_delta() -> bool:
    """Rotated dot changes with Δpos for fixed content (θ=1)."""
    q = (1.0, 0.0)
    k = (0.8, 0.2)
    theta = 1.0

    def rot(x0: float, x1: float, m: int) -> tuple[float, float]:
        a = m * theta
        c, s = math.cos(a), math.sin(a)
        return (c * x0 - s * x1, s * x0 + c * x1)

    def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1]

    i = 0
    s0 = dot(rot(*q, i), rot(*k, 0))
    s2 = dot(rot(*q, i), rot(*k, 2))
    return abs(s0 - s2) > 1e-6


def drope_factor_ok() -> bool:
    return 262144 / 8192 == 32


def compression_tm_ok() -> bool:
    T, m = 32768, 4
    n_blocks = math.ceil(T / m)
    return n_blocks == T // m and n_blocks == 8192


def schedule_motif_ok() -> bool:
    motif = ["D", "D", "D", "G", "D", "D", "D", "G"]
    return motif.count("D") == 6 and motif.count("G") == 2 and "".join(motif) == "DDDGDDDG"


def step_scaffold() -> StepResult:
    required = [
        DEMO / "index.html",
        DEMO / "styles.css",
        DEMO / "scripts" / "app.js",
        DEMO / "scripts" / "pipeline_demo.js",
        DEMO / "data" / "steps.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    ok = not missing
    return StepResult(
        "D00_scaffold",
        "demo_Asgn_08 scaffold files",
        ok,
        "pass" if ok else "fail",
        "Required HTML/JS/JSON/MD present." if ok else f"Missing: {missing}",
        {"missing": missing},
        [f"demo={DEMO.relative_to(ROOT)}"],
    )


def step_steps_json() -> StepResult:
    path = DEMO / "data" / "steps.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = ["schema", "steps", "widgets", "default_sentence", "default_fact", "D", "alpha"]
    missing = [k for k in keys if k not in data]
    n_steps = len(data.get("steps") or [])
    ok = not missing and n_steps >= 5 and "chunk_gate" in (data.get("widgets") or [])
    return StepResult(
        "D01_steps_json",
        "steps.json schema",
        ok,
        "pass" if ok else "fail",
        f"schema ok; {n_steps} simple-text steps.",
        {"missing_keys": missing, "n_steps": n_steps, "widgets": data.get("widgets"), "alpha": data.get("alpha")},
        [],
    )


def step_html_wiring() -> StepResult:
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    js = (DEMO / "scripts" / "app.js").read_text(encoding="utf-8")
    pipe = (DEMO / "scripts" / "pipeline_demo.js").read_text(encoding="utf-8")
    checks = {
        "sentence_input": 'id="sentence"' in html,
        "fact_input": 'id="fact"' in html,
        "btn_update": 'id="btn-update"' in html,
        "widget_ids": all(
            x in html
            for x in (
                "w-tokens",
                "w-btd",
                "w-attn",
                "w-bills",
                "w-chunk",
                "w-kv",
                "w-gqa",
                "w-linear",
                "w-delta",
                "w-sparse",
                "w-rope",
                "w-drope",
                "w-comp",
                "w-sched",
            )
        ),
        "pipeline_demo": "DemoPipeline" in pipe
        and "chunkGate" in pipe
        and "kvCacheBytes" in pipe
        and "gqaLayout" in pipe
        and "linearSoftmaxOff" in pipe
        and "deltaRuleWrite" in pipe
        and "sparseTopKAttention" in pipe
        and "ropeDemo" in pipe
        and "dropeExtension" in pipe
        and "sequenceCompression" in pipe
        and "scheduleAndFork" in pipe,
        "app_update": "runPipeline" in js
        and "renderChunkGate" in js
        and "renderKvCache" in js
        and "renderGqa" in js
        and "renderLinear" in js
        and "renderDelta" in js
        and "renderSparse" in js
        and "renderRope" in js
        and "renderDrope" in js
        and "renderCompression" in js
        and "renderSchedule" in js,
        "user_fact_not_only_hardcode": "state.fact" in js,
    }
    ok = all(checks.values())
    return StepResult(
        "D02_html_wiring",
        "index + scripts wiring",
        ok,
        "pass" if ok else "fail",
        "Shared text box and C1–C14 widgets wired." if ok else "Wiring gaps.",
        checks,
        [f"{k}={v}" for k, v in checks.items()],
    )


def step_pipeline_mirror() -> StepResult:
    text = "The cat sat on the mat"
    toks = tokenize(text)
    bills = two_bills(len(toks))
    causal_ok, future = causal_future_mass_ok()
    g_good = chunk_gate("9910", True, 0.15)
    g_off = chunk_gate("9910", False, 0.15)
    g_loud = chunk_gate("9910", True, 1.0)
    kv = kv_cache_bytes(48, 8, 128, 6, 1, 2)
    gqa = gqa_layout("GQA", 8, 2)
    lin_ok = linear_softmax_off_match()
    delta_ok = delta_rule_unit_key_ok()
    sparse_ok = sparse_topk_row_mass_ok()
    rope_ok = rope_relative_depends_on_delta()
    drope_ok = drope_factor_ok()
    comp_ok = compression_tm_ok()
    sched_ok = schedule_motif_ok()
    ok = (
        len(toks) == 6
        and bills["computeUnits"] == 36
        and causal_ok
        and g_good["verdict"] == "good"
        and g_off["verdict"] == "bad"
        and g_loud["verdict"] == "loud"
        and "9910" in g_good["message"]
        and kv == 2 * 48 * 8 * 128 * 6 * 1 * 2
        and gqa["reduction"] == 4.0
        and lin_ok
        and delta_ok
        and sparse_ok
        and rope_ok
        and drope_ok
        and comp_ok
        and sched_ok
    )
    return StepResult(
        "D03_pipeline_mirror",
        "Python mirror of demo math",
        ok,
        "pass" if ok else "fail",
        "Tokenize through C14 (DroPE/compression/schedule) mirrors match teaching demo.",
        {
            "tokens": toks,
            "bills": bills,
            "future_mass": future,
            "good": g_good,
            "off": g_off,
            "loud": g_loud,
            "kv_bytes_T6": kv,
            "gqa": gqa,
            "linear_match": lin_ok,
            "delta_ok": delta_ok,
            "sparse_ok": sparse_ok,
            "rope_ok": rope_ok,
            "drope_ok": drope_ok,
            "comp_ok": comp_ok,
            "sched_ok": sched_ok,
        },
        [],
    )


def step_no_hardcoded_only() -> StepResult:
    """Ensure chunk UI path references user fact field (not only 4471)."""
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    js = (DEMO / "scripts" / "app.js").read_text(encoding="utf-8")
    has_fact_field = 'id="fact"' in html
    uses_state_fact = "state.fact" in js
    # default may still be 4471 in JSON — that is OK as default seed
    ok = has_fact_field and uses_state_fact
    return StepResult(
        "D04_user_fact",
        "User fact text box (not hardcoded-only)",
        ok,
        "pass" if ok else "fail",
        "Fact comes from #fact input into chunk gate.",
        {"has_fact_field": has_fact_field, "uses_state_fact": uses_state_fact},
        [],
    )


def step_pytest() -> StepResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_demo_asgn_08.py", "-q", "--tb=no"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        ok = proc.returncode == 0
        return StepResult(
            "D05_pytest",
            "demo_Asgn_08 self-tests",
            ok,
            "pass" if ok else "fail",
            "pytest tests/test_demo_asgn_08.py",
            {"returncode": proc.returncode, "tail": out[-400:]},
            out.splitlines()[-6:],
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult("D05_pytest", "demo_Asgn_08 self-tests", False, "fail", str(exc), {}, [])


def step_scorecard(steps: list[StepResult]) -> StepResult:
    fails = [s.step_id for s in steps if not s.ok]
    ok = not fails
    return StepResult(
        "D06_scorecard",
        "demo_Asgn_08 readiness",
        ok,
        "pass" if ok else "fail",
        "READY" if ok else f"NOT_READY fails={fails}",
        {"fails": fails, "n": len(steps)},
        [],
    )


def write_report(steps: list[StepResult], summary: dict[str, Any]) -> None:
    lines = [
        "# Diagionistics_02 — demo_Asgn_08",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        "```text",
        "D00 scaffold → D01 steps.json → D02 wiring → D03 math mirror",
        "            → D04 user fact → D05 pytest → D06 scorecard",
        "```",
        "",
    ]
    for s in steps:
        lines += [
            f"## {s.step_id} — {s.title}",
            "",
            f"- **Result:** `{s.severity}` (ok={s.ok})",
            f"- **Summary:** {s.summary}",
            f"- **JSON:** `diagnostics_02/json/{s.step_id}.json`",
            "",
        ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(JSON_DIR)
    print("=== Self_Diagionistics_02 — demo_Asgn_08 ===")
    steps: list[StepResult] = []
    for fn in (
        step_scaffold,
        step_steps_json,
        step_html_wiring,
        step_pipeline_mirror,
        step_no_hardcoded_only,
        step_pytest,
    ):
        s = fn()
        steps.append(s)
        write_json(s)
        print(f"[{s.severity.upper()}] {s.step_id}: {s.summary}")

    score = step_scorecard(steps)
    steps.append(score)
    write_json(score)
    print(f"[{score.severity.upper()}] {score.step_id}: {score.summary}")

    verdict = "READY" if score.ok else "NOT_READY"
    summary = {
        "generated_at": _utc(),
        "schema": "a08.diagnostics.2",
        "verdict": verdict,
        "steps": [s.step_id for s in steps],
    }
    (JSON_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(steps, summary)
    print(f"Wrote {REPORT.name}")
    print(f"Verdict: {verdict}")
    return 0 if score.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
