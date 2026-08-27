"""Asgn_09_Self_diagionistics_01.py — Session 9 whole-code self-diagnostics.

Assign_005-style case ledger with provenance. Audits imports, shift/PPL/masks,
chunked CE, MTP, memory ratio, Curriculum PNGs, and pytest.

Run (from Assign_009_Init project root):
  .\\.venv\\Scripts\\python.exe tests\\Asgn_09_Self_diagionistics_01.py
  .\\.venv\\Scripts\\python.exe tests\\Asgn_09_Self_diagionistics_01.py --full
  .\\.venv\\Scripts\\python.exe tests\\Asgn_09_Self_diagionistics_01.py --no-pytest

Exit codes: 0 = all PASS, 1 = one or more FAIL, 2 = unexpected crash.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "data" / "evaluation" / "self_diagnostics_01.json"

PROVENANCE: dict[str, dict[str, str]] = {
    "ENV-01": {"file": "runtime", "what": "torch importable; device reported"},
    "IMP-01": {"file": "src/llm/tokenizer.py", "what": "import WordTokenizer"},
    "IMP-02": {"file": "src/llm/nano_lm.py", "what": "import NanoLM"},
    "IMP-03": {"file": "src/llm/output_head.py", "what": "import DenseOutputHead / LowRank / compare"},
    "IMP-04": {"file": "src/llm/chunked_ce.py", "what": "import ChunkedCrossEntropy"},
    "IMP-05": {"file": "src/llm/mtp_heads.py", "what": "import MTPDualHeads"},
    "IMP-06": {"file": "src/llm/stability.py", "what": "import soft_cap / z_loss / centering"},
    "IMP-07": {"file": "src/utils/shapes.py", "what": "import dump_batch_shapes"},
    "IMP-08": {"file": "src/utils/decode_strings.py", "what": "import print_shift_table"},
    "IMP-09": {"file": "src/utils/peak_vram.py", "what": "import estimate_logits_bytes"},
    "IMP-10": {"file": "src/pipeline/harness.py", "what": "import run_harness"},
    "IMP-11": {"file": "src/pipeline/memory_profile.py", "what": "import measure_full_vs_chunked"},
    "IMP-12": {"file": "src/pipeline/train_smoke.py", "what": "import run_mtp_smoke"},
    "IMP-13": {"file": "src/pipeline/plot_stats.py", "what": "import run_plots"},
    "IMP-14": {"file": "src/pipeline/run_all.py", "what": "import run_all main"},
    "TOK-01": {"file": "src/llm/tokenizer.py", "what": "build vocab from shakespeare_tiny.txt; V>1"},
    "SHF-01": {"file": "src/utils/decode_strings.py", "what": "shift string pairs consecutive"},
    "SHF-02": {"file": "src/llm/chunked_ce.py", "what": "logits[:,:-1] vs tokens[:,1:] shapes"},
    "PAD-01": {"file": "src/pipeline/harness.py", "what": "pad mask reduces contributing count"},
    "BND-01": {"file": "src/pipeline/harness.py", "what": "boundary mask drops contrib; loss sum changes"},
    "PPL-01": {"file": "src/llm/nano_lm.py", "what": "untrained |L-ln(V)|/ln(V) < 0.35"},
    "CE-01": {"file": "src/llm/chunked_ce.py", "what": "chunked vs full CE abs diff < 1e-5"},
    "MEM-01": {"file": "src/pipeline/memory_profile.py", "what": "full/chunked activation ratio >= 1.5 @ N=4096 C=1024"},
    "TIE-01": {"file": "src/llm/output_head.py", "what": "untied_params == 2 * tied_params"},
    "MTP-01": {"file": "src/llm/mtp_heads.py", "what": "Head2 targets tokens[:,2:]; losses finite"},
    "STB-01": {"file": "src/llm/stability.py", "what": "soft-cap bound; centering helper documented"},
    "ART-01": {"file": "data/evaluation/seven_numbers.json", "what": "required Part-1 keys present"},
    "ART-02": {"file": "reports/Curriculum_Required_stats_*.png", "what": "three PNGs exist size>0"},
    "ART-03": {"file": "data/evaluation/part2_losses.json", "what": "optional MTP artifact when --full"},
    "WID-01": {"file": "dist/", "what": "landing + 5 widgets + engine + JSON + PNGs after build_dist"},
    "PYT-01": {"file": "tests/", "what": "pytest -q exit code 0"},
}


@dataclass
class Case:
    id: str
    name: str
    ok: bool
    detail: str
    provenance: dict


def _case(cid: str, name: str, ok: bool, detail: str = "") -> Case:
    return Case(
        id=cid,
        name=name,
        ok=ok,
        detail=detail,
        provenance=PROVENANCE.get(cid, {}),
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


def _try_import(module: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        if attr is not None and not hasattr(mod, attr):
            return False, f"missing attr {attr} on {module}"
        return True, f"ok {module}" + (f".{attr}" if attr else "")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_diagnostics(
    run_pytest: bool = True,
    refresh_artifacts: bool = True,
    full: bool = False,
) -> dict:
    import torch
    import torch.nn.functional as F

    cases: list[Case] = []

    # --- ENV-01 ---
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cases.append(
            _case(
                "ENV-01",
                "torch_device",
                True,
                f"torch={torch.__version__} device={device}",
            )
        )
    except Exception as e:
        cases.append(_case("ENV-01", "torch_device", False, str(e)))

    # --- IMP-* ---
    import_specs = [
        ("IMP-01", "src.llm.tokenizer", "WordTokenizer"),
        ("IMP-02", "src.llm.nano_lm", "NanoLM"),
        ("IMP-03", "src.llm.output_head", "compare_tied_untied_counts"),
        ("IMP-04", "src.llm.chunked_ce", "ChunkedCrossEntropy"),
        ("IMP-05", "src.llm.mtp_heads", "MTPDualHeads"),
        ("IMP-06", "src.llm.stability", "soft_cap"),
        ("IMP-07", "src.utils.shapes", "dump_batch_shapes"),
        ("IMP-08", "src.utils.decode_strings", "verify_shift_strings"),
        ("IMP-09", "src.utils.peak_vram", "estimate_logits_bytes"),
        ("IMP-10", "src.pipeline.harness", "run_harness"),
        ("IMP-11", "src.pipeline.memory_profile", "measure_full_vs_chunked"),
        ("IMP-12", "src.pipeline.train_smoke", "run_mtp_smoke"),
        ("IMP-13", "src.pipeline.plot_stats", "run_plots"),
        ("IMP-14", "src.pipeline.run_all", "main"),
    ]
    for cid, mod, attr in import_specs:
        ok, detail = _try_import(mod, attr)
        cases.append(_case(cid, f"import_{attr}", ok, detail))

    # Early abort soft-continue: still run remaining if imports mostly ok
    from src.llm.chunked_ce import (
        ChunkedCrossEntropy,
        full_cross_entropy,
        shift_logits_and_targets,
    )
    from src.llm.mtp_heads import MTPDualHeads
    from src.llm.nano_lm import NanoLM
    from src.llm.output_head import compare_tied_untied_counts
    from src.llm.stability import (
        center_output_weights,
        explain_centering_vs_logz,
        soft_cap,
        z_loss_penalty,
    )
    from src.llm.tokenizer import WordTokenizer
    from src.pipeline.harness import build_loss_mask, load_tokenizer, pad_batch, run_harness
    from src.pipeline.plot_stats import run_plots
    from src.utils.decode_strings import print_shift_table, verify_shift_strings
    from src.utils.peak_vram import estimate_logits_bytes

    torch.manual_seed(0)

    # --- TOK-01 ---
    try:
        tok = load_tokenizer()
        ok = tok.vocab_size > 1 and (ROOT / "data" / "raw" / "shakespeare_tiny.txt").exists()
        cases.append(
            _case("TOK-01", "tokenizer_vocab", ok, f"V={tok.vocab_size} pad_id={tok.pad_id}")
        )
    except Exception as e:
        cases.append(_case("TOK-01", "tokenizer_vocab", False, str(e)))
        tok = None  # type: ignore

    if tok is None:
        report = _finalize(cases)
        return report

    # --- SHF-01 ---
    try:
        ids = tok.encode("First Citizen: Speak.", add_bos=True, add_eos=True)
        ok = verify_shift_strings(tok, ids)
        pairs = print_shift_table(tok, ids, max_rows=4)
        cases.append(
            _case(
                "SHF-01",
                "shift_strings",
                ok,
                f"pairs_sample={pairs[:3]}",
            )
        )
    except Exception as e:
        cases.append(_case("SHF-01", "shift_strings", False, str(e)))

    # --- SHF-02 ---
    try:
        model = NanoLM(
            vocab_size=tok.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq=64,
        )
        model.eval()
        tokens = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens)
        sl, st = shift_logits_and_targets(logits, tokens)
        ok = sl.shape[0] == st.shape[0] and sl.shape[1] == st.shape[1] == tokens.size(1) - 1
        cases.append(
            _case(
                "SHF-02",
                "shift_shapes",
                ok,
                f"logits_shift={list(sl.shape)} targets={list(st.shape)}",
            )
        )
    except Exception as e:
        cases.append(_case("SHF-02", "shift_shapes", False, str(e)))
        model = None  # type: ignore

    # --- PAD-01 ---
    try:
        short = tok.encode("Come", add_bos=True, add_eos=True)
        long = tok.encode("First Citizen: Speak speak.", add_bos=True, add_eos=True)
        batch = pad_batch([short, long], tok.pad_id)
        targets = batch[:, 1:]
        before = int(torch.ones_like(targets).sum().item())
        mask = build_loss_mask(batch, tok.pad_id)
        after = int(mask.sum().item())
        ok = after < before
        cases.append(
            _case("PAD-01", "pad_count_decreases", ok, f"before={before} after={after}")
        )
    except Exception as e:
        cases.append(_case("PAD-01", "pad_count_decreases", False, str(e)))

    # --- BND-01 ---
    try:
        doc_a = tok.encode("First Citizen: Speak.", add_bos=True, add_eos=True)
        doc_b = tok.encode("All: Away away.", add_bos=True, add_eos=True)
        packed = torch.tensor([doc_a + doc_b], dtype=torch.long)
        if model is None:
            raise RuntimeError("model unavailable")
        with torch.no_grad():
            logits_p = model(packed)
        lp, tp = shift_logits_and_targets(logits_p, packed)
        boundary_j = len(doc_a) - 1
        mask_no = build_loss_mask(packed, tok.pad_id)
        mask_b = build_loss_mask(packed, tok.pad_id, boundary_positions={(0, boundary_j)})
        bsz, tlen, vdim = lp.shape
        per = F.cross_entropy(
            lp.reshape(bsz * tlen, vdim),
            tp.reshape(bsz * tlen),
            reduction="none",
        ).view(bsz, tlen)
        sum_before = float((per * mask_no).sum())
        sum_after = float((per * mask_b).sum())
        c0, c1 = int(mask_no.sum().item()), int(mask_b.sum().item())
        ok = c1 < c0 and abs(sum_after - sum_before) > 1e-8
        cases.append(
            _case(
                "BND-01",
                "boundary_mask_sum",
                ok,
                f"contrib={c0}->{c1} sum={sum_before:.6f}->{sum_after:.6f}",
            )
        )
    except Exception as e:
        cases.append(_case("BND-01", "boundary_mask_sum", False, str(e)))

    # --- PPL-01 ---
    try:
        if model is None:
            raise RuntimeError("model unavailable")
        with torch.no_grad():
            logits = model(tokens)
        sl, st = shift_logits_and_targets(logits, tokens)
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), st.reshape(-1))
        loss_f = float(loss)
        ln_v = math.log(tok.vocab_size)
        rel = abs(loss_f - ln_v) / ln_v
        ok = rel < 0.35
        cases.append(
            _case(
                "PPL-01",
                "untrained_ppl_near_v",
                ok,
                f"L={loss_f:.4f} lnV={ln_v:.4f} rel={rel:.4f} ppl={math.exp(loss_f):.2f}",
            )
        )
    except Exception as e:
        cases.append(_case("PPL-01", "untrained_ppl_near_v", False, str(e)))

    # --- CE-01 ---
    try:
        n, v = 200, 50
        logits = torch.randn(n, v)
        targets = torch.randint(0, v, (n,))
        full = full_cross_entropy(logits, targets)
        chunked = ChunkedCrossEntropy(chunk_size=32)(logits, targets)
        diff = abs(float(full - chunked))
        ok = diff < 1e-5
        cases.append(_case("CE-01", "chunked_ce_parity", ok, f"abs_diff={diff:.2e}"))
    except Exception as e:
        cases.append(_case("CE-01", "chunked_ce_parity", False, str(e)))

    # --- MEM-01 ---
    try:
        n, c, v = 4096, 1024, tok.vocab_size
        full_b = estimate_logits_bytes(n, v, 4)
        chunk_b = estimate_logits_bytes(c, v, 4)
        ratio = full_b / max(chunk_b, 1)
        ok = ratio >= 1.5
        cases.append(
            _case(
                "MEM-01",
                "memory_ratio_estimate",
                ok,
                f"N={n} C={c} V={v} ratio={ratio:.4f} full_MiB={full_b/1024**2:.4f} chunk_MiB={chunk_b/1024**2:.4f}",
            )
        )
    except Exception as e:
        cases.append(_case("MEM-01", "memory_ratio_estimate", False, str(e)))

    # --- TIE-01 ---
    try:
        counts = compare_tied_untied_counts(d_model=64, vocab_size=tok.vocab_size)
        ok = counts["untied_params"] == 2 * counts["tied_params"]
        cases.append(
            _case(
                "TIE-01",
                "tied_vs_untied_counts",
                ok,
                f"tied={counts['tied_params']} untied={counts['untied_params']}",
            )
        )
    except Exception as e:
        cases.append(_case("TIE-01", "tied_vs_untied_counts", False, str(e)))

    # --- MTP-01 ---
    try:
        mtp = MTPDualHeads(d_model=64, vocab_size=tok.vocab_size)
        if model is None:
            raise RuntimeError("model unavailable")
        with torch.no_grad():
            _, hidden = model(tokens, return_hidden=True)
            losses = mtp.forward_losses(hidden, tokens, ignore_index=tok.pad_id)
        l1, l2, s = float(losses["loss1"]), float(losses["loss2"]), float(losses["loss_sum"])
        ok = (
            math.isfinite(l1)
            and math.isfinite(l2)
            and abs(s - (l1 + l2)) < 1e-5
            and losses["logits2"].shape[1] == tokens.size(1) - 2
        )
        cases.append(
            _case(
                "MTP-01",
                "mtp_k2_losses",
                ok,
                f"L1={l1:.4f} L2={l2:.4f} sum={s:.4f} logits2_T={losses['logits2'].shape[1]}",
            )
        )
    except Exception as e:
        cases.append(_case("MTP-01", "mtp_k2_losses", False, str(e)))

    # --- STB-01 ---
    try:
        z = torch.randn(2, 8, tok.vocab_size) * 50
        capped = soft_cap(z, c=30.0)
        ok_cap = bool(capped.abs().max() <= 30.0 + 1e-4)
        w = torch.randn(tok.vocab_size, 64)
        wc = center_output_weights(w)
        mean_abs = float(wc.mean(dim=0).abs().max())
        note = explain_centering_vs_logz()
        zl = float(z_loss_penalty(z[:, 0, :], lam=1e-4))
        ok = ok_cap and mean_abs < 1e-5 and len(note) > 20 and math.isfinite(zl)
        cases.append(
            _case(
                "STB-01",
                "stability_helpers",
                ok,
                f"cap_ok={ok_cap} center_mean_abs={mean_abs:.2e} zloss={zl:.6f}",
            )
        )
    except Exception as e:
        cases.append(_case("STB-01", "stability_helpers", False, str(e)))

    # --- Refresh artifacts ---
    if refresh_artifacts:
        try:
            _log("--- refresh: run_harness ---")
            run_harness(device="cpu", seed=0)
            _log("--- refresh: run_plots ---")
            run_plots()
            from src.pipeline.export_widget import run_export

            _log("--- refresh: export_widget ---")
            run_export()
            from tests.build_dist import build

            _log("--- refresh: build_dist ---")
            build()
            if full:
                from src.pipeline.train_smoke import run_mtp_smoke

                _log("--- refresh: run_mtp_smoke(steps=5) ---")
                run_mtp_smoke(steps=5, device="cpu", seed=1)
        except Exception as e:
            _log(f"artifact refresh error: {e}")
            traceback.print_exc()

    # --- ART-01 ---
    try:
        path = ROOT / "data" / "evaluation" / "seven_numbers.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        required = [
            "shapes_ok",
            "shift_strings_ok",
            "pad_count_before",
            "pad_count_after",
            "boundary_loss_before",
            "boundary_loss_after",
            "ppl0",
            "loss0",
            "tied_params",
            "untied_params",
            "peak_full_mib",
            "peak_chunked_mib",
            "ratio_full_over_chunked",
        ]
        missing = [k for k in required if k not in data]
        ok = path.exists() and not missing and data.get("ratio_full_over_chunked", 0) >= 1.5
        cases.append(
            _case(
                "ART-01",
                "seven_numbers_json",
                ok,
                f"missing={missing} ratio={data.get('ratio_full_over_chunked')}",
            )
        )
    except Exception as e:
        cases.append(_case("ART-01", "seven_numbers_json", False, str(e)))

    # --- ART-02 ---
    try:
        names = [
            "Curriculum_Required_stats_01_peak_vram.png",
            "Curriculum_Required_stats_02_chunk_frontier.png",
            "Curriculum_Required_stats_03_swiglu.png",
        ]
        infos = []
        ok = True
        for n in names:
            p = ROOT / "reports" / n
            sz = p.stat().st_size if p.exists() else 0
            infos.append(f"{n}:{sz}")
            if sz <= 0:
                ok = False
        cases.append(_case("ART-02", "curriculum_pngs", ok, "; ".join(infos)))
    except Exception as e:
        cases.append(_case("ART-02", "curriculum_pngs", False, str(e)))

    # --- ART-03 (informational if not --full; required when full) ---
    try:
        p2 = ROOT / "data" / "evaluation" / "part2_losses.json"
        if p2.exists():
            d2 = json.loads(p2.read_text(encoding="utf-8"))
            ok = "L1_final" in d2 and "L2_final" in d2
            cases.append(
                _case(
                    "ART-03",
                    "part2_losses_json",
                    ok,
                    f"L1={d2.get('L1_final')} L2={d2.get('L2_final')}",
                )
            )
        else:
            # Not a hard fail unless --full requested
            cases.append(
                _case(
                    "ART-03",
                    "part2_losses_json",
                    not full,
                    "missing (ok unless --full); run with --full to require",
                )
            )
    except Exception as e:
        cases.append(_case("ART-03", "part2_losses_json", False, str(e)))

    # --- WID-01 ---
    try:
        dist = ROOT / "dist"
        widget_names = [
            "s9_widget_0_loss_flow.html",
            "s9_widget_1_chunk_ce.html",
            "s9_widget_2_swiglu.html",
            "s9_widget_3_mtp.html",
            "s9_widget_4_stats_board.html",
            "s9_engine.js",
            "s9_boot.js",
            "s9_swiglu_diagram.js",
        ]
        json_names = [
            "seven_numbers.json",
            "part2_losses.json",
            "self_diagnostics_01.json",
            "widget_weights.json",
        ]
        png_names = [
            "Curriculum_Required_stats_01_peak_vram.png",
            "Curriculum_Required_stats_02_chunk_frontier.png",
            "Curriculum_Required_stats_03_swiglu.png",
        ]
        missing = []
        if not (dist / "index.html").is_file():
            missing.append("dist/index.html")
        for w in widget_names:
            if not (dist / "widgets" / w).is_file():
                missing.append(f"widgets/{w}")
        for j in json_names:
            if not (dist / "data" / j).is_file():
                missing.append(f"data/{j}")
        for p in png_names:
            if not (dist / "reports" / p).is_file():
                missing.append(f"reports/{p}")
        wsize = (dist / "data" / "widget_weights.json").stat().st_size if (
            dist / "data" / "widget_weights.json"
        ).is_file() else 0
        ok = not missing and wsize > 1000
        cases.append(
            _case(
                "WID-01",
                "dist_widget_suite",
                ok,
                f"missing={missing} widget_weights_bytes={wsize}",
            )
        )
    except Exception as e:
        cases.append(_case("WID-01", "dist_widget_suite", False, str(e)))

    # --- PYT-01 ---
    if run_pytest:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            ok = proc.returncode == 0
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
            cases.append(
                _case(
                    "PYT-01",
                    "pytest_suite",
                    ok,
                    f"rc={proc.returncode} {' | '.join(tail)}",
                )
            )
        except Exception as e:
            cases.append(_case("PYT-01", "pytest_suite", False, str(e)))
    else:
        cases.append(_case("PYT-01", "pytest_suite", True, "skipped (--no-pytest)"))

    return _finalize(cases)


def _finalize(cases: list[Case]) -> dict:
    n_pass = sum(1 for c in cases if c.ok)
    n_fail = sum(1 for c in cases if not c.ok)
    report = {
        "ok": n_fail == 0,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "cases": [asdict(c) for c in cases],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Session 9 self-diagnostics")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run short MTP smoke and require part2_losses.json",
    )
    parser.add_argument("--no-pytest", action="store_true", help="Skip PYT-01")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not re-run harness/plots (use existing artifacts)",
    )
    args = parser.parse_args()

    _log("=== Asgn_09_Self_diagionistics_01.py ===")
    try:
        report = run_diagnostics(
            run_pytest=not args.no_pytest,
            refresh_artifacts=not args.no_refresh,
            full=args.full,
        )
    except Exception:
        traceback.print_exc()
        return 2

    for c in report["cases"]:
        status = "PASS" if c["ok"] else "FAIL"
        prov = c.get("provenance") or {}
        _log(
            f"[{status}] {c['id']} {c['name']}: {c['detail']} "
            f"| {prov.get('file', '')} — {prov.get('what', '')}"
        )
    _log(f"passed={report['n_pass']} failed={report['n_fail']}")
    _log(f"wrote {OUT_JSON}")
    if not report["ok"]:
        _log("DIAGNOSTICS FAILED")
        return 1
    _log("DIAGNOSTICS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
