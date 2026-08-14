"""Staged Fourier + HRR/VSA verification suites (S0–S7) with negative controls.

Problem #4 verification layer — not Problem #1 arithmetic claim, not Problem #5 VQ.
"""
from __future__ import annotations

from typing import Any

import torch

import config as cfg
from src.embeddings import KroneckerByteEmbedding
from src.fourier_vsa_bridge import FourierVSABridge
from src.hrr import bind, bundle, cosine, dropout_noise, normalize, random_hv, unbind, unbind_div
from src.vsa_symbols import DIGITS, EMOJI, LEXICAL, MEDIA, OPS, ROLES


def _case(
    stage: str,
    name: str,
    ok: bool,
    detail: str,
    *,
    metric: float | None = None,
    threshold: float | None = None,
    negative: bool = False,
    critical: bool = True,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "name": name,
        "pass": bool(ok),
        "detail": detail,
        "metric": metric,
        "threshold": threshold,
        "negative": negative,
        "critical": critical,
    }


def _nearest(query: torch.Tensor, candidates: dict[str, torch.Tensor]) -> tuple[str, float]:
    best_name, best_c = "", -2.0
    for n, v in candidates.items():
        c = cosine(query, v)
        if c > best_c:
            best_name, best_c = n, c
    return best_name, best_c


def run_s0_hrr_axioms(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    d = bridge.hrr_dim
    cases: list[dict[str, Any]] = []
    a, b = random_hv(d, seed=1), random_hv(d, seed=2)
    bound = bind(a, b)
    cases.append(_case("S0", "dim_invariance", bound.shape == a.shape, f"shape={tuple(bound.shape)}"))

    retrieved = unbind(bound, a)
    c_match = cosine(retrieved, b)
    cases.append(
        _case("S0", "identity_unbind", c_match > 0.99, f"cos={c_match:.4f}", metric=c_match, threshold=0.99)
    )

    # orthogonality
    vecs = [random_hv(d, seed=100 + i) for i in range(40)]
    dots = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dots.append(abs(cosine(vecs[i], vecs[j])))
    mean_abs = sum(dots) / max(len(dots), 1)
    thr = 0.08 if d >= 2048 else 0.12
    cases.append(
        _case("S0", "orthogonality_mean_abs_cos", mean_abs < thr, f"mean|cos|={mean_abs:.4f}", metric=mean_abs, threshold=thr)
    )

    # Neg: zero vector
    zero = torch.zeros(d)
    zbound = bind(a, zero)
    znorm = normalize(zbound)
    cases.append(
        _case(
            "S0",
            "zero_bind_no_nan",
            bool(torch.isfinite(znorm).all()),
            f"norm_finite={bool(torch.isfinite(znorm).all())} cos_to_a={cosine(zbound, a):.4f}",
            negative=True,
        )
    )

    # Neg: conjugate stable; division may be unstable but finite
    conj_c = cosine(unbind(bound, a), b)
    div_v = unbind_div(bound, a)
    cases.append(
        _case(
            "S0",
            "conjugate_preferred_over_division",
            conj_c > 0.95 and bool(torch.isfinite(div_v).all()),
            f"conj_cos={conj_c:.4f} div_finite={bool(torch.isfinite(div_v).all())}",
            negative=True,
            metric=conj_c,
            threshold=0.95,
        )
    )
    return cases


def run_s1_lexical_emoji(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    va = bridge.from_spec(LEXICAL["a"])
    vb = bridge.from_spec(LEXICAL["b"])
    vc = bridge.from_spec(LEXICAL["c"])
    apple_e = bridge.from_spec(EMOJI["apple_icon"])
    fruit = bridge.from_spec(LEXICAL["Fruit"])
    mango_e = bridge.from_spec(EMOJI["mango_icon"])

    bound = bind(va, vb)
    ret = unbind(bound, va)
    c_b, c_c, c_e = cosine(ret, vb), cosine(ret, vc), cosine(ret, apple_e)
    cases.append(
        _case(
            "S1",
            "unbind_a_retrieves_b_not_c_or_emoji",
            c_b > c_c and c_b > c_e and c_b > 0.5,
            f"cos_b={c_b:.3f} cos_c={c_c:.3f} cos_emoji={c_e:.3f}",
            metric=c_b,
            threshold=0.5,
        )
    )

    mem = bind(apple_e, fruit)
    ret_f = unbind(mem, apple_e)
    cases.append(
        _case(
            "S1",
            "emoji_apple_binds_fruit_over_mango_icon",
            cosine(ret_f, fruit) > cosine(ret_f, mango_e),
            f"fruit={cosine(ret_f, fruit):.3f} mango_icon={cosine(ret_f, mango_e):.3f}",
            metric=cosine(ret_f, fruit),
            threshold=0.3,
        )
    )

    # Neg: wrong language role
    eng = bridge.from_spec(ROLES["ENGLISH"], prefer_fourier_text=False)
    french = bridge.from_spec(ROLES["FRENCH"], prefer_fourier_text=False)
    apple = bridge.from_spec(LEXICAL["Apple"])
    mem_en = bind(eng, apple)
    wrong = unbind(mem_en, french)
    cases.append(
        _case(
            "S1",
            "neg_wrong_role_french_on_english_memory",
            cosine(wrong, apple) < 0.25,
            f"cos_apple={cosine(wrong, apple):.3f}",
            metric=cosine(wrong, apple),
            threshold=0.25,
            negative=True,
        )
    )

    # Neg: anagram distinguishable in Fourier space before bind
    ab = bridge.from_text("अब")
    ba = bridge.from_text("बा")
    cases.append(
        _case(
            "S1",
            "neg_fourier_anagram_codes_differ",
            cosine(ab, ba) < 0.99,
            f"cos={cosine(ab, ba):.4f}",
            metric=cosine(ab, ba),
            threshold=0.99,
            negative=True,
        )
    )
    return cases


def run_s2_abugida(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    ka = bridge.from_spec(LEXICAL["क"])
    kaa = bridge.from_spec(LEXICAL["का"])
    ksha = bridge.from_spec(LEXICAL["क्ष"])
    cases.append(
        _case(
            "S2",
            "abugida_codepoints_differ",
            cosine(ka, kaa) < 0.999 and cosine(ka, ksha) < 0.999,
            f"ka-kaa={cosine(ka, kaa):.4f} ka-ksha={cosine(ka, ksha):.4f}",
        )
    )

    hi = bridge.from_spec(ROLES["SCRIPT_HI"], prefer_fourier_text=False)
    ram = bridge.from_spec(LEXICAL["राम"])
    mem = bind(hi, ram)
    ret = unbind(mem, hi)
    cases.append(
        _case(
            "S2",
            "script_hi_binds_ram",
            cosine(ret, ram) > 0.9,
            f"cos={cosine(ret, ram):.4f}",
            metric=cosine(ret, ram),
            threshold=0.9,
        )
    )

    long_a = bridge.from_spec(LEXICAL["अंतर्राष्ट्रीयकरण"])
    long_b = bridge.from_spec(LEXICAL["अंतर्राष्ट्रीयता"])
    lemma = bridge.from_spec(ROLES["LEMMA"], prefer_fourier_text=False)
    m = bundle(bind(lemma, long_a), bind(bridge.from_spec(ROLES["ICON"], prefer_fourier_text=False), long_b))
    # retrieve lemma binding
    r = unbind(m, lemma)
    cases.append(
        _case(
            "S2",
            "long_hindi_lookalikes_separable",
            cosine(long_a, long_b) < 0.999 and cosine(r, long_a) > cosine(r, long_b),
            f"code_cos={cosine(long_a, long_b):.4f} retr_a={cosine(r, long_a):.3f} retr_b={cosine(r, long_b):.3f}",
        )
    )

    # Neg: Kronecker@32 may collide — document baseline failure mode
    kron = KroneckerByteEmbedding(d_model=8, pos_dim=32)
    # Use long Hindi strings; may or may not collide — check difference
    k1 = kron.encode_string("अंतर्राष्ट्रीयकरण")
    k2 = kron.encode_string("अंतर्राष्ट्रीयता")
    kron_same = bool(torch.allclose(k1, k2))
    cases.append(
        _case(
            "S2",
            "neg_kronecker32_collision_diagnostic",
            True,  # informational always pass; detail records collision
            f"kron32_identical={kron_same} (known Indic byte-window risk)",
            negative=True,
            critical=False,
        )
    )

    # Neg: aggressive truncate hurts long token identity
    full = bridge.from_text("अंतर्राष्ट्रीयकरण")
    trunc = bridge.from_text_truncated("अंतर्राष्ट्रीयकरण", max_chars=3)
    cases.append(
        _case(
            "S2",
            "neg_aggressive_truncate_changes_code",
            cosine(full, trunc) < 0.999,
            f"cos_full_vs_trunc3={cosine(full, trunc):.4f}",
            metric=cosine(full, trunc),
            negative=True,
        )
    )
    return cases


def _math_expr(bridge: FourierVSABridge, left: str, right: str, op: str, result: str) -> torch.Tensor:
    return bundle(
        bind(bridge.from_spec(ROLES["LEFT"], prefer_fourier_text=False), bridge.from_spec(DIGITS[left], prefer_fourier_text=False)),
        bind(bridge.from_spec(ROLES["RIGHT"], prefer_fourier_text=False), bridge.from_spec(DIGITS[right], prefer_fourier_text=False)),
        bind(bridge.from_spec(ROLES["OP"], prefer_fourier_text=False), bridge.from_spec(OPS[op], prefer_fourier_text=False)),
        bind(bridge.from_spec(ROLES["RESULT"], prefer_fourier_text=False), bridge.from_spec(DIGITS[result], prefer_fourier_text=False)),
    )


def run_s3_math(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    add_m = _math_expr(bridge, "8", "8", "ADD", "16")
    mul_m = _math_expr(bridge, "8", "8", "MUL", "64")
    div_m = _math_expr(bridge, "8", "8", "DIV", "1")

    op_role = bridge.from_spec(ROLES["OP"], prefer_fourier_text=False)
    res_role = bridge.from_spec(ROLES["RESULT"], prefer_fourier_text=False)
    add_v = bridge.from_spec(OPS["ADD"], prefer_fourier_text=False)
    mul_v = bridge.from_spec(OPS["MUL"], prefer_fourier_text=False)
    d16 = bridge.from_spec(DIGITS["16"], prefer_fourier_text=False)
    d64 = bridge.from_spec(DIGITS["64"], prefer_fourier_text=False)
    d1 = bridge.from_spec(DIGITS["1"], prefer_fourier_text=False)

    ret_op = unbind(add_m, op_role)
    cases.append(
        _case(
            "S3",
            "math_8plus8_op_is_add_not_mul",
            cosine(ret_op, add_v) > cosine(ret_op, mul_v),
            f"add={cosine(ret_op, add_v):.3f} mul={cosine(ret_op, mul_v):.3f}",
            metric=cosine(ret_op, add_v),
        )
    )

    ret_mul_res = unbind(mul_m, res_role)
    cases.append(
        _case(
            "S3",
            "math_8mul8_result_64",
            cosine(ret_mul_res, d64) > cosine(ret_mul_res, d16) and cosine(ret_mul_res, d64) > cosine(ret_mul_res, d1),
            f"64={cosine(ret_mul_res, d64):.3f} 16={cosine(ret_mul_res, d16):.3f} 1={cosine(ret_mul_res, d1):.3f}",
            metric=cosine(ret_mul_res, d64),
        )
    )

    ret_div = unbind(div_m, res_role)
    cases.append(
        _case(
            "S3",
            "math_8div8_result_1",
            cosine(ret_div, d1) > cosine(ret_div, d16),
            f"1={cosine(ret_div, d1):.3f} 16={cosine(ret_div, d16):.3f}",
            metric=cosine(ret_div, d1),
        )
    )

    # Neg: wrong op contamination
    wrong = unbind(add_m, op_role)
    # asking whether MUL "result" appears — use querying mul memory pattern wrongly
    cases.append(
        _case(
            "S3",
            "neg_wrong_op_mul_not_preferred_on_add_memory",
            cosine(wrong, mul_v) < cosine(wrong, add_v),
            f"mul={cosine(wrong, mul_v):.3f} add={cosine(wrong, add_v):.3f}",
            negative=True,
        )
    )

    # Neg: SUB non-commutative roles — 8-3 vs 3-8
    sub_a = _math_expr(bridge, "8", "3", "SUB", "5")
    left = bridge.from_spec(ROLES["LEFT"], prefer_fourier_text=False)
    d8 = bridge.from_spec(DIGITS["8"], prefer_fourier_text=False)
    d3 = bridge.from_spec(DIGITS["3"], prefer_fourier_text=False)
    left_ret = unbind(sub_a, left)
    cases.append(
        _case(
            "S3",
            "neg_sub_roles_left_is_8_not_3",
            cosine(left_ret, d8) > cosine(left_ret, d3),
            f"left8={cosine(left_ret, d8):.3f} left3={cosine(left_ret, d3):.3f}",
            negative=True,
        )
    )

    # Neg: zero digit
    zero_d = torch.zeros(bridge.hrr_dim)
    zbind = bind(bridge.from_spec(ROLES["LEFT"], prefer_fourier_text=False), zero_d)
    cases.append(
        _case(
            "S3",
            "neg_zero_digit_bind_finite",
            bool(torch.isfinite(normalize(zbind)).all()),
            "finite_after_normalize",
            negative=True,
        )
    )
    return cases


def run_s4_mixed(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    hi = bridge.from_spec(ROLES["SCRIPT_HI"], prefer_fourier_text=False)
    en = bridge.from_spec(ROLES["SCRIPT_EN"], prefer_fourier_text=False)
    icon = bridge.from_spec(ROLES["ICON"], prefer_fourier_text=False)
    ram_hi = bridge.from_spec(LEXICAL["राम"])
    ram_en = bridge.from_spec(LEXICAL["Ram"])
    person = bridge.from_spec(EMOJI["person"])
    mem = bundle(bind(hi, ram_hi), bind(en, ram_en), bind(icon, person))
    ret = unbind(mem, hi)
    cases.append(
        _case(
            "S4",
            "mixed_query_hi_retrieves_ram_devanagari",
            cosine(ret, ram_hi) > cosine(ret, ram_en) and cosine(ret, ram_hi) > cosine(ret, person),
            f"hi={cosine(ret, ram_hi):.3f} en={cosine(ret, ram_en):.3f} icon={cosine(ret, person):.3f}",
            metric=cosine(ret, ram_hi),
        )
    )
    # Neg: false friend — query EN expecting Devanagari
    ret_en = unbind(mem, en)
    cases.append(
        _case(
            "S4",
            "neg_en_query_prefers_latin_ram",
            cosine(ret_en, ram_en) > cosine(ret_en, ram_hi),
            f"en={cosine(ret_en, ram_en):.3f} hi={cosine(ret_en, ram_hi):.3f}",
            negative=True,
        )
    )
    return cases


def run_s5_nested(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    L = bridge.from_spec(ROLES["LEFT"], prefer_fourier_text=False)
    R = bridge.from_spec(ROLES["RIGHT"], prefer_fourier_text=False)
    A = bridge.from_spec(MEDIA["tree_A"])
    B = bridge.from_spec(MEDIA["tree_B"])
    C = bridge.from_spec(MEDIA["tree_C"])
    D = bridge.from_spec(MEDIA["tree_D"])
    node1 = bundle(bind(L, A), bind(R, B))
    node2 = bundle(bind(L, C), bind(R, D))
    root = bundle(bind(L, node1), bind(R, node2))
    # path to B: unbind LEFT -> node1; unbind RIGHT -> B
    n1 = unbind(root, L)
    b_ret = unbind(n1, R)
    thr = 0.5
    cases.append(
        _case(
            "S5",
            "nested_path_left_right_recovers_B",
            cosine(b_ret, B) > thr and cosine(b_ret, B) > cosine(b_ret, A),
            f"B={cosine(b_ret, B):.3f} A={cosine(b_ret, A):.3f}",
            metric=cosine(b_ret, B),
            threshold=thr,
        )
    )
    wrong = unbind(n1, L)  # should be nearer A than B if path wrong for B
    cases.append(
        _case(
            "S5",
            "neg_wrong_nested_path_not_B",
            cosine(wrong, B) < cosine(b_ret, B),
            f"wrong_B={cosine(wrong, B):.3f} correct_B={cosine(b_ret, B):.3f}",
            negative=True,
        )
    )
    return cases


def run_s6_media(bridge: FourierVSABridge) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    img = bridge.from_spec(ROLES["MODALITY_IMAGE"], prefer_fourier_text=False)
    aud = bridge.from_spec(ROLES["MODALITY_AUDIO"], prefer_fourier_text=False)
    pA = bridge.from_spec(MEDIA["patch_A"])
    pB = bridge.from_spec(MEDIA["patch_B"])
    cA = bridge.from_spec(MEDIA["clip_A"])
    cB = bridge.from_spec(MEDIA["clip_B"])
    mem = bundle(bind(img, pA), bind(aud, cA))
    r_img = unbind(mem, img)
    cases.append(
        _case(
            "S6",
            "synthetic_image_unbind",
            cosine(r_img, pA) > cosine(r_img, pB),
            f"pA={cosine(r_img, pA):.3f} pB={cosine(r_img, pB):.3f}",
            metric=cosine(r_img, pA),
            critical=False,
        )
    )
    r_aud = unbind(mem, aud)
    cases.append(
        _case(
            "S6",
            "synthetic_audio_unbind",
            cosine(r_aud, cA) > cosine(r_aud, cB),
            f"cA={cosine(r_aud, cA):.3f} cB={cosine(r_aud, cB):.3f}",
            metric=cosine(r_aud, cA),
            critical=False,
        )
    )
    noisy = dropout_noise(mem, cfg.VSA_NOISE_FRAC, seed=42)
    r_n = unbind(noisy, img)
    cases.append(
        _case(
            "S6",
            "neg_30pct_dropout_still_retrieves_patch",
            cosine(r_n, pA) > cosine(r_n, pB),
            f"pA={cosine(r_n, pA):.3f} pB={cosine(r_n, pB):.3f} noise={cfg.VSA_NOISE_FRAC}",
            metric=cosine(r_n, pA),
            negative=True,
            critical=False,
        )
    )
    cases.append(
        _case(
            "S6",
            "high_bitrate_tests_skipped_hermetic",
            True,
            "high_bitrate_tests=skipped_hermetic (no WAV/TIFF downloads)",
            critical=False,
        )
    )
    return cases


def run_s7_capacity(bridge: FourierVSABridge) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    curve: dict[str, list[float]] = {}

    def capacity_for_dim(dim: int, max_k: int = 40) -> tuple[int | None, list[float]]:
        cos_list: list[float] = []
        pairs = []
        for i in range(max_k):
            a = random_hv(dim, seed=7000 + i * 2)
            b = random_hv(dim, seed=7000 + i * 2 + 1)
            pairs.append((a, b))
        mem = torch.zeros(dim)
        fail_k = None
        for k, (a, b) in enumerate(pairs, start=1):
            mem = mem + bind(a, b)
            ret = unbind(mem, a)
            c = cosine(ret, b)
            cos_list.append(c)
            if fail_k is None and c < cfg.VSA_CAPACITY_TARGET_COS:
                fail_k = k
        return fail_k, cos_list

    k2048, curve_2048 = capacity_for_dim(2048 if bridge.hrr_dim >= 2048 else bridge.hrr_dim)
    k512, curve_512 = capacity_for_dim(512)
    curve["2048"] = curve_2048
    curve["512"] = curve_512

    cases.append(
        _case(
            "S7",
            "capacity_threshold_recorded",
            k2048 is not None or True,
            f"first_k_below_{cfg.VSA_CAPACITY_TARGET_COS} at D~2048: {k2048}",
            metric=float(k2048 or -1),
            critical=False,
        )
    )
    # directional: larger D should survive at least as long on average early ks
    early_2048 = sum(curve_2048[:5]) / 5
    early_512 = sum(curve_512[:5]) / 5
    cases.append(
        _case(
            "S7",
            "neg_capacity_grows_with_dim_early_window",
            early_2048 >= early_512 - 0.05,
            f"early5_mean D2048={early_2048:.3f} D512={early_512:.3f}",
            negative=True,
            critical=False,
        )
    )
    meta = {"capacity_k_2048": k2048, "capacity_k_512": k512, "curves": curve}
    return cases, meta


def run_all_stages() -> dict[str, Any]:
    bridge = FourierVSABridge()
    all_cases: list[dict[str, Any]] = []
    all_cases.extend(run_s0_hrr_axioms(bridge))
    all_cases.extend(run_s1_lexical_emoji(bridge))
    all_cases.extend(run_s2_abugida(bridge))
    all_cases.extend(run_s3_math(bridge))
    all_cases.extend(run_s4_mixed(bridge))
    all_cases.extend(run_s5_nested(bridge))
    all_cases.extend(run_s6_media(bridge))
    s7_cases, s7_meta = run_s7_capacity(bridge)
    all_cases.extend(s7_cases)

    critical = [c for c in all_cases if c.get("critical", True)]
    soft = [c for c in all_cases if not c.get("critical", True)]
    return {
        "problem_id": cfg.PROBLEM_ID,
        "hrr_dim": bridge.hrr_dim,
        "cases": all_cases,
        "n_pass": sum(1 for c in all_cases if c["pass"]),
        "n_total": len(all_cases),
        "n_critical_fail": sum(1 for c in critical if not c["pass"]),
        "n_soft_fail": sum(1 for c in soft if not c["pass"]),
        "ok_critical": all(c["pass"] for c in critical),
        "vq_comparison": "skipped" if not cfg.ENABLE_VQ_PROBLEM5 else "enabled",
        "high_bitrate_tests": "skipped_hermetic",
        "capacity_meta": s7_meta,
        "math_note": "S3 proves symbolic role-filler algebra on Fourier/HRR vectors, not LM arithmetic (Problem #1).",
    }
