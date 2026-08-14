#!/usr/bin/env python3
"""Export animated GIF distinguishers for demo_07/coach_demo (Pillow + matplotlib).

Run once (or after visual tweaks):
  uv run python demo_07/export_coach_anim_gifs.py

Writes:
  demo_07/figures/anim_fourier_wave_sum_bharat.gif
  demo_07/figures/anim_fourier_compare_apple_affle.gif
  demo_07/figures/anim_kronecker_vs_fourier_indic.gif
  demo_07/figures/anim_dense_vs_fourier_params.gif
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings.fourier_baseline import FourierCodepointEmbedding  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CODE_DIM = 512
N_FREQ = 32


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 90) -> None:
    if not frames:
        raise RuntimeError(f"no frames for {path}")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size // 1024} KB)")


def _omega(cp: int) -> float:
    return 0.5 + ((cp * 2654435761) % 8000) / 1000.0


def _encode(text: str) -> np.ndarray:
    emb = FourierCodepointEmbedding(d_model=8, code_dim=CODE_DIM, n_freq=N_FREQ)
    with torch.no_grad():
        return emb.encode_string(text).numpy()


def gif_wave_sum_bharat() -> Path:
    text = "भारत"
    chars = list(text)
    xs = np.linspace(0, 2 * math.pi, 240)
    frames: list[Image.Image] = []
    n = 24
    code = _encode(text)
    code_vis = code[:120]
    for fi in range(n):
        u = fi / (n - 1)
        fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor="#0b1220")
        ax.set_facecolor("#0b1220")
        for i, ch in enumerate(chars):
            om = _omega(ord(ch))
            amp = 0.35 + 0.55 * min(1.0, u * (n / (i + 1)) / 3)
            y = np.sin(xs * om * (i + 1) * 0.35 + u * 4) * amp + (i * 0.55)
            ax.plot(xs, y, color=plt.cm.cool((i + 1) / (len(chars) + 1)), lw=1.4, alpha=0.85)
            ax.text(xs[0] - 0.35, y[0], f"c{i}", color="#94a3b8", fontsize=9)
        # summed / code preview grows in
        mid = -0.2
        preview = code_vis[: len(xs)]
        if len(preview) < len(xs):
            preview = np.interp(np.linspace(0, 1, len(xs)), np.linspace(0, 1, len(preview)), preview)
        ax.plot(xs, mid + preview * 0.45 * u, color="#34d399", lw=2.4)
        ax.set_xlim(-0.5, xs[-1] + 0.2)
        ax.set_ylim(-1.2, len(chars) * 0.55 + 0.8)
        ax.axis("off")
        ax.set_title(
            f"Fourier sum (Bharat) -> token code  (code_dim={CODE_DIM})   u={u:.2f}",
            color="#e2e8f0",
            fontsize=11,
            pad=8,
        )
        frames.append(_fig_to_pil(fig))
    # hold last frames
    frames.extend([frames[-1]] * 6)
    path = OUT / "anim_fourier_wave_sum_bharat.gif"
    _save_gif(frames, path, duration_ms=80)
    return path


def gif_compare_apple_affle() -> Path:
    a, b = "Apple", "Affle"
    ca, cb = _encode(a), _encode(b)
    cos = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8))
    xs = np.linspace(0, 2 * math.pi, 200)
    frames: list[Image.Image] = []
    n = 28
    for fi in range(n):
        u = fi / (n - 1)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), facecolor="#0b1220")
        for ax, text, code, color in (
            (axes[0], a, ca, "#34d399"),
            (axes[1], b, cb, "#38bdf8"),
        ):
            ax.set_facecolor("#0b1220")
            chars = list(text)
            for i, ch in enumerate(chars):
                om = _omega(ord(ch))
                y = np.sin(xs * om * (i + 1) * 0.3 + u * 5) * (0.4 * u) + i * 0.4
                ax.plot(xs, y, color=plt.cm.plasma((i + 1) / (len(chars) + 1)), lw=1.2, alpha=0.8)
            vis = code[: len(xs)]
            if len(vis) < len(xs):
                vis = np.interp(np.linspace(0, 1, len(xs)), np.linspace(0, 1, len(code[:120])), code[:120])
            ax.plot(xs, -0.5 + vis * 0.5 * u, color=color, lw=2.2)
            ax.set_title(text, color="#e2e8f0", fontsize=12)
            ax.axis("off")
        fig.suptitle(
            f"Fourier compare  cosine={cos * u:.3f}   (final {cos:.3f})",
            color="#fbbf24",
            fontsize=12,
            y=1.02,
        )
        frames.append(_fig_to_pil(fig))
    frames.extend([frames[-1]] * 8)
    path = OUT / "anim_fourier_compare_apple_affle.gif"
    _save_gif(frames, path, duration_ms=75)
    return path


def gif_kronecker_vs_fourier_indic() -> Path:
    text = "अंतर्राष्ट्रीय"
    utf8 = text.encode("utf-8")
    pos_dim = 32
    frames: list[Image.Image] = []
    n = 30
    chars = list(text)
    xs = np.linspace(0, 2 * math.pi, 180)
    for fi in range(n):
        u = fi / (n - 1)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), facecolor="#0b1220")
        # Left: Kronecker byte slots filling
        ax = axes[0]
        ax.set_facecolor("#0b1220")
        filled = int(min(len(utf8), pos_dim) * u)
        grid = np.zeros((1, pos_dim))
        for i in range(filled):
            grid[0, i] = (utf8[i] / 255.0) if i < len(utf8) else 0
        ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        trunc = len(utf8) > pos_dim
        ax.set_title(
            f"Kronecker byte×pos ({filled}/{pos_dim} bytes)"
            + ("  TRUNCATE" if trunc and filled >= pos_dim else ""),
            color="#f87171" if trunc and u > 0.85 else "#e2e8f0",
            fontsize=10,
        )
        ax.set_yticks([])
        ax.set_xlabel("byte slots", color="#94a3b8")
        ax.tick_params(colors="#64748b")
        # Right: Fourier waves
        ax = axes[1]
        ax.set_facecolor("#0b1220")
        show_n = max(1, int(len(chars) * u + 0.999))
        for i, ch in enumerate(chars[:show_n]):
            om = _omega(ord(ch))
            y = np.sin(xs * om * (i + 1) * 0.25 + u * 3) * 0.45 + i * 0.35
            ax.plot(xs, y, lw=1.3, alpha=0.85)
        ax.set_title(f"Fourier codepoints ({show_n}/{len(chars)} chars)", color="#34d399", fontsize=10)
        ax.axis("off")
        fig.suptitle(
            "Indic: UTF-8 burns the 32-byte window · Fourier sums Unicode codepoints",
            color="#e2e8f0",
            fontsize=11,
            y=1.02,
        )
        frames.append(_fig_to_pil(fig))
    frames.extend([frames[-1]] * 8)
    path = OUT / "anim_kronecker_vs_fourier_indic.gif"
    _save_gif(frames, path, duration_ms=70)
    return path


def gif_dense_vs_fourier_params() -> Path:
    """Dense V×D gather vs Fourier f(x) + note that D×V head is separate (#5).

    Pacing: further -60% speed vs prior cut (duration x2.5, more dwell frames).
    Examples (exact spelling): apple · Bharat · apfle  (NOT Affle).
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for fname in ("Segoe UI", "Nirmala UI", "DejaVu Sans"):
        if fname in available:
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False

    V, D = 131_072, 8_096
    dense_n = V * D
    kron_n = (256 * 32) * D
    fourier_n = 8_192 * D
    head_n = D * V
    dense_gb = dense_n * 16 / 1e9
    # Exact spellings requested (apfle = near-typo of apple; not Affle)
    examples = ("apple", "Bharat", "apfle")
    frames: list[Image.Image] = []
    # Step 1 (table) is visually static — cap dwell at ≤8s (19 * 418ms ≈ 7.9s)
    n_table, n_gather, n_wave, n_head = 19, 28, 45, 32
    total = n_table + n_gather + n_wave + n_head
    duration_ms = 418
    hold_n = 28
    fig_w, fig_h = 11.5, 5.6

    # Cache per-example char lists + base wave curves (reuse every frame)
    xs = np.linspace(0, 2 * math.pi, 160)
    x_plot = 1.8 + xs / (2 * math.pi) * 5.5
    wave_cache: dict[str, list[tuple[str, np.ndarray, float]]] = {}
    for ex in examples:
        chars = list("भारत" if ex == "Bharat" else ex)
        nch = max(len(chars), 1)
        gap = min(0.72, 3.8 / nch)
        y0 = 6.85
        cached: list[tuple[str, np.ndarray, float]] = []
        for i, ch in enumerate(chars):
            om = _omega(ord(ch))
            base = np.sin(xs * om * 0.35) * 0.22
            cached.append((ch, base, y0 - i * gap))
        wave_cache[ex] = cached
        safe = ",".join(c[0] if c[0].isascii() else "?" for c in cached)
        print(f"  wave cache '{ex}': {len(chars)} chars -> [{safe}]")

    step_meta = {
        "table": {
            "title": "Step 1/4 - Dense input table (V5 reference)",
            "caption": (
                f"STORE: W_in = {V:,} x {D:,} = {dense_n:,} params (~{dense_gb:.1f} GB AdamW).\n"
                "Example rows (IDs only):  apple  |  Bharat  |  apfle\n"
                "Tiny grid is a visual stand-in; Assign_007 never allocates the full V x D tensor."
            ),
        },
        "gather": {
            "title": "Step 2/4 - Encode = gather one row",
            "caption": (
                "ENCODE (dense): token_id -> gather ONE row -> emb in R^D. Spelling is invisible.\n"
                "Cycle: apple -> Bharat -> apfle (each a different unrelated row).\n"
                "Typo apfle does NOT share structure with apple in a dense table."
            ),
        },
        "wave": {
            "title": "Step 3/4 - Fourier f(text): apple / Bharat / apfle",
            "caption": (
                "ENCODE (Fourier #4): per-char waves -> SUM -> z-norm -> Linear -> emb.\n"
                "apple and apfle share most character waves; Bharat uses Unicode codepoints.\n"
                f"STORE only Linear({8192}->{D}) ~ {fourier_n:,} params - no row per word."
            ),
        },
        "head": {
            "title": "Step 4/4 - Head predicts token IDs (not spelling invert)",
            "caption": (
                f"PREDICT: W_out = {D:,} x {V:,} = {head_n:,} logits (apple / Bharat / apfle as IDs).\n"
                "This is NOT recovering characters from the embedding. Problem #5 / VQ is gated off.\n"
                f"Kron~{kron_n:,}  Fourier~{fourier_n:,}  << dense {dense_n:,}."
            ),
        },
    }

    def _phase(fi: int) -> tuple[str, float]:
        if fi < n_table:
            return "table", fi / max(n_table - 1, 1)
        if fi < n_table + n_gather:
            return "gather", (fi - n_table) / max(n_gather - 1, 1)
        if fi < n_table + n_gather + n_wave:
            return "wave", (fi - n_table - n_gather) / max(n_wave - 1, 1)
        return "head", (fi - n_table - n_gather - n_wave) / max(n_head - 1, 1)

    def _active_example(u: float) -> str:
        idx = min(int(u * len(examples)), len(examples) - 1)
        return examples[idx]

    def _panel_title(ax, text: str, color: str) -> None:
        ax.text(
            0.5,
            0.98,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=True,
        )

    def _body(ax, x: float, y: float, text: str, color: str, size: float = 10.0) -> None:
        ax.text(x, y, text, color=color, fontsize=size, ha="center", va="center", linespacing=1.25, clip_on=True)

    def _example_chips(ax, active: str, y: float = 8.4) -> None:
        """Monospace chips so apfle (p-f) is not misread as Affle."""
        xs_chip = (2.2, 5.0, 7.8)
        for x, lab in zip(xs_chip, examples):
            on = lab == active
            ax.add_patch(
                plt.Rectangle(
                    (x - 1.2, y - 0.42),
                    2.4,
                    0.84,
                    fill=True,
                    facecolor="#134e4a" if on else "#1e293b",
                    edgecolor="#34d399" if on else "#475569",
                    lw=1.8 if on else 1.0,
                    clip_on=True,
                )
            )
            ax.text(
                x,
                y,
                lab,
                color="#ecfdf5" if on else "#e2e8f0",
                fontsize=11 if on else 10,
                fontweight="bold",
                fontfamily="monospace",
                ha="center",
                va="center",
                clip_on=True,
            )

    def _boxed(ax, x, y, w, h, edge, face="#1e293b") -> None:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor=face, edgecolor=edge, lw=1.6, clip_on=True)
        )

    def _frames_left_in_phase(fi: int) -> int:
        if fi < n_table:
            return n_table - fi
        if fi < n_table + n_gather:
            return n_table + n_gather - fi
        if fi < n_table + n_gather + n_wave:
            return n_table + n_gather + n_wave - fi
        return total - fi

    def _countdown_s(fi: int) -> int:
        # Remaining seconds in this step (ceil), small HUD top-right
        left = _frames_left_in_phase(fi)
        return max(0, int(math.ceil(left * duration_ms / 1000.0)))

    for fi in range(total):
        phase, u = _phase(fi)
        meta = step_meta[phase]
        active = _active_example(u)
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0b1220")

        title = meta["title"]
        if phase == "gather":
            title = f'Step 2/4 - Encode = gather row  "{active}"'
        elif phase == "wave":
            title = f'Step 3/4 - Fourier compose  "{active}" ({len(wave_cache[active])} chars)'

        fig.text(0.5, 0.965, title, color="#f8fafc", fontsize=13, fontweight="bold", ha="center", va="top")
        # Small countdown (seconds) top-right — bright yellow
        fig.text(
            0.965,
            0.965,
            f"{_countdown_s(fi)}s",
            color="#facc15",
            fontsize=8,
            fontfamily="monospace",
            ha="right",
            va="top",
        )

        ax_l = fig.add_axes([0.05, 0.32, 0.43, 0.55])
        ax_r = fig.add_axes([0.52, 0.32, 0.43, 0.55])
        for ax in (ax_l, ax_r):
            ax.set_facecolor("#0b1220")
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")
            ax.add_patch(
                plt.Rectangle((0.15, 0.15), 9.7, 9.7, fill=False, edgecolor="#334155", lw=1.2, clip_on=False)
            )

        if phase in ("table", "gather"):
            _panel_title(ax_l, "LEFT - Dense path", "#fca5a5")
            rows, cols = 12, 8
            grid = np.full((rows, cols), 0.2, dtype=float)
            grid += 0.05 * np.linspace(0, 1, cols)
            row_map = {"apple": 2, "Bharat": 5, "apfle": 8}
            if phase == "gather":
                grid[row_map[active], :] = 1.0
            ax_l.imshow(grid, extent=[2.4, 8.8, 1.8, 7.0], aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax_l.add_patch(plt.Rectangle((2.4, 1.8), 6.4, 5.2, fill=False, edgecolor="#f87171", lw=1.8))
            for lab, ridx in row_map.items():
                y = 7.0 - (ridx + 0.5) / rows * (7.0 - 1.8)
                on = lab == active and phase == "gather"
                ax_l.text(
                    2.55,
                    y,
                    lab,
                    color="#fde68a" if on else "#fecaca",
                    fontsize=9,
                    fontfamily="monospace",
                    fontweight="bold" if on else "normal",
                    ha="left",
                    va="center",
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0b1220", edgecolor="none", alpha=0.75),
                )
            _body(ax_l, 5.5, 8.55, f"W_in  {V:,} x {D:,}\n{dense_n:,} params", "#fecaca", 10.5)
            if phase == "gather":
                _body(ax_l, 5.5, 0.85, f'gather "{active}" -> emb in R^D', "#fde68a", 10)
            else:
                _body(ax_l, 5.5, 0.85, "rows: apple | Bharat | apfle", "#fca5a5", 10)
        else:
            _panel_title(ax_l, "LEFT - No V x D table for input", "#e2e8f0")
            _boxed(ax_l, 0.8, 2.2, 8.4, 5.4, "#94a3b8")
            _body(
                ax_l,
                5.0,
                5.8,
                "Fourier builds codes from text\napple  |  Bharat  |  apfle\n(not from a vocab table)",
                "#f8fafc",
                11,
            )
            if active == "Bharat":
                _body(ax_l, 5.0, 3.4, "Active: Bharat (Unicode)", "#a7f3d0", 10.5)
            else:
                _body(ax_l, 5.0, 3.4, f'Active: "{active}"', "#a7f3d0", 10.5)
            _body(ax_l, 5.0, 0.9, f"Dense reference still {dense_n:,}", "#e2e8f0", 9.5)

        if phase in ("table", "gather"):
            _panel_title(ax_r, "RIGHT - Same words later as Fourier", "#a7f3d0")
            _example_chips(ax_r, active if phase == "gather" else examples[0], y=8.35)
            _boxed(ax_r, 0.8, 3.6, 8.4, 3.4, "#134e4a", face="#0f172a")
            _body(
                ax_r,
                5.0,
                5.5,
                "Dense cannot see that\napple ~ apfle by spelling.\nBharat is just another row.",
                "#a7f3d0",
                10.5,
            )
            _body(
                ax_r,
                5.0,
                2.3,
                f"Next: char waves + Linear {8192}->{D}\n~ {fourier_n:,} params",
                "#6ee7b7",
                10,
            )
            _body(ax_r, 5.0, 0.85, "Params independent of vocab size V", "#e2e8f0", 9.5)
        elif phase == "wave":
            _panel_title(ax_r, "RIGHT - Fourier encode path (all chars)", "#a7f3d0")
            _example_chips(ax_r, active, y=8.55)
            cached = wave_cache[active]
            # Wave region (leave clear strip at bottom for Linear — no overlap with sum)
            _boxed(ax_r, 0.55, 2.35, 8.9, 5.55, "#0f766e", face="#0b1220")
            phase_shift = u * 4.0
            for i, (ch, base, y_mid) in enumerate(cached):
                y = y_mid + base * np.cos(phase_shift * 0.25) + 0.08 * np.sin(xs * 0.5 + phase_shift + i)
                ax_r.plot(
                    x_plot,
                    y,
                    color=plt.cm.cool((i + 1) / (len(cached) + 1)),
                    lw=1.6,
                    alpha=0.95,
                )
                ax_r.text(
                    1.15,
                    float(y_mid),
                    ch,
                    color="#e2e8f0",
                    fontsize=10,
                    fontfamily="monospace" if ch.isascii() else "Nirmala UI",
                    ha="center",
                    va="center",
                    clip_on=True,
                )
            # Sum curve near bottom of wave box; label on the RIGHT (not over Linear)
            sum_y = 2.85 + 0.22 * np.sin(xs * 1.15 + phase_shift)
            ax_r.plot(x_plot, sum_y, color="#34d399", lw=2.5)
            ax_r.text(
                8.55,
                3.15,
                "sum",
                color="#34d399",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0b1220", edgecolor="#34d399", lw=1.0),
            )
            # Linear box BELOW wave box (separate strip — fixes apfle overlap)
            ax_r.add_patch(
                plt.Rectangle(
                    (0.55, 0.45),
                    8.9,
                    1.55,
                    fill=True,
                    facecolor="#0f766e",
                    edgecolor="#34d399",
                    lw=1.5,
                    clip_on=True,
                )
            )
            _body(ax_r, 5.0, 1.2, f'Linear -> emb("{active}")   ~ {fourier_n:,} params', "#ecfdf5", 9.5)
        else:
            _panel_title(ax_r, "RIGHT - Output head (separate)", "#fbbf24")
            _boxed(ax_r, 0.8, 3.0, 8.4, 5.0, "#fbbf24")
            _body(ax_r, 5.0, 7.2, f"W_out  {D:,} x {V:,}", "#f8fafc", 12)
            _body(ax_r, 5.0, 6.2, "logits over vocab IDs", "#fbbf24", 10.5)
            _example_chips(ax_r, active, y=4.7)
            _body(ax_r, 5.0, 3.4, "!= spelling invert from emb (#5 off)", "#f87171", 10)
            _body(
                ax_r,
                5.0,
                0.9,
                f"Kron~{kron_n:,}  Fourier~{fourier_n:,}\n<< dense {dense_n:,}",
                "#e2e8f0",
                9.5,
            )

        fig.patches.append(
            plt.Rectangle(
                (0.04, 0.035),
                0.92,
                0.24,
                transform=fig.transFigure,
                facecolor="#111827",
                edgecolor="#475569",
                lw=1.2,
                zorder=0,
            )
        )
        fig.text(0.06, 0.22, meta["caption"], color="#f8fafc", fontsize=9.5, ha="left", va="top", linespacing=1.35)

        # In-GIF step markers (+15% size; brighter active/inactive)
        names = ("table", "gather", "wave", "head")
        for i, name in enumerate(names):
            fig.text(
                0.72 + i * 0.055,
                0.955,
                "*" if name == phase else "o",
                color="#4ade80" if name == phase else "#94a3b8",
                fontsize=15.2,
                fontweight="bold" if name == phase else "normal",
                ha="center",
                va="center",
            )

        frames.append(_fig_to_pil_hires(fig))

    # End hold with 0s countdown frames
    for _ in range(hold_n):
        frames.append(frames[-1])
    path = OUT / "anim_dense_vs_fourier_params.gif"
    _save_gif(frames, path, duration_ms=duration_ms)
    print(f"  speed check: {len(frames)} frames x {duration_ms}ms ~= {len(frames) * duration_ms / 1000:.1f}s total")
    return path


def _fig_to_pil_hires(fig: plt.Figure) -> Image.Image:
    """Higher DPI for readable param-story GIF text."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.12)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=180)


def main() -> None:
    gif_wave_sum_bharat()
    gif_compare_apple_affle()
    gif_kronecker_vs_fourier_indic()
    gif_dense_vs_fourier_params()
    print("done ->", OUT)


if __name__ == "__main__":
    main()

