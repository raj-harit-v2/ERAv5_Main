"""SwiGLU diagram asset and plot_stats smoke."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_swiglu_diagram_js_in_widgets():
    p = ROOT / "widgets" / "s9_swiglu_diagram.js"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "S9SwigluDiagram" in text
    assert "marker-end" in text
    assert "paramStats" in text


def test_plot_swiglu_generates_png(tmp_path):
    from src.pipeline.plot_stats import plot_swiglu

    out = tmp_path / "swiglu.png"
    plot_swiglu(out)
    assert out.stat().st_size > 5000
