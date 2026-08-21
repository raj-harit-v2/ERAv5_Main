"""Self-tests for demo_Asgn_08 scaffold + teaching math mirrors."""

from __future__ import annotations

import json

import config
from Self_Diagionistics_02 import chunk_gate, tokenize, two_bills, causal_future_mass_ok

DEMO = config.ROOT / "demo_Asgn_08"


def test_demo_files_exist():
    assert (DEMO / "index.html").is_file()
    assert (DEMO / "scripts" / "pipeline_demo.js").is_file()
    assert (DEMO / "scripts" / "app.js").is_file()
    assert (DEMO / "data" / "steps.json").is_file()
    assert (DEMO / "styles.css").is_file()


def test_steps_json_has_widgets():
    data = json.loads((DEMO / "data" / "steps.json").read_text(encoding="utf-8"))
    assert data["schema"].startswith("a08.demo_asgn_08")
    assert "attention" in data["widgets"]
    assert "chunk_gate" in data["widgets"]
    assert len(data["steps"]) >= 5


def test_index_has_shared_inputs():
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    for wid in (
        "sentence",
        "chunk1",
        "fact",
        "question",
        "btn-update",
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
        "stage-nav",
    ):
        assert f'id="{wid}"' in html
    assert 'id="sample-hint"' in html
    assert "tip-btn" in html
    assert "Grade-10" not in html
    assert "What this is / is not" not in html
    assert 'data-stage="pipeline"' in html


def test_ui_is_staged():
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    for stage in ("pipeline", "memory", "compute", "position", "system"):
        assert f'data-stage="{stage}"' in html
    js = (DEMO / "scripts" / "app.js").read_text(encoding="utf-8")
    assert "showStage" in js
    assert "STAGE_ORDER" in js
    assert "widgetTipHtml" in js
    assert "bindTipButtons" in js
    assert "getWidgetTip" in js
    assert "refreshPanelCaptions" in js
    assert "applyTallScroll" in js


def test_panel_captions_status_and_scroll():
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    assert "panel-caption" in html
    assert 'data-caption="tokens"' in html
    assert "stage-grid-scroll" in html
    assert "hint-lead" in html
    assert "hint-body" in html
    css = (DEMO / "styles.css").read_text(encoding="utf-8")
    assert "overflow-x: hidden" in css
    assert "scroll-panel-tall" in css
    assert ".scroll-panel {" not in css
    assert ".hint-lead" in css
    js = (DEMO / "scripts" / "app.js").read_text(encoding="utf-8")
    assert 'Last run · T=' in js
    assert "els.status.title" in js
    assert "summary" in js and "formula" in js
    assert "refreshSampleHint" in js
    assert "state.gateStep = 0" in js


def test_tokenize_sentence():
    assert tokenize("The cat sat on the mat") == ["The", "cat", "sat", "on", "the", "mat"]
    assert tokenize("  ") == []


def test_two_bills_quadratic():
    b = two_bills(6)
    assert b["computeUnits"] == 36
    assert b["kvUnits"] == 6


def test_chunk_gate_user_fact():
    g = chunk_gate("9910", True, 0.15)
    assert g["verdict"] == "good"
    assert "9910" in g["message"]
    assert chunk_gate("9910", False, 0.15)["verdict"] == "bad"
    assert chunk_gate("9910", True, 1.0)["verdict"] == "loud"


def test_causal_future_mass_near_zero():
    ok, mass = causal_future_mass_ok()
    assert ok
    assert mass < 1e-8


def test_kv_gqa_linear_mirrors():
    from Self_Diagionistics_02 import kv_cache_bytes, gqa_layout, linear_softmax_off_match

    assert kv_cache_bytes(48, 8, 128, 32768, 1, 2) == 2 * 48 * 8 * 128 * 32768 * 1 * 2
    g = gqa_layout("GQA", 8, 2)
    assert g["H_KV"] == 2 and g["reduction"] == 4.0
    assert gqa_layout("MHA", 8, 99)["H_KV"] == 8
    assert gqa_layout("MQA", 8, 99)["H_KV"] == 1
    assert linear_softmax_off_match() is True


def test_delta_sparse_rope_mirrors():
    from Self_Diagionistics_02 import (
        delta_rule_unit_key_ok,
        sparse_topk_row_mass_ok,
        rope_relative_depends_on_delta,
    )

    assert delta_rule_unit_key_ok() is True
    assert sparse_topk_row_mass_ok() is True
    assert rope_relative_depends_on_delta() is True


def test_drope_compression_schedule_mirrors():
    from Self_Diagionistics_02 import drope_factor_ok, compression_tm_ok, schedule_motif_ok

    assert drope_factor_ok() is True
    assert compression_tm_ok() is True
    assert schedule_motif_ok() is True


def test_pipeline_js_exports_chunk_gate():
    js = (DEMO / "scripts" / "pipeline_demo.js").read_text(encoding="utf-8")
    assert "chunkGate" in js
    assert "runPipeline" in js
    assert "causalAttention" in js
    assert "kvCacheBytes" in js
    assert "gqaLayout" in js
    assert "linearSoftmaxOff" in js
    assert "deltaRuleWrite" in js
    assert "sparseTopKAttention" in js
    assert "ropeDemo" in js
    assert "dropeExtension" in js
    assert "sequenceCompression" in js
    assert "scheduleAndFork" in js


def test_steps_json_c6_c14():
    data = json.loads((DEMO / "data" / "steps.json").read_text(encoding="utf-8"))
    for w in (
        "kv_cache",
        "gqa",
        "linear_softmax_off",
        "delta",
        "sparse_topk",
        "rope",
        "drope",
        "compression",
        "schedule_fork",
    ):
        assert w in data["widgets"]
    assert len(data["steps"]) >= 14


def test_index_has_c12_c14():
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    for wid in ("w-drope", "w-comp", "w-sched"):
        assert f'id="{wid}"' in html
