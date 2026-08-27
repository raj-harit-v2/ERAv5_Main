/**
 * SwiGLU / Classic FFN interactive SVG diagrams — readable arrows, hover, sync highlight.
 * Arrow markers use userSpaceOnUse so heads stay proportional (not stroke-scaled).
 */
(function (global) {
  "use strict";

  var TIPS = {
    d_model_in: "Input hidden dimension D (residual stream width).",
    d_model_out: "Output hidden dimension D (back to residual stream).",
    W1: "Up-projection W₁: maps D → d_ff. Classic FFN uses one up matrix.",
    W1_swiglu: "Up-projection W₁ on the activated branch; output passes through Swish.",
    ReLU: "ReLU activation: max(0, x). Classic FFN nonlinearity.",
    Swish: "Swish / SiLU (β=1): x·σ(x). Gated branch activation in SwiGLU.",
    V: "Value branch V (Llama: W_up): maps D → d_ff without activation before ⊗.",
    otimes: "Element-wise multiply ⊗: gates Swish(W₁·x) with V·x (GLU mechanism).",
    W2: "Down-projection W₂ (Llama: W_down): maps d_ff → D.",
    d_ff: "Feed-forward hidden width (expanded dimension).",
  };

  /* Fixed toy walkthrough: D=2, d_ff=2 — same x for both paths (token emb for \"the\"). */
  var TOY_NOTE = "Toy D=2, d_ff=2 · token emb for “the”";
  var WALK_CLASSIC = [
    { node: "d_model_in", step: "Input x", shape: "[D]", values: "[0.60, −0.40]", note: "residual stream" },
    { node: "W1", step: "After W₁", shape: "[d_ff]", values: "[0.40, −0.70]", note: "xW₁ (one up path)" },
    { node: "ReLU", step: "After ReLU", shape: "[d_ff]", values: "[0.40, 0.00]", note: "negatives → 0" },
    { node: "W2", step: "After W₂", shape: "[D]", values: "[0.20, 0.08]", note: "down-project" },
    { node: "d_model_out", step: "Output", shape: "[D]", values: "[0.20, 0.08]", note: "back to residual" },
  ];
  var WALK_SWIGLU = [
    { node: "d_model_in", step: "Input x", shape: "[D]", values: "[0.60, −0.40]", note: "same x as classic" },
    { node: "W1", step: "After W₁", shape: "[d_ff]", values: "[0.40, −0.70]", note: "gate pre-act" },
    { node: "Swish", step: "After Swish", shape: "[d_ff]", values: "[0.24, −0.23]", note: "SiLU(gate)" },
    { node: "V", step: "After V", shape: "[d_ff]", values: "[0.56, −0.12]", note: "value / W_up" },
    { node: "otimes", step: "After ⊗", shape: "[d_ff]", values: "[0.13, 0.03]", note: "Swish ⊗ V" },
    { node: "W2", step: "After W₂", shape: "[D]", values: "[0.075, 0.016]", note: "down-project" },
    { node: "d_model_out", step: "Output", shape: "[D]", values: "[0.075, 0.016]", note: "back to residual" },
  ];

  var SYNC = {
    W1: ["W1", "Swish"],
    ReLU: ["Swish"],
    Swish: ["ReLU", "Swish"],
    V: ["V"],
    otimes: ["otimes", "ReLU"],
    W2: ["W2"],
    d_model_in: ["d_model_in"],
    d_model_out: ["d_model_out"],
  };

  var bus = { listeners: [] };

  function emitHover(nodeId, panelId) {
    bus.listeners.forEach(function (fn) {
      fn(nodeId, panelId);
    });
  }

  function onHover(fn) {
    bus.listeners.push(fn);
  }

  function round256(n) {
    return Math.round(n / 256) * 256;
  }

  function fmtM(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(Math.round(n));
  }

  function paramStats(dModel) {
    var dFfClassic = 4 * dModel;
    var dFfSwi = round256((8 / 3) * dModel);
    return {
      d_model: dModel,
      classic: { d_ff: dFfClassic, matrices: 2, params: 2 * dModel * dFfClassic },
      swiglu: { d_ff: dFfSwi, matrices: 3, params: 3 * dModel * dFfSwi },
    };
  }

  /* Fixed-size arrow head — do NOT use strokeWidth units (those balloon at stroke=2+) */
  function markerDef(id) {
    return (
      '<marker id="' + id + '" viewBox="0 0 10 10" refX="9" refY="5" ' +
      'markerWidth="7" markerHeight="7" orient="auto" markerUnits="userSpaceOnUse">' +
      '<path d="M1,1 L9,5 L1,9 Z" fill="#334155"/></marker>'
    );
  }

  function arrow(x1, y1, x2, y2, mid) {
    return (
      '<path class="s9-arrow" d="M' + x1 + "," + y1 + " L" + x2 + "," + y2 + '" ' +
      'fill="none" stroke="#475569" stroke-width="1.6" stroke-linecap="round" ' +
      'marker-end="url(#' + mid + ')"/>'
    );
  }

  function trap(cx, cy, w, h, widenUp, label, nodeId) {
    var hw = w / 2;
    var pts;
    if (widenUp) {
      pts = [
        cx - hw * 0.72, cy + h / 2,
        cx + hw * 0.72, cy + h / 2,
        cx + hw, cy - h / 2,
        cx - hw, cy - h / 2,
      ];
    } else {
      pts = [
        cx - hw, cy + h / 2,
        cx + hw, cy + h / 2,
        cx + hw * 0.72, cy - h / 2,
        cx - hw * 0.72, cy - h / 2,
      ];
    }
    return (
      '<g class="s9-node" data-node="' + nodeId + '" tabindex="0" role="img">' +
      '<polygon points="' + pts.join(" ") + '" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.25"/>' +
      '<text x="' + cx + '" y="' + (cy + 1) + '" text-anchor="middle" dominant-baseline="middle" ' +
      'font-size="13" font-weight="700" fill="#0f172a" pointer-events="none">' + label + "</text></g>"
    );
  }

  function actBox(cx, cy, w, h, label, nodeId) {
    return (
      '<g class="s9-node" data-node="' + nodeId + '" tabindex="0" role="img">' +
      '<rect x="' + (cx - w / 2) + '" y="' + (cy - h / 2) + '" width="' + w + '" height="' + h + '" ' +
      'rx="6" fill="#fef3c7" stroke="#d97706" stroke-width="1.25"/>' +
      '<text x="' + cx + '" y="' + (cy + 1) + '" text-anchor="middle" dominant-baseline="middle" ' +
      'font-size="12" font-weight="700" fill="#0f172a" pointer-events="none">' + label + "</text></g>"
    );
  }

  function dimText(x, y, text, anchor, italic) {
    return (
      '<text x="' + x + '" y="' + y + '" text-anchor="' + (anchor || "middle") + '" ' +
      'font-size="11" fill="#64748b" font-weight="600"' +
      (italic ? ' font-style="italic"' : "") +
      ">" + text + "</text>"
    );
  }

  /* Bottom → top layout, roomy viewBox 320×380 */
  function renderClassic(markerId) {
    var cx = 160;
    var s = "";
    /* title room at top; formula under title */
    s += dimText(cx, 36, "ReLU(xW₁)W₂", "middle", true);
    s += (
      '<g class="s9-node" data-node="d_model_out" tabindex="0">' +
      '<text x="' + cx + '" y="58" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">d_model</text></g>'
    );
    s += arrow(cx, 64, cx, 88, markerId);
    s += trap(cx, 118, 72, 44, false, "W₂", "W2");
    s += arrow(cx, 142, cx, 168, markerId);
    s += actBox(cx, 190, 64, 28, "ReLU", "ReLU");
    s += dimText(cx + 48, 194, "d_ff", "start");
    s += arrow(cx, 206, cx, 232, markerId);
    s += trap(cx, 262, 72, 44, true, "W₁", "W1");
    s += arrow(cx, 286, cx, 310, markerId);
    s += (
      '<g class="s9-node" data-node="d_model_in" tabindex="0">' +
      '<text x="' + cx + '" y="332" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">d_model</text></g>'
    );
    return s;
  }

  function renderSwiglu(markerId) {
    var cx = 160;
    var lx = 98;
    var rx = 222;
    var s = "";
    /* top → formula → d_model out → W2 → ⊗ ← Swish|V ← W1|V ← d_model in (bottom) */
    s += dimText(cx, 36, "(Swish(xW₁) ⊗ xV)W₂", "middle", true);
    s += (
      '<g class="s9-node" data-node="d_model_out" tabindex="0">' +
      '<text x="' + cx + '" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">d_model</text></g>'
    );
    s += arrow(cx, 62, cx, 84, markerId);
    s += trap(cx, 112, 74, 42, false, "W₂", "W2");
    s += arrow(cx, 135, cx, 156, markerId);
    s += (
      '<g class="s9-node" data-node="otimes" tabindex="0">' +
      '<circle cx="' + cx + '" cy="172" r="15" fill="#e0f2fe" stroke="#0284c7" stroke-width="1.4"/>' +
      '<text x="' + cx + '" y="176" text-anchor="middle" font-size="14" font-weight="700" pointer-events="none">⊗</text></g>'
    );
    /* into ⊗ from Swish (left) and V (right) */
    s += arrow(lx, 210, cx - 10, 184, markerId);
    s += arrow(rx, 250, cx + 10, 184, markerId);
    s += actBox(lx, 230, 60, 28, "Swish", "Swish");
    s += dimText(lx + 44, 234, "d_ff", "start");
    s += arrow(lx, 262, lx, 246, markerId);
    s += trap(lx, 290, 60, 40, true, "W₁", "W1");
    s += trap(rx, 290, 60, 40, true, "V", "V");
    s += arrow(cx, 340, lx, 312, markerId);
    s += arrow(cx, 340, rx, 312, markerId);
    s += (
      '<g class="s9-node" data-node="d_model_in" tabindex="0">' +
      '<text x="' + cx + '" y="362" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">d_model</text></g>'
    );
    return s;
  }

  function render(container, mode, opts) {
    opts = opts || {};
    if (typeof container === "string") {
      container = document.getElementById(container);
    }
    if (!container) return null;

    var panelId = opts.panelId || mode;
    var markerId = "arrowHead_" + panelId.replace(/\W/g, "_");
    var title = mode === "classic" ? "Original Feed Forward Layer" : "Feed Forward with SwiGLU";
    var body = mode === "classic" ? renderClassic(markerId) : renderSwiglu(markerId);
    var vbH = mode === "classic" ? 350 : 380;

    container.innerHTML =
      '<svg class="s9-swiglu-svg' + (opts.static ? " s9-swiglu-svg-static" : "") + '" viewBox="0 0 320 ' + vbH + '" role="img" aria-label="' + title + '">' +
      "<defs>" + markerDef(markerId) + "</defs>" +
      '<text x="160" y="18" text-anchor="middle" font-size="13" font-weight="800" fill="#0f172a">' +
      title +
      "</text>" +
      body +
      "</svg>";

    if (opts.static) {
      container.classList.add("s9-static");
    } else {
      container.classList.remove("s9-static");
      wireInteractions(container.querySelector("svg"), panelId, mode);
    }
    return container.querySelector("svg");
  }

  function wireInteractions(svg, panelId, mode) {
    if (!svg) return;
    svg.querySelectorAll(".s9-node").forEach(function (node) {
      var id = node.getAttribute("data-node");
      node.addEventListener("mouseenter", function () {
        node.classList.add("hl-local");
        showTip(node, id, mode);
        emitHover(id, panelId);
      });
      node.addEventListener("mouseleave", function () {
        node.classList.remove("hl-local");
        hideTip();
        emitHover(null, panelId);
      });
      node.addEventListener("focus", function () {
        showTip(node, id, mode);
      });
      node.addEventListener("blur", hideTip);
    });
  }

  var tipEl = null;

  function ensureTip() {
    if (tipEl) return tipEl;
    tipEl = document.createElement("div");
    tipEl.className = "s9-tip";
    tipEl.setAttribute("aria-live", "polite");
    tipEl.hidden = true;
    document.body.appendChild(tipEl);
    return tipEl;
  }

  function walkRowFor(mode, nodeId) {
    var rows = mode === "swiglu" ? WALK_SWIGLU : WALK_CLASSIC;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].node === nodeId) return rows[i];
    }
    return null;
  }

  function showTip(node, nodeId, mode) {
    var key = nodeId === "W1" && mode === "swiglu" ? "W1_swiglu" : nodeId;
    var tip = ensureTip();
    var base = TIPS[key] || TIPS[nodeId] || nodeId;
    var row = walkRowFor(mode, nodeId);
    var html = "<div class=\"s9-tip-base\">" + base + "</div>";
    if (row) {
      html +=
        "<div class=\"s9-tip-sample\">" +
        "<b>" + row.step + "</b> · " + row.shape + "<br>" +
        "<code>" + row.values + "</code>" +
        (row.note ? "<br><span class=\"s9-tip-note\">" + row.note + "</span>" : "") +
        "</div>";
    }
    tip.innerHTML = html;
    tip.hidden = false;
    var r = node.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - 300, Math.max(8, r.left)) + "px";
    tip.style.top = r.bottom + 8 + window.scrollY + "px";
  }

  function hideTip() {
    if (tipEl) tipEl.hidden = true;
  }

  function applySyncHighlight(nodeId) {
    document.querySelectorAll(".s9-node").forEach(function (n) {
      n.classList.remove("hl-sync");
    });
    document.querySelectorAll(".s9-walk-table tr[data-node]").forEach(function (tr) {
      tr.classList.remove("hl-row");
    });
    if (!nodeId) return;
    var targets = SYNC[nodeId] || [nodeId];
    document.querySelectorAll(".s9-node").forEach(function (n) {
      var id = n.getAttribute("data-node");
      if (targets.indexOf(id) >= 0 || id === nodeId) n.classList.add("hl-sync");
    });
    document.querySelectorAll(".s9-walk-table tr[data-node]").forEach(function (tr) {
      var id = tr.getAttribute("data-node");
      if (targets.indexOf(id) >= 0 || id === nodeId) tr.classList.add("hl-row");
    });
  }

  onHover(function (nodeId) {
    applySyncHighlight(nodeId);
  });

  function renderWalkTable(container, mode) {
    if (typeof container === "string") {
      container = document.getElementById(container);
    }
    if (!container) return;
    var rows = mode === "swiglu" ? WALK_SWIGLU : WALK_CLASSIC;
    var html =
      '<p class="s9-walk-note">' + TOY_NOTE + "</p>" +
      '<table class="s9-walk-table" data-mode="' + mode + '">' +
      "<thead><tr><th>Step</th><th>Shape</th><th>Values</th><th>What happens</th></tr></thead><tbody>";
    rows.forEach(function (r) {
      html +=
        '<tr data-node="' + r.node + '">' +
        "<td>" + r.step + "</td>" +
        "<td><code>" + r.shape + "</code></td>" +
        "<td><code>" + r.values + "</code></td>" +
        "<td>" + r.note + "</td></tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
    container.querySelectorAll("tr[data-node]").forEach(function (tr) {
      var id = tr.getAttribute("data-node");
      tr.addEventListener("mouseenter", function () {
        emitHover(id, mode);
      });
      tr.addEventListener("mouseleave", function () {
        emitHover(null, mode);
      });
    });
  }

  function renderParamStrip(container, dModel) {
    if (typeof container === "string") {
      container = document.getElementById(container);
    }
    if (!container) return;
    var st = paramStats(dModel);
    container.innerHTML =
      '<table class="s9-param-table">' +
      "<thead><tr><th></th><th>d_ff</th><th>Matrices</th><th>≈ Params</th></tr></thead><tbody>" +
      "<tr><td><b>Classic ReLU</b></td><td>" + st.classic.d_ff.toLocaleString() +
      "</td><td>" + st.classic.matrices + "</td><td>" + fmtM(st.classic.params) + "</td></tr>" +
      "<tr><td><b>SwiGLU</b></td><td>" + st.swiglu.d_ff.toLocaleString() +
      "</td><td>" + st.swiglu.matrices + "</td><td>" + fmtM(st.swiglu.params) + "</td></tr>" +
      "</tbody></table>" +
      '<p class="s9-param-note">At D=4096 curriculum anchor: d_ff=11008, ~135.3M SwiGLU vs ~134.2M classic. ' +
      "NanoLM harness uses GELU FFN at D=64 (not SwiGLU).</p>";
  }

  function initWidget(opts) {
    opts = opts || {};
    var dModel = opts.dModel || 4096;
    render("swiglu-classic", "classic", { panelId: "classic" });
    render("swiglu-gated", "swiglu", { panelId: "swiglu" });
    renderWalkTable("swiglu-walk-classic", "classic");
    renderWalkTable("swiglu-walk-swiglu", "swiglu");
    renderParamStrip("swiglu-params", dModel);

    var slider = document.getElementById("dmodel-slider");
    var label = document.getElementById("dmodel-val");
    if (slider) {
      slider.value = String(dModel);
      if (label) label.textContent = String(dModel);
      slider.oninput = function () {
        dModel = +slider.value;
        if (label) label.textContent = String(dModel);
        renderParamStrip("swiglu-params", dModel);
      };
    }
  }

  function initExportPreview() {
    var details = document.getElementById("swiglu-export-details");
    if (!details) {
      render("swiglu-export-classic", "classic", { panelId: "export_classic", static: true });
      render("swiglu-export-gated", "swiglu", { panelId: "export_swiglu", static: true });
      return;
    }
    var rendered = false;
    function renderExport() {
      if (rendered) return;
      rendered = true;
      render("swiglu-export-classic", "classic", { panelId: "export_classic", static: true });
      render("swiglu-export-gated", "swiglu", { panelId: "export_swiglu", static: true });
    }
    if (details.open) renderExport();
    details.addEventListener("toggle", function () {
      if (details.open) renderExport();
    });
  }

  global.S9SwigluDiagram = {
    render: render,
    initWidget: initWidget,
    initExportPreview: initExportPreview,
    paramStats: paramStats,
    onHover: onHover,
    renderParamStrip: renderParamStrip,
    renderWalkTable: renderWalkTable,
  };
})(typeof window !== "undefined" ? window : globalThis);
