(function () {
  "use strict";

  const state = {
    steps: null,
    D: 8,
    alpha: 0.391,
    sentence: "",
    chunk1: "",
    fact: "",
    question: "",
    gateStep: 0,
    streamOn: true,
    gate: 0.15,
    pipeline: null,
    // C6 KV wireframe (yardstick defaults from formulas §12)
    kvL: 48,
    kvH: 8,
    kvDhead: 128,
    kvB: 1,
    kvPb: 2,
    // C7 GQA wireframe
    gqaMode: "GQA",
    gqaHQ: 8,
    gqaHKV: 2,
    // C9–C11
    deltaNorm: true,
    sparseK: 3,
    ropeI: 0,
    ropeJ: 1,
    ropeTheta: 1,
    // C12–C14
    dropeTrain: 8192,
    dropeServe: 262144,
    compM: 4,
    compTopK: 2,
    // Use yardstick T for compression bill; sentence T for block chips
    compT: 32768,
    forkRoad: "stretch",
    uiStage: "pipeline",
  };

  const STAGE_ORDER = ["pipeline", "memory", "compute", "position", "system"];
  const STAGE_META = {
    pipeline: {
      title: "Pipeline",
      blurb: "Text becomes tokens, vectors, causal attention, and the two cost bills.",
      help: "<strong>Pipeline</strong>Standard path: tokenize → embed → scaled causal attention → compute ≈ T² and KV ≈ T.",
    },
    memory: {
      title: "Memory",
      blurb: "Cross-chunk summary, exact KV bytes, and shared KV heads (GQA/MQA).",
      help: "<strong>Memory</strong>When chunks end, a summary vector may cross the boundary. Cache size is 2·L·H_KV·d_head·T·B·P_b. GQA shrinks H_KV.",
    },
    compute: {
      title: "Compute",
      blurb: "Linear state, delta overwrite, and sparse top-k — attacks on the T² bill.",
      help: "<strong>Compute</strong>Softmax-off folds history into fixed S. Delta writes only the correction. Sparse keeps softmax on k keys.",
    },
    position: {
      title: "Position",
      blurb: "Relative rotation (RoPE) and train-short / serve-long extension factor.",
      help: "<strong>Position</strong>RoPE makes scores depend on distance. ExtensionFactor = serve / train (reported 32× for 8K→256K).",
    },
    system: {
      title: "System",
      blurb: "Block compression and the hybrid schedule / long-context roads.",
      help: "<strong>System</strong>Store T/m summaries. Motif DDDGDDDG mixes delta and sparse layers. Stretch vs native training.",
    },
  };

  function getWidgetTip(key) {
    const p = state.pipeline;
    const T = p ? p.T : 0;
    const D = state.D;
    const fact = (state.fact || "").trim() || "4471";
    const tips = {
      tokens: {
        title: "Tokens",
        summary: "Whitespace split into pieces (toy tokenizer).",
        live: "T=" + T + " from your sentence",
        formula: "",
      },
      btd: {
        title: "Embedding grid",
        summary: "Each token → length-D vector; shape [B=1, T, D].",
        live: "[1, " + T + ", " + D + "]",
        formula: "Toy hash vectors — not a trained embedding table.",
      },
      attn: {
        title: "Causal attention",
        summary: "Scores → scale → mask → softmax → weighted V.",
        live: T + "×" + T + " weights on your tokens",
        formula: "Q×Kᵀ / √d → mask future (−∞) → softmax → V\nDark cells = masked future (cannot look ahead).",
      },
      bills: {
        title: "Two bills",
        summary: "Compute ≈ T²; KV memory ≈ T per user.",
        live: "compute " + T * T + ", KV " + T,
        formula: "",
      },
      chunk: {
        title: "Cross-chunk gate",
        summary: "Summary m crosses the boundary; gate mixes it in.",
        live: "fact “" + fact + "”, gate " + state.gate.toFixed(2) + ", α=" + state.alpha,
        formula: "injection = α × gate\nStream OFF or injection < 1% → unavailable\n1%…18% → answer fact · >18% → too loud",
      },
      kv: {
        title: "KV cache bytes",
        summary: "Exact memory bill from yardstick knobs × T.",
        live: "T=" + T + " from sentence",
        formula: "KVCacheBytes = 2 · L · H_KV · d_head · T · B · P_b\n(No extra trailing ×2.)",
      },
      gqa: {
        title: "GQA / MQA",
        summary: "Share KV heads; shrinks cache, not T² prefill.",
        live: state.gqaMode + ", H_Q=" + state.gqaHQ,
        formula: "KVReduction = H_Q / H_KV\nMHA: H_KV=H_Q · GQA: 1<H_KV<H_Q · MQA: H_KV=1",
      },
      linear: {
        title: "Softmax off → S",
        summary: "History folds into fixed-size state S.",
        live: "toy check on first tokens",
        formula: "direct: y = Σ (q·k_j) v_j\nS = Σ v kᵀ ; y = S q\nS ← S + v kᵀ  (same only with softmax OFF)",
      },
      delta: {
        title: "Delta rule",
        summary: "Write only the correction Δ = v − S k.",
        live: "normalize keys " + (state.deltaNorm ? "on" : "off"),
        formula: "v̂ = S k\nΔ = v − v̂\nS ← S + Δ kᵀ\nExact overwrite needs ||k||₂ = 1",
      },
      sparse: {
        title: "Sparse top-k",
        summary: "Softmax on top-k past keys only.",
        live: "k=" + state.sparseK + ", budget ~" + T * state.sparseK,
        formula: "Keep top-k causal scores → softmax on survivors\nBudget ≈ T·k instead of T²",
      },
      rope: {
        title: "RoPE",
        summary: "Rotate Q/K so distance (j−i) enters the score.",
        live: "i=" + state.ropeI + ", j=" + state.ropeJ + ", θ=" + state.ropeTheta.toFixed(2),
        formula: "(R_i q)·(R_j k) depends on (j−i)\nR_m rotates the 2D pair by m·θ",
      },
      drope: {
        title: "Extension factor",
        summary: "Train short, serve long (factor only).",
        live: state.dropeServe + "/" + state.dropeTrain + "=" + (state.dropeServe / state.dropeTrain).toFixed(2) + "×",
        formula: "ExtensionFactor = serve_len / train_len\nV4 story: 262144 / 8192 = 32×\n(Algorithm not simulated.)",
      },
      comp: {
        title: "Compression",
        summary: "Store block summaries: T → ⌈T/m⌉.",
        live: "T=" + state.compT + ", m=" + state.compM + " → " + Math.ceil(state.compT / state.compM),
        formula: "StoredPositions: T → T/m\nOptional top-k block reads (indexer not in size formula)",
      },
      sched: {
        title: "Schedule & roads",
        summary: "DDDGDDDG motif; stretch vs native road.",
        live: "road=" + state.forkRoad,
        formula: "Motif = [D,D,D,G,D,D,D,G]  (6 D + 2 G)\nRoad 1 stretch · Road 2 native long train",
      },
    };
    return tips[key] || null;
  }

  function widgetTipHtml(key) {
    const t = getWidgetTip(key);
    if (!t) return "";
    const formula = (t.formula || "").trim();
    let html =
      "<strong>" +
      escapeHtml(t.title) +
      "</strong>" +
      escapeHtml(t.summary);
    if (formula) {
      html += "<pre class='tip-formula'>" + escapeHtml(formula) + "</pre>";
    }
    html +=
      "<div class='tip-live'>Live: " + escapeHtml(t.live) + "</div>";
    return html;
  }

  function refreshPanelCaptions() {
    document.querySelectorAll("[data-caption]").forEach((el) => {
      const key = el.getAttribute("data-caption");
      const t = getWidgetTip(key);
      if (!t) {
        el.textContent = "";
        return;
      }
      el.innerHTML =
        escapeHtml(t.summary) +
        "<span class='cap-live'>Live: " +
        escapeHtml(t.live) +
        "</span>";
    });
  }

  function applyTallScroll(el, rowHint) {
    if (!el) return;
    const rows = typeof rowHint === "number" ? rowHint : 0;
    const textLines = (el.textContent || "").split(/\n/).length;
    const tall = rows > 12 || textLines > 15 || el.scrollHeight > 280;
    el.classList.toggle("scroll-panel-tall", tall);
  }

  function closeAllTips() {
    document.querySelectorAll(".tip-pop").forEach((pop) => {
      pop.hidden = true;
    });
    document.querySelectorAll(".tip-btn").forEach((btn) => {
      btn.setAttribute("aria-expanded", "false");
    });
  }

  function openTip(btn, html) {
    closeAllTips();
    let pop = btn.parentElement && btn.parentElement.querySelector(".tip-pop");
    if (!pop) {
      pop = document.createElement("div");
      pop.className = "tip-pop";
      btn.parentElement.appendChild(pop);
    }
    pop.innerHTML = html;
    pop.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  }

  function bindTipButtons() {
    document.querySelectorAll(".tip-btn[data-tip]").forEach((btn) => {
      if (btn.dataset.bound === "1") return;
      btn.dataset.bound = "1";
      const host = btn.parentElement;
      const show = () => openTip(btn, widgetTipHtml(btn.getAttribute("data-tip")));
      const hide = () => {
        const pop = host && host.querySelector(".tip-pop");
        if (pop) pop.hidden = true;
        btn.setAttribute("aria-expanded", "false");
      };
      btn.addEventListener("mouseenter", show);
      btn.addEventListener("focus", show);
      btn.addEventListener("blur", () => {
        // Keep open if focus moves into the tip pop
        requestAnimationFrame(() => {
          const pop = host && host.querySelector(".tip-pop");
          if (pop && pop.contains(document.activeElement)) return;
          hide();
        });
      });
      if (host) {
        host.addEventListener("mouseleave", hide);
      } else {
        btn.addEventListener("mouseleave", hide);
      }
    });
  }

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function refreshSampleHint(stage) {
    const el = els.sampleHint || $("sample-hint");
    if (!el) return;
    const id = stage || state.uiStage || "pipeline";
    if (id === "memory") {
      el.innerHTML =
        '<span class="hint-lead">Try: chunk</span>' +
        '<span class="hint-body">' +
        ' fact <code>9910</code> or <code>4471</code> · invoice story in Chunk 1 ·' +
        ' question <code>What was the invoice number?</code> ·' +
        " gate restarts at step 1 — Cross boundary → Ask." +
        "</span>";
      return;
    }
    el.innerHTML =
      '<span class="hint-lead">Try: sentence</span>' +
      '<span class="hint-body">' +
      " <code>The cat sat on the mat</code> · after Run → Tokens / Embedding / Attention / Bills." +
      " Chunk data starts on Memory (Chunk 1 + fact + question)." +
      "</span>";
  }

  function showStage(id) {
    if (!STAGE_META[id]) id = "pipeline";
    state.uiStage = id;
    closeAllTips();
    document.querySelectorAll(".stage-panel").forEach((panel) => {
      const on = panel.getAttribute("data-stage") === id;
      panel.classList.toggle("on", on);
      panel.hidden = !on;
    });
    document.querySelectorAll(".stage-tab").forEach((tab) => {
      tab.classList.toggle("on", tab.getAttribute("data-stage") === id);
    });
    const meta = STAGE_META[id];
    if (els.stageTitle) els.stageTitle.textContent = meta.title;
    if (els.stageBlurb) els.stageBlurb.textContent = meta.blurb;
    if (els.stageInfoPop) {
      els.stageInfoPop.innerHTML = meta.help;
      els.stageInfoPop.hidden = true;
      if (els.stageInfo) els.stageInfo.setAttribute("aria-expanded", "false");
    }
    const idx = STAGE_ORDER.indexOf(id);
    if (els.stagePrev) els.stagePrev.disabled = idx <= 0;
    if (els.stageNext) els.stageNext.disabled = idx >= STAGE_ORDER.length - 1;
    if (id === "memory") {
      state.gateStep = 0;
      if (els.wChunk) renderChunkGate();
    }
    refreshSampleHint(id);
    refreshPanelCaptions();
  }

  function heatColor(v) {
    const t = Math.max(0, Math.min(1, v));
    const r = Math.round(30 + t * 200);
    const g = Math.round(50 + t * 120);
    const b = Math.round(80 + (1 - t) * 100);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function renderTokens(p) {
    const box = els.wTokens;
    if (!p.tokens.length) {
      box.innerHTML = "<p class='muted'>Enter a sentence and click Run.</p>";
      return;
    }
    box.innerHTML =
      "<p><strong>T = " +
      p.T +
      "</strong> tokens</p><div class='chips'>" +
      p.tokens
        .map((t, i) => "<span class='chip'><b>" + i + "</b> " + escapeHtml(t) + "</span>")
        .join("") +
      "</div>";
  }

  function renderBtd(p) {
    const box = els.wBtd;
    if (!p.X.length) {
      box.innerHTML = "<p class='muted'>No matrix yet.</p>";
      return;
    }
    const preview = p.X.slice(0, 6)
      .map((row, i) => {
        const cells = row
          .slice(0, p.D)
          .map((v) => "<td style='background:" + heatColor((v + 1) / 2) + "'>" + v.toFixed(2) + "</td>")
          .join("");
        return "<tr><th>" + escapeHtml(p.tokens[i]) + "</th>" + cells + "</tr>";
      })
      .join("");
    box.innerHTML =
      "<p>Shape sketch: <code>[B=1, T=" +
      p.T +
      ", D=" +
      p.D +
      "]</code> (toy vectors from token text — not a real embedding table)</p>" +
      "<div class='table-wrap'><table class='matrix'><tbody>" +
      preview +
      "</tbody></table></div>";
  }

  function renderAttention(p) {
    const box = els.wAttn;
    const W = p.attn.weights;
    if (!W.length) {
      box.innerHTML = "<p class='muted'>No attention yet.</p>";
      applyTallScroll(box, 0);
      return;
    }
    const head =
      "<tr><th></th>" +
      p.tokens.map((t) => "<th>" + escapeHtml(t.slice(0, 8)) + "</th>").join("") +
      "</tr>";
    const body = W.map((row, i) => {
      return (
        "<tr><th>" +
        escapeHtml(p.tokens[i].slice(0, 8)) +
        "</th>" +
        row
          .map((v) => {
            const dead = v < 1e-8;
            return (
              "<td title='" +
              v.toFixed(3) +
              "' style='background:" +
              (dead ? "#1a1f2a" : heatColor(v)) +
              "'>" +
              (dead ? "·" : v.toFixed(2)) +
              "</td>"
            );
          })
          .join("") +
        "</tr>"
      );
    }).join("");
    box.innerHTML =
      "<p>Scale = 1/√D = 1/" +
      p.attn.scale.toFixed(2) +
      ". Row = query token.</p>" +
      "<div class='table-wrap'><table class='matrix heat'>" +
      head +
      body +
      "</table></div>" +
      "<pre class='flow-mini'>Q×K → scores → /√d → mask → softmax → weighted V</pre>";
    applyTallScroll(box, W.length);
  }

  function renderBills(p) {
    const b = p.bills;
    els.wBills.innerHTML =
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Compute bill</strong><div class='big'>" +
      b.computeUnits +
      "</div><span>" +
      b.computeLabel +
      " (T=" +
      b.T +
      ")</span></div>" +
      "<div class='bill'><strong>KV memory bill</strong><div class='big'>" +
      b.kvUnits +
      "</div><span>" +
      b.kvLabel +
      "</span></div>" +
      "</div>" +
      "<p class='muted'>Lengthen the sentence to grow T — T² rises faster than KV.</p>";
  }

  function renderChunkGate() {
    const box = els.wChunk;
    const g = window.DemoPipeline.chunkGate({
      fact: state.fact.trim(),
      streamOn: state.streamOn,
      gate: state.gate,
      alpha: state.alpha,
    });

    const stepLabel = ["step 1 of 3", "step 2 of 3", "step 3 of 3"][state.gateStep];
    const btnLabel = ["Cross the boundary", "Ask in chunk 2", "Replay"][state.gateStep];

    let memory = "one empty vector";
    let write = "Nothing written yet.";
    let answerClass = "wait";
    let answer = "Ask after crossing the boundary.";
    let readout = "Waiting for the boundary.";

    if (state.gateStep >= 1) {
      write = state.streamOn
        ? "Chunk 1 writes final hidden state (summary)."
        : "Chunk ends with no memory stream.";
      memory = state.streamOn
        ? "fact “" + (state.fact.trim() || "?") + "” · one vector"
        : "nothing carried";
    }
    if (state.gateStep === 1) {
      readout = "Chunk 2 has not used the vector yet.";
      answer = "Fact is inside the vector — or gone.";
    }
    if (state.gateStep === 2) {
      readout =
        (state.streamOn
          ? "Current token + " + (g.injection * 100).toFixed(1) + "% memory (α " + state.alpha + " × gate " + state.gate.toFixed(2) + ")"
          : "No cross-chunk state.") +
        "<br><span class='muted'>stop_gradient(m): Chunk 2 can read m; training does not backprop into Chunk 1 through m.</span>";
      answerClass = g.verdict === "good" ? "good" : g.verdict === "wait" ? "wait" : "bad";
      answer = g.message;
    }

    box.innerHTML =
      "<div class='chunk-controls'>" +
      "<div class='seg' id='streamSeg'>" +
      "<button type='button' data-on='0' class='" +
      (!state.streamOn ? "on" : "") +
      "'>OFF</button>" +
      "<button type='button' data-on='1' class='" +
      (state.streamOn ? "on" : "") +
      "'>ON</button></div>" +
      "<label class='gate-lab'>Gate <output id='gateOut'>" +
      state.gate.toFixed(2) +
      "</output><input id='gateRange' type='range' min='0' max='100' value='" +
      Math.round(state.gate * 100) +
      "'/></label>" +
      "<button type='button' class='primary' id='gateNext'>" +
      btnLabel +
      "</button>" +
      "<span class='stage'>" +
      stepLabel +
      "</span></div>" +
      "<div class='chunk-flow'>" +
      "<article class='chunk " +
      (state.gateStep === 0 ? "active" : "") +
      "'><h4>Chunk 1</h4><p>" +
      escapeHtml(state.chunk1 || "(empty)") +
      "</p><p class='fact-line'>Remember fact: <strong>" +
      escapeHtml(state.fact || "?") +
      "</strong></p></article>" +
      "<article class='bridge " +
      (state.gateStep === 1 ? "active" : "") +
      "'><h4>Boundary</h4><p>" +
      escapeHtml(write) +
      "</p><div class='memory'>" +
      escapeHtml(memory) +
      "</div><p class='stopg'>× backward gradient stops here<br>Next chunk can read m; loss does not backprop into chunk 1.</p></article>" +
      "<article class='chunk " +
      (state.gateStep === 2 ? "active" : "") +
      "'><h4>Chunk 2</h4><p><strong>" +
      escapeHtml(state.question || "What was the fact?") +
      "</strong></p><p class='readout'>" +
      readout +
      "</p><div class='answer " +
      answerClass +
      "'>" +
      escapeHtml(answer) +
      "</div></article></div>";

    // re-bind controls after innerHTML
    box.querySelectorAll("#streamSeg button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.streamOn = btn.getAttribute("data-on") === "1";
        renderChunkGate();
      });
    });
    const range = box.querySelector("#gateRange");
    if (range) {
      range.addEventListener("input", () => {
        state.gate = Number(range.value) / 100;
        renderChunkGate();
      });
    }
    const next = box.querySelector("#gateNext");
    if (next) {
      next.addEventListener("click", () => {
        state.gateStep = state.gateStep === 2 ? 0 : state.gateStep + 1;
        renderChunkGate();
      });
    }
    refreshPanelCaptions();
  }

  function fmtBytes(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + " TB";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + " KB";
    return String(n) + " B";
  }

  function renderKvCache(p) {
    const box = els.wKv;
    const T = p.T || 0;
    const kv = window.DemoPipeline.kvCacheBytes({
      L: state.kvL,
      H_KV: state.kvH,
      d_head: state.kvDhead,
      T: T,
      B: state.kvB,
      P_b: state.kvPb,
    });
    const yardT = 32768;
    const yard = window.DemoPipeline.kvCacheBytes({
      L: 48,
      H_KV: 8,
      d_head: 128,
      T: yardT,
      B: 1,
      P_b: 2,
    });

    box.innerHTML =
      "<p class='wire-lead'>Yardstick knobs: L, H_KV, d_head, precision (T = " +
      T +
      ").</p>" +
      "<div class='chunk-controls wire-controls'>" +
      "<label>L <output>" +
      state.kvL +
      "</output><input id='kvL' type='range' min='1' max='96' value='" +
      state.kvL +
      "'/></label>" +
      "<label>H_KV <output>" +
      state.kvH +
      "</output><input id='kvH' type='range' min='1' max='64' value='" +
      state.kvH +
      "'/></label>" +
      "<label>d_head <output>" +
      state.kvDhead +
      "</output><input id='kvD' type='range' min='16' max='256' step='16' value='" +
      state.kvDhead +
      "'/></label>" +
      "<label>B users <output>" +
      state.kvB +
      "</output><input id='kvB' type='range' min='1' max='16' value='" +
      state.kvB +
      "'/></label>" +
      "<label>P_b <select id='kvPb'><option value='2'" +
      (state.kvPb === 2 ? " selected" : "") +
      ">2 (bf16)</option><option value='4'" +
      (state.kvPb === 4 ? " selected" : "") +
      ">4 (fp32)</option></select></label>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Your sentence cache</strong><div class='big'>" +
      fmtBytes(kv.bytes) +
      "</div><span>T=" +
      T +
      " · " +
      kv.gib.toFixed(4) +
      " GiB</span></div>" +
      "<div class='bill'><strong>Yardstick @ T=32K</strong><div class='big'>" +
      fmtBytes(yard.bytes) +
      "</div><span>~" +
      yard.gib.toFixed(2) +
      " GiB / user (L=48…)</span></div>" +
      "</div>" +
      "<pre class='flow-mini'>KVCacheBytes = 2 · L · H_KV · d_head · T · B · P_b\n" +
      "             = 2 · " +
      state.kvL +
      " · " +
      state.kvH +
      " · " +
      state.kvDhead +
      " · " +
      T +
      " · " +
      state.kvB +
      " · " +
      state.kvPb +
      " = " +
      kv.bytes.toLocaleString() +
      " bytes</pre>" +
      "<p class='muted'>No GPU allocation — illustration only.</p>";

    const bindRange = (id, key) => {
      const el = box.querySelector("#" + id);
      if (!el) return;
      el.addEventListener("input", () => {
        state[key] = Number(el.value);
        renderKvCache(state.pipeline || { T: 0 });
      });
    };
    bindRange("kvL", "kvL");
    bindRange("kvH", "kvH");
    bindRange("kvD", "kvDhead");
    bindRange("kvB", "kvB");
    const pb = box.querySelector("#kvPb");
    if (pb) {
      pb.addEventListener("change", () => {
        state.kvPb = Number(pb.value);
        renderKvCache(state.pipeline || { T: 0 });
      });
    }
  }

  function renderGqa(p) {
    const box = els.wGqa;
    const T = p.T || 0;
    const g = window.DemoPipeline.gqaLayout({
      mode: state.gqaMode,
      H_Q: state.gqaHQ,
      H_KV: state.gqaHKV,
    });
    state.gqaHKV = g.H_KV;
    const mhaBytes = window.DemoPipeline.kvCacheBytes({
      L: state.kvL,
      H_KV: g.H_Q,
      d_head: state.kvDhead,
      T: T,
      B: state.kvB,
      P_b: state.kvPb,
    });
    const thisBytes = window.DemoPipeline.kvCacheBytes({
      L: state.kvL,
      H_KV: g.H_KV,
      d_head: state.kvDhead,
      T: T,
      B: state.kvB,
      P_b: state.kvPb,
    });

    let headMap = "";
    for (let kv = 0; kv < g.H_KV; kv++) {
      const qs = [];
      for (let q = 0; q < g.groupSize; q++) {
        qs.push("<span class='chip head-q'>Q" + (kv * g.groupSize + q + 1) + "</span>");
      }
      headMap +=
        "<div class='gqa-row'><span class='chip head-kv'>KV" +
        (kv + 1) +
        "</span><span class='gqa-arrow'>←</span>" +
        qs.join("") +
        "</div>";
    }

    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<div class='seg' id='gqaMode'>" +
      ["MHA", "GQA", "MQA"]
        .map(
          (m) =>
            "<button type='button' data-mode='" +
            m +
            "' class='" +
            (state.gqaMode === m ? "on" : "") +
            "'>" +
            m +
            "</button>"
        )
        .join("") +
      "</div>" +
      "<label>H_Q <output>" +
      g.H_Q +
      "</output><input id='gqaHQ' type='range' min='2' max='32' step='2' value='" +
      g.H_Q +
      "'/></label>" +
      (state.gqaMode === "GQA"
        ? "<label>H_KV <output>" +
          g.H_KV +
          "</output><input id='gqaHKV' type='range' min='1' max='" +
          Math.max(1, g.H_Q - 1) +
          "' value='" +
          g.H_KV +
          "'/></label>"
        : "") +
      "</div>" +
      "<p><strong>" +
      escapeHtml(g.label) +
      "</strong> · reduction <code>H_Q/H_KV = " +
      g.reduction.toFixed(2) +
      "×</code> · T=" +
      T +
      "</p>" +
      "<div class='gqa-map'>" +
      headMap +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>If MHA (H_KV=H_Q)</strong><div class='big'>" +
      fmtBytes(mhaBytes.bytes) +
      "</div><span>same L,d_head,T as C6</span></div>" +
      "<div class='bill'><strong>This mode</strong><div class='big'>" +
      fmtBytes(thisBytes.bytes) +
      "</div><span>" +
      (mhaBytes.bytes ? ((thisBytes.bytes / mhaBytes.bytes) * 100).toFixed(0) : "—") +
      "% of MHA cache</span></div>" +
      "</div>" +
      "<pre class='flow-mini'>KVReduction = H_Q / H_KV = " +
      g.H_Q +
      " / " +
      g.H_KV +
      " = " +
      g.reduction.toFixed(2) +
      "\nGQA shrinks cache slope — long context still needs linear, sparse, or compression.</pre>";

    box.querySelectorAll("#gqaMode button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.gqaMode = btn.getAttribute("data-mode");
        renderGqa(state.pipeline || { T: 0 });
      });
    });
    const hq = box.querySelector("#gqaHQ");
    if (hq) {
      hq.addEventListener("input", () => {
        state.gqaHQ = Number(hq.value);
        renderGqa(state.pipeline || { T: 0 });
      });
    }
    const hkv = box.querySelector("#gqaHKV");
    if (hkv) {
      hkv.addEventListener("input", () => {
        state.gqaHKV = Number(hkv.value);
        renderGqa(state.pipeline || { T: 0 });
      });
    }
    refreshPanelCaptions();
  }

  function renderLinear(p) {
    const box = els.wLinear;
    const lin = window.DemoPipeline.linearSoftmaxOff(p.X || []);
    const last = lin.last;
    const trajRows = lin.traj
      .map(
        (step) =>
          "<tr><td>t=" +
          step.t +
          "</td><td>" +
          step.direct.join(", ") +
          "</td><td>" +
          step.y.join(", ") +
          "</td><td>" +
          (step.match ? "✓ same" : "≠") +
          "</td></tr>"
      )
      .join("");
    const Shtml = last.Sflat
      .map((row) => "<tr>" + row.map((v) => "<td>" + v + "</td>").join("") + "</tr>")
      .join("");

    box.innerHTML =
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Direct (softmax off)</strong><div class='big' style='font-size:1.1rem'>" +
      escapeHtml(lin.formulaDirect) +
      "</div><span>visit every past k,v</span></div>" +
      "<div class='bill'><strong>Regroup (session)</strong><div class='big' style='font-size:1.1rem'>" +
      escapeHtml(lin.formulaState) +
      "</div><span>S is d_v×d_k · fixed</span></div>" +
      "</div>" +
      "<p>Toy from your sentence vectors (first " +
      lin.nSteps +
      " tokens, dims d_k=d_v=" +
      lin.dK +
      "). Softmax-off paths must match:</p>" +
      "<div class='table-wrap'><table class='matrix'><thead><tr><th>prefix</th><th>direct y</th><th>S q</th><th>check</th></tr></thead><tbody>" +
      trajRows +
      "</tbody></table></div>" +
      "<p>Final state matrix S (" +
      lin.dV +
      "×" +
      lin.dK +
      "):</p>" +
      "<div class='table-wrap'><table class='matrix'><tbody>" +
      Shtml +
      "</tbody></table></div>" +
      "<pre class='flow-mini'>S ← S + v kᵀ   (additive write)\ny = S q         (session convention — not q S)\n" +
      escapeHtml(lin.note) +
      "</pre>" +
      "<p class='muted'>Not a production linear-attention stack — teaching identity only.</p>";
    applyTallScroll(box, lin.nSteps || 0);
  }

  function renderDelta(p) {
    const box = els.wDelta;
    const d = window.DemoPipeline.deltaRuleWrite(p.X || [], state.deltaNorm);
    const rows = d.traj
      .map(
        (s) =>
          "<tr><td>t=" +
          s.t +
          "</td><td>" +
          s.vHat.join(", ") +
          "</td><td>" +
          s.delta.join(", ") +
          "</td><td>" +
          s.afterDelta.join(", ") +
          "</td><td>" +
          s.afterAdd.join(", ") +
          "</td><td>" +
          (s.overwriteOk ? "✓ ≈ v" : "loose") +
          "</td></tr>"
      )
      .join("");
    const last = d.last;

    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<div class='seg' id='deltaNorm'>" +
      "<button type='button' data-on='0' class='" +
      (!state.deltaNorm ? "on" : "") +
      "'>keys raw</button>" +
      "<button type='button' data-on='1' class='" +
      (state.deltaNorm ? "on" : "") +
      "'>||k||=1</button></div>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Delta write</strong><div class='big' style='font-size:1rem'>" +
      escapeHtml(d.formula) +
      "</div><span>err after " +
      last.errAfter +
      "</span></div>" +
      "<div class='bill'><strong>Additive only (C8)</strong><div class='big' style='font-size:1rem'>S ← S + v kᵀ</div><span>err after " +
      last.errAdd +
      " (no overwrite)</span></div>" +
      "</div>" +
      "<div class='table-wrap'><table class='matrix'><thead><tr><th>step</th><th>read v̂</th><th>Δ</th><th>S_Δ k</th><th>S_add k</th><th>delta?</th></tr></thead><tbody>" +
      rows +
      "</tbody></table></div>" +
      "<pre class='flow-mini'>norm(k)=" +
      (d.normalizeKeys ? "on" : "off") +
      " · last ||k||=" +
      last.knorm +
      "\n" +
      escapeHtml(d.note) +
      "</pre>" +
      "<p class='muted'>From your sentence vectors (first " +
      d.nSteps +
      " tokens).</p>";

    applyTallScroll(box, d.nSteps || 0);
    box.querySelectorAll("#deltaNorm button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.deltaNorm = btn.getAttribute("data-on") === "1";
        renderDelta(state.pipeline || { X: [] });
        refreshPanelCaptions();
      });
    });
  }

  function renderSparse(p) {
    const box = els.wSparse;
    const T = p.T || 0;
    if (!T) {
      box.innerHTML = "<p class='muted'>Enter a sentence and click Run.</p>";
      applyTallScroll(box, 0);
      return;
    }
    const kMax = Math.max(1, T);
    if (state.sparseK > kMax) state.sparseK = kMax;
    if (state.sparseK < 1) state.sparseK = Math.min(3, kMax);
    const sp = window.DemoPipeline.sparseTopKAttention(p.X, state.sparseK);
    const tokens = p.tokens || [];
    const head =
      "<tr><th></th>" +
      tokens.map((t) => "<th>" + escapeHtml(String(t).slice(0, 8)) + "</th>").join("") +
      "</tr>";
    const body = sp.weights
      .map((row, i) => {
        return (
          "<tr><th>" +
          escapeHtml(String(tokens[i] || i).slice(0, 8)) +
          "</th>" +
          row
            .map((v) => {
              const dead = v < 1e-8;
              return (
                "<td title='" +
                v.toFixed(3) +
                "' style='background:" +
                (dead ? "#1a1f2a" : heatColor(v)) +
                "'>" +
                (dead ? "·" : v.toFixed(2)) +
                "</td>"
              );
            })
            .join("") +
          "</tr>"
        );
      })
      .join("");

    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<label>k <output>" +
      sp.k +
      "</output><input id='sparseK' type='range' min='1' max='" +
      kMax +
      "' value='" +
      sp.k +
      "'/></label>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Dense comparisons</strong><div class='big'>" +
      sp.denseComparisons +
      "</div><span>≈ T² (T=" +
      T +
      ")</span></div>" +
      "<div class='bill'><strong>Sparse budget</strong><div class='big'>" +
      sp.sparseComparisons +
      "</div><span>≈ T·k (k=" +
      sp.k +
      ")</span></div>" +
      "</div>" +
      "<div class='table-wrap'><table class='matrix heat'>" +
      head +
      body +
      "</table></div>" +
      "<pre class='flow-mini'>scores → keep top-k (causal j≤i) → softmax on survivors → weighted V\nMissed key → wrong answer (the sparse trade-off).</pre>";

    applyTallScroll(box, T);
    const range = box.querySelector("#sparseK");
    if (range) {
      range.addEventListener("input", () => {
        state.sparseK = Number(range.value);
        renderSparse(state.pipeline || { T: 0, X: [], tokens: [] });
        refreshPanelCaptions();
      });
    }
  }

  function renderRope(p) {
    const box = els.wRope;
    const T = p.T || 0;
    if (T >= 2) {
      if (state.ropeI > T - 1) state.ropeI = 0;
      if (state.ropeJ > T - 1) state.ropeJ = Math.min(1, T - 1);
    }
    const r = window.DemoPipeline.ropeDemo({
      X: p.X || [],
      i: state.ropeI,
      j: state.ropeJ,
      theta: state.ropeTheta,
    });
    const sweepRows = r.sweep
      .map((s) => "<tr><td>" + s.delta + "</td><td>" + s.score + "</td></tr>")
      .join("");
    const iMax = Math.max(0, T - 1);
    const jMax = Math.max(0, T - 1);

    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<label>i (query) <output>" +
      r.i +
      "</output><input id='ropeI' type='range' min='0' max='" +
      iMax +
      "' value='" +
      r.i +
      "'/></label>" +
      "<label>j (key) <output>" +
      r.j +
      "</output><input id='ropeJ' type='range' min='0' max='" +
      jMax +
      "' value='" +
      r.j +
      "'/></label>" +
      "<label>θ <output>" +
      r.theta.toFixed(2) +
      "</output><input id='ropeTh' type='range' min='10' max='200' value='" +
      Math.round(r.theta * 100) +
      "'/></label>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Unrotated q·k</strong><div class='big'>" +
      r.unrotatedDot +
      "</div><span>same for any distance</span></div>" +
      "<div class='bill'><strong>Rotated (R_i q)·(R_j k)</strong><div class='big'>" +
      r.rotatedDot +
      "</div><span>Δpos = j−i = " +
      r.deltaPos +
      "</span></div>" +
      "</div>" +
      "<p>2D pairs from your tokens (dims 0–1). Sweep Δpos with fixed content:</p>" +
      "<div class='table-wrap'><table class='matrix'><thead><tr><th>Δpos</th><th>rotated score</th></tr></thead><tbody>" +
      sweepRows +
      "</tbody></table></div>" +
      "<pre class='flow-mini'>" +
      escapeHtml(r.formula) +
      "\nq=[" +
      r.qPair.join(", ") +
      "] → R_i q=[" +
      r.qR.join(", ") +
      "]\nk=[" +
      r.kPair.join(", ") +
      "] → R_j k=[" +
      r.kR.join(", ") +
      "]\n" +
      escapeHtml(r.note) +
      "</pre>" +
      "<p class='muted'>One 2D pair + one θ — not the full multi-frequency stack.</p>";

    const bind = (id, key, scale) => {
      const el = box.querySelector("#" + id);
      if (!el) return;
      el.addEventListener("input", () => {
        state[key] = scale ? Number(el.value) / scale : Number(el.value);
        renderRope(state.pipeline || { T: 0, X: [] });
        refreshPanelCaptions();
      });
    };
    bind("ropeI", "ropeI");
    bind("ropeJ", "ropeJ");
    bind("ropeTh", "ropeTheta", 100);
  }

  function renderDrope() {
    const box = els.wDrope;
    const d = window.DemoPipeline.dropeExtension({
      trainLen: state.dropeTrain,
      serveLen: state.dropeServe,
    });
    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<label>Train T <output>" +
      d.trainLen +
      "</output><input id='dropeTrain' type='range' min='1024' max='32768' step='1024' value='" +
      d.trainLen +
      "'/></label>" +
      "<label>Serve T <output>" +
      d.serveLen +
      "</output><input id='dropeServe' type='range' min='8192' max='262144' step='8192' value='" +
      d.serveLen +
      "'/></label>" +
      "<button type='button' class='primary' id='dropeV4'>Reset V4 (8K→256K)</button>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>ExtensionFactor</strong><div class='big'>" +
      d.factorLabel +
      "×</div><span>" +
      d.serveLen.toLocaleString() +
      " / " +
      d.trainLen.toLocaleString() +
      "</span></div>" +
      "<div class='bill'><strong>V4 yardstick</strong><div class='big'>" +
      (d.reportedV4 ? "32×" : "—") +
      "</div><span>" +
      (d.reportedV4 ? "matches reported 256K/8K" : "drag to 8192 & 262144") +
      "</span></div>" +
      "</div>" +
      "<pre class='flow-mini'>" +
      escapeHtml(d.formula) +
      " = " +
      d.factorLabel +
      "\n" +
      escapeHtml(d.flow) +
      "\n" +
      escapeHtml(d.note) +
      "</pre>" +
      "<p class='muted'>Illustration only — no long-context tensor allocation.</p>";

    const tTrain = box.querySelector("#dropeTrain");
    if (tTrain) {
      tTrain.addEventListener("input", () => {
        state.dropeTrain = Number(tTrain.value);
        if (state.dropeServe < state.dropeTrain) state.dropeServe = state.dropeTrain;
        renderDrope();
      });
    }
    const tServe = box.querySelector("#dropeServe");
    if (tServe) {
      tServe.addEventListener("input", () => {
        state.dropeServe = Number(tServe.value);
        renderDrope();
      });
    }
    const reset = box.querySelector("#dropeV4");
    if (reset) {
      reset.addEventListener("click", () => {
        state.dropeTrain = 8192;
        state.dropeServe = 262144;
        renderDrope();
      });
    }
    refreshPanelCaptions();
  }

  function renderCompression(p) {
    const box = els.wComp;
    const c = window.DemoPipeline.sequenceCompression({
      T: state.compT,
      m: state.compM,
      topKBlocks: state.compTopK,
      sentenceT: p.T || 0,
    });
    const sentT = p.T || 0;
    const sentBlocks = sentT === 0 ? 0 : Math.ceil(sentT / c.m);
    const pct = sentBlocks <= 1 ? 100 : Math.round((c.m / Math.max(sentT, 1)) * 100);
    const pct2 = Math.max(0, 100 - pct);

    box.innerHTML =
      "<div class='chunk-controls wire-controls'>" +
      "<label>T <output>" +
      c.T +
      "</output><input id='compT' type='range' min='64' max='65536' step='64' value='" +
      c.T +
      "'/></label>" +
      "<label>m <output>" +
      c.m +
      "</output><input id='compM' type='range' min='2' max='64' value='" +
      c.m +
      "'/></label>" +
      "<label>top-k <output>" +
      c.topKBlocks +
      "</output><input id='compK' type='range' min='1' max='" +
      Math.max(1, c.nBlocks) +
      "' value='" +
      c.topKBlocks +
      "'/></label>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>Full slots</strong><div class='big'>" +
      c.fullSlots.toLocaleString() +
      "</div><span>T tokens</span></div>" +
      "<div class='bill'><strong>Summaries</strong><div class='big'>" +
      c.summarySlots.toLocaleString() +
      "</div><span>~" +
      c.reduction +
      "× · read ≤ " +
      c.readSlots +
      "</span></div>" +
      "</div>" +
      "<p class='range-meta'>Your sentence: <strong>T=" +
      sentT +
      "</strong> → <strong>" +
      sentBlocks +
      "</strong> block" +
      (sentBlocks === 1 ? "" : "s") +
      " (m=" +
      c.m +
      ") · range 0…" +
      Math.max(0, sentT - 1) +
      "</p>" +
      (sentT
        ? "<div class='range-bar' title='tokens per block vs remainder'><span class='seg-a' style='width:" +
          pct +
          "%'></span><span class='seg-b' style='width:" +
          pct2 +
          "%'></span></div>"
        : "") +
      "<pre class='flow-mini'>" +
      escapeHtml(c.formula) +
      " → " +
      c.summarySlots +
      " summaries\n" +
      escapeHtml(c.note) +
      "</pre>";

    const bind = (id, key) => {
      const el = box.querySelector("#" + id);
      if (!el) return;
      el.addEventListener("input", () => {
        state[key] = Number(el.value);
        renderCompression(state.pipeline || { T: 0 });
      });
    };
    bind("compT", "compT");
    bind("compM", "compM");
    bind("compK", "compTopK");
    refreshPanelCaptions();
  }

  function renderSchedule() {
    const box = els.wSched;
    const s = window.DemoPipeline.scheduleAndFork({ road: state.forkRoad });
    const motifHtml = s.motif
      .map(
        (letter, i) =>
          "<span class='chip " +
          (letter === "D" ? "head-kv" : "head-q") +
          "' title='" +
          (letter === "D" ? "DeltaNet state" : "sparse G") +
          "'>" +
          letter +
          (i + 1) +
          "</span>"
      )
      .join("");
    const sel = s.selected;

    box.innerHTML =
      "<p class='range-meta'><strong>" +
      escapeHtml(s.motifStr) +
      "</strong> · " +
      s.nD +
      " D + " +
      s.nG +
      " G</p>" +
      "<div class='chips motif-row'>" +
      motifHtml +
      "</div>" +
      "<div class='chunk-controls wire-controls'>" +
      "<div class='seg' id='forkRoad'>" +
      "<button type='button' data-road='stretch' class='" +
      (state.forkRoad === "stretch" ? "on" : "") +
      "'>Road 1 stretch</button>" +
      "<button type='button' data-road='native' class='" +
      (state.forkRoad === "native" ? "on" : "") +
      "'>Road 2 native</button></div>" +
      "</div>" +
      "<div class='bill-grid'>" +
      "<div class='bill'><strong>" +
      escapeHtml(sel.title) +
      "</strong><div class='big' style='font-size:1rem'>Buy</div><span>" +
      escapeHtml(sel.buy) +
      "</span></div>" +
      "<div class='bill'><strong>Trade-off</strong><div class='big' style='font-size:1rem'>Give up</div><span>" +
      escapeHtml(sel.give) +
      "</span></div>" +
      "</div>" +
      "<pre class='flow-mini'>" +
      escapeHtml(s.formula) +
      "\nWhen: " +
      escapeHtml(sel.when) +
      "\n" +
      escapeHtml(s.note) +
      "\n" +
      escapeHtml(s.targetLabel) +
      "</pre>";

    box.querySelectorAll("#forkRoad button").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.forkRoad = btn.getAttribute("data-road");
        renderSchedule();
      });
    });
    refreshPanelCaptions();
  }

  function updateAll() {
    state.sentence = els.sentence.value;
    state.chunk1 = els.chunk1.value;
    state.fact = els.fact.value;
    state.question = els.question.value;
    state.pipeline = window.DemoPipeline.runPipeline(state.sentence, state.D);
    renderTokens(state.pipeline);
    renderBtd(state.pipeline);
    renderAttention(state.pipeline);
    renderBills(state.pipeline);
    renderChunkGate();
    renderKvCache(state.pipeline);
    renderGqa(state.pipeline);
    renderLinear(state.pipeline);
    renderDelta(state.pipeline);
    renderSparse(state.pipeline);
    renderRope(state.pipeline);
    renderDrope();
    renderCompression(state.pipeline);
    renderSchedule();
    const when = new Date().toLocaleTimeString();
    els.status.textContent = "Last run · T=" + state.pipeline.T + " · D=" + state.D;
    els.status.title = "Last updated " + when;
    els.status.setAttribute(
      "aria-label",
      "Last run T=" + state.pipeline.T + ", D=" + state.D + ", at " + when
    );
    refreshPanelCaptions();
  }

  async function boot() {
    els.sentence = $("sentence");
    els.chunk1 = $("chunk1");
    els.fact = $("fact");
    els.question = $("question");
    els.wTokens = $("w-tokens");
    els.wBtd = $("w-btd");
    els.wAttn = $("w-attn");
    els.wBills = $("w-bills");
    els.wChunk = $("w-chunk");
    els.wKv = $("w-kv");
    els.wGqa = $("w-gqa");
    els.wLinear = $("w-linear");
    els.wDelta = $("w-delta");
    els.wSparse = $("w-sparse");
    els.wRope = $("w-rope");
    els.wDrope = $("w-drope");
    els.wComp = $("w-comp");
    els.wSched = $("w-sched");
    els.status = $("status");
    els.sampleHint = $("sample-hint");
    els.btnUpdate = $("btn-update");
    els.stageTitle = $("stage-title");
    els.stageBlurb = $("stage-blurb");
    els.stageInfo = $("stage-info");
    els.stageInfoPop = $("stage-info-pop");
    els.stagePrev = $("stage-prev");
    els.stageNext = $("stage-next");

    let data;
    try {
      const res = await fetch("data/steps.json");
      data = await res.json();
    } catch (err) {
      data = {
        schema: "a08.demo_asgn_08.1",
        D: 8,
        alpha: 0.391,
        default_sentence: "The cat sat on the mat",
        default_chunk1: "The shipment belongs in bay 3. Its invoice number is 4471.",
        default_fact: "4471",
        default_chunk2_question: "What was the invoice number?",
        steps: [],
        widgets: [
          "tokens",
          "btd",
          "attention",
          "two_bills",
          "chunk_gate",
          "kv_cache",
          "gqa",
          "linear_softmax_off",
          "delta",
          "sparse_topk",
          "rope",
          "drope",
          "compression",
          "schedule_fork",
        ],
      };
      console.warn("steps.json fetch failed; using embedded defaults", err);
    }
    state.steps = data;
    state.D = state.steps.D || 8;
    state.alpha = state.steps.alpha || 0.391;
    els.sentence.value = state.steps.default_sentence || "The cat sat on the mat";
    els.chunk1.value = state.steps.default_chunk1 || "";
    els.fact.value = state.steps.default_fact || "";
    els.question.value = state.steps.default_chunk2_question || "";

    document.querySelectorAll(".stage-tab").forEach((tab) => {
      tab.addEventListener("click", () => showStage(tab.getAttribute("data-stage")));
    });
    if (els.stagePrev) {
      els.stagePrev.addEventListener("click", () => {
        const i = STAGE_ORDER.indexOf(state.uiStage);
        if (i > 0) showStage(STAGE_ORDER[i - 1]);
      });
    }
    if (els.stageNext) {
      els.stageNext.addEventListener("click", () => {
        const i = STAGE_ORDER.indexOf(state.uiStage);
        if (i < STAGE_ORDER.length - 1) showStage(STAGE_ORDER[i + 1]);
      });
    }
    if (els.stageInfo && els.stageInfoPop) {
      const showStageTip = () => {
        els.stageInfoPop.innerHTML = STAGE_META[state.uiStage].help;
        els.stageInfoPop.hidden = false;
        els.stageInfo.setAttribute("aria-expanded", "true");
      };
      const hideStageTip = () => {
        els.stageInfoPop.hidden = true;
        els.stageInfo.setAttribute("aria-expanded", "false");
      };
      els.stageInfo.addEventListener("mouseenter", showStageTip);
      els.stageInfo.addEventListener("mouseleave", hideStageTip);
      els.stageInfo.addEventListener("focus", showStageTip);
      els.stageInfo.addEventListener("blur", hideStageTip);
    }

    bindTipButtons();
    showStage("pipeline");
    els.btnUpdate.addEventListener("click", updateAll);
    els.sentence.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) updateAll();
    });
    updateAll();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
