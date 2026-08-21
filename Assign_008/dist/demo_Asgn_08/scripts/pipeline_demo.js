/**
 * Toy browser pipeline: sentence → tokens → fake [T,D] → causal attention.
 * Grade-10 demo only — not a trained model.
 */
(function (global) {
  "use strict";

  function hashSeed(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function tokenize(text) {
    const cleaned = (text || "").trim();
    if (!cleaned) return [];
    return cleaned.split(/\s+/).filter(Boolean).slice(0, 24);
  }

  function embedTokens(tokens, D) {
    const rows = [];
    tokens.forEach((tok, i) => {
      const rnd = mulberry32(hashSeed(tok + "#" + i));
      const row = [];
      for (let d = 0; d < D; d++) {
        row.push(rnd() * 2 - 1);
      }
      // L2 normalize for stable scores
      let n = Math.sqrt(row.reduce((s, v) => s + v * v, 0)) || 1;
      rows.push(row.map((v) => v / n));
    });
    return rows;
  }

  function matMulAt(a, bT) {
    // a [T,D], bT [T,D] → scores [T,T] via a @ bT^T
    const T = a.length;
    const D = a[0].length;
    const out = Array.from({ length: T }, () => Array(T).fill(0));
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        let s = 0;
        for (let d = 0; d < D; d++) s += a[i][d] * bT[j][d];
        out[i][j] = s;
      }
    }
    return out;
  }

  function causalAttention(X) {
    const T = X.length;
    if (!T) {
      return { scores: [], weights: [], out: [], scale: 1 };
    }
    const D = X[0].length;
    const scale = Math.sqrt(D);
    const scores = matMulAt(X, X);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        scores[i][j] = scores[i][j] / scale;
        if (j > i) scores[i][j] = -1e9;
      }
    }
    const weights = scores.map((row) => {
      const m = Math.max(...row.filter((v) => v > -1e8), -1e8);
      const ex = row.map((v) => (v <= -1e8 ? 0 : Math.exp(v - m)));
      const z = ex.reduce((s, v) => s + v, 0) || 1;
      return ex.map((v) => v / z);
    });
    const out = Array.from({ length: T }, () => Array(D).fill(0));
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        for (let d = 0; d < D; d++) {
          out[i][d] += weights[i][j] * X[j][d];
        }
      }
    }
    return { scores, weights, out, scale };
  }

  function twoBills(T) {
    const computeUnits = T * T;
    const kvUnits = T; // per user sketch
    return { T, computeUnits, kvUnits, computeLabel: "≈ T² comparisons", kvLabel: "≈ T · users stored" };
  }

  function chunkGate({ fact, streamOn, gate, alpha }) {
    const injection = streamOn ? alpha * gate : 0;
    let verdict = "wait";
    let message = "Cross the boundary, then ask.";
    if (!streamOn || injection < 0.01) {
      verdict = "bad";
      message = streamOn
        ? "Fact unavailable — gate almost closed."
        : "Fact unavailable — nothing crossed the boundary.";
    } else if (injection > 0.18) {
      verdict = "loud";
      message =
        (fact || "?") +
        " survives, but the old chunk is too loud (" +
        (injection * 100).toFixed(1) +
        "%).";
    } else {
      verdict = "good";
      message = "Answer: " + (fact || "?") + " (small memory nudge).";
    }
    return { injection, verdict, message, fact: fact || "" };
  }

  /** Exact KV-cache bytes (Full Doc §10 / formulas §12). No extra trailing ×2. */
  function kvCacheBytes({ L, H_KV, d_head, T, B, P_b }) {
    const bytes = 2 * L * H_KV * d_head * T * B * P_b;
    return {
      bytes,
      gib: bytes / (1024 ** 3),
      formula: "2 · L · H_KV · d_head · T · B · P_b",
      L,
      H_KV,
      d_head,
      T,
      B,
      P_b,
    };
  }

  /** MHA / GQA / MQA head layout (formulas §13). */
  function gqaLayout({ mode, H_Q, H_KV }) {
    let hQ = Math.max(1, Math.round(H_Q) || 8);
    let hKV;
    let label;
    if (mode === "MHA") {
      hKV = hQ;
      label = "MHA · H_KV = H_Q";
    } else if (mode === "MQA") {
      hKV = 1;
      label = "MQA · H_KV = 1";
    } else {
      // GQA: keep user H_KV if it divides H_Q; else pick a valid divisor
      hKV = Math.max(1, Math.round(H_KV) || 2);
      if (hKV >= hQ) hKV = Math.max(1, Math.floor(hQ / 2) || 1);
      if (hQ % hKV !== 0) {
        const divisors = [];
        for (let d = 1; d < hQ; d++) if (hQ % d === 0) divisors.push(d);
        hKV = divisors.length ? divisors[Math.min(divisors.length - 1, 1)] : 1;
      }
      label = "GQA · 1 < H_KV < H_Q";
    }
    const reduction = hQ / hKV;
    const groupSize = hQ / hKV;
    return {
      mode: mode === "MHA" || mode === "MQA" ? mode : "GQA",
      H_Q: hQ,
      H_KV: hKV,
      reduction,
      groupSize,
      label,
      formula: "KVReduction = H_Q / H_KV",
    };
  }

  /**
   * Softmax-off linear demo: direct sum vs fixed state S (session: y = S q).
   * Uses first min(T,4) toy keys/values from X (or a fixed 2-step cartoon if empty).
   */
  function linearSoftmaxOff(X) {
    const rows = X && X.length ? X.slice(0, Math.min(4, X.length)) : null;
    let steps;
    if (!rows || rows.length < 2) {
      // Session-style tiny cartoon (scalars as 1-d vectors)
      steps = [
        { k: [1], v: [2], q: [3] },
        { k: [2], v: [4], q: [3] },
      ];
    } else {
      const D = rows[0].length;
      const take = Math.min(2, D);
      steps = rows.map((row) => {
        const k = row.slice(0, take);
        const v = row.slice(0, take).map((x, i) => row[(i + 1) % D]);
        return { k, v, q: rows[rows.length - 1].slice(0, take) };
      });
    }

    const dK = steps[0].k.length;
    const dV = steps[0].v.length;
    // S is dV × dK
    let S = Array.from({ length: dV }, () => Array(dK).fill(0));
    const traj = [];
    steps.forEach((s, t) => {
      for (let a = 0; a < dV; a++) {
        for (let b = 0; b < dK; b++) {
          S[a][b] += s.v[a] * s.k[b];
        }
      }
      // direct: sum_j (q·k_j) v_j  for this prefix
      const direct = Array(dV).fill(0);
      for (let j = 0; j <= t; j++) {
        let dot = 0;
        for (let b = 0; b < dK; b++) dot += s.q[b] * steps[j].k[b];
        for (let a = 0; a < dV; a++) direct[a] += dot * steps[j].v[a];
      }
      // y = S q
      const y = Array(dV).fill(0);
      for (let a = 0; a < dV; a++) {
        for (let b = 0; b < dK; b++) y[a] += S[a][b] * s.q[b];
      }
      const maxAbs = Math.max(...direct.map(Math.abs), 1e-9);
      const err = direct.reduce((acc, v, i) => acc + Math.abs(v - y[i]), 0) / dV;
      traj.push({
        t: t + 1,
        direct: direct.map((v) => +v.toFixed(4)),
        y: y.map((v) => +v.toFixed(4)),
        match: err / maxAbs < 1e-5,
        err: +err.toFixed(6),
        Sflat: S.map((row) => row.map((v) => +v.toFixed(3))),
      });
    });

    const last = traj[traj.length - 1];
    return {
      nSteps: steps.length,
      dK,
      dV,
      traj,
      last,
      note: "Same only with softmax OFF. Softmax’s shared denominator blocks this regroup.",
      formulaDirect: "y = Σ (q·k_j) v_j",
      formulaState: "S = Σ v kᵀ ; y = S q",
    };
  }

  function l2norm(vec) {
    const n = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
    return vec.map((v) => v / n);
  }

  function matVec(S, k) {
    return S.map((row) => row.reduce((s, v, b) => s + v * k[b], 0));
  }

  function cloneMat(S) {
    return S.map((row) => row.slice());
  }

  /**
   * Delta rule vs additive write (formulas §9).
   * normalizeKeys=true → ||k||₂=1 so S k ≈ v after write.
   */
  function deltaRuleWrite(X, normalizeKeys) {
    const rows = X && X.length ? X.slice(0, Math.min(3, X.length)) : null;
    let pairs;
    if (!rows || rows.length < 2) {
      pairs = [
        { k: [1], v: [2] },
        { k: [0], v: [5] },
      ];
    } else {
      const D = rows[0].length;
      const take = Math.min(2, D);
      pairs = rows.map((row) => ({
        k: row.slice(0, take),
        v: row.slice(0, take).map((_, i) => row[(i + 2) % D]),
      }));
    }
    if (normalizeKeys !== false) {
      pairs = pairs.map((p) => ({ k: l2norm(p.k), v: p.v.slice() }));
    }

    const dK = pairs[0].k.length;
    const dV = pairs[0].v.length;
    let Sdelta = Array.from({ length: dV }, () => Array(dK).fill(0));
    let Sadd = Array.from({ length: dV }, () => Array(dK).fill(0));
    const traj = [];

    pairs.forEach((p, t) => {
      const vHat = matVec(Sdelta, p.k);
      const delta = p.v.map((vi, a) => vi - vHat[a]);
      const errBefore = delta.reduce((s, e) => s + Math.abs(e), 0) / dV;

      for (let a = 0; a < dV; a++) {
        for (let b = 0; b < dK; b++) {
          Sdelta[a][b] += delta[a] * p.k[b];
          Sadd[a][b] += p.v[a] * p.k[b];
        }
      }

      const afterDelta = matVec(Sdelta, p.k);
      const afterAdd = matVec(Sadd, p.k);
      const errAfter = afterDelta.reduce((s, e, a) => s + Math.abs(e - p.v[a]), 0) / dV;
      const errAdd = afterAdd.reduce((s, e, a) => s + Math.abs(e - p.v[a]), 0) / dV;
      const knorm = Math.sqrt(p.k.reduce((s, v) => s + v * v, 0));

      traj.push({
        t: t + 1,
        v: p.v.map((x) => +x.toFixed(4)),
        vHat: vHat.map((x) => +x.toFixed(4)),
        delta: delta.map((x) => +x.toFixed(4)),
        afterDelta: afterDelta.map((x) => +x.toFixed(4)),
        afterAdd: afterAdd.map((x) => +x.toFixed(4)),
        errBefore: +errBefore.toFixed(6),
        errAfter: +errAfter.toFixed(6),
        errAdd: +errAdd.toFixed(6),
        overwriteOk: errAfter < 1e-4,
        knorm: +knorm.toFixed(4),
      });
    });

    return {
      nSteps: pairs.length,
      dK,
      dV,
      normalizeKeys: normalizeKeys !== false,
      traj,
      last: traj[traj.length - 1],
      Sdelta: cloneMat(Sdelta).map((r) => r.map((v) => +v.toFixed(3))),
      formula: "v̂=S k ; Δ=v−v̂ ; S←S+Δ kᵀ",
      note: "Exact overwrite needs ||k||₂=1 (or equivalent key normalization).",
    };
  }

  /** Sparse top-k causal attention: softmax kept on survivors only. */
  function sparseTopKAttention(X, k) {
    const dense = causalAttention(X);
    const T = X.length;
    if (!T) {
      return {
        weights: [],
        scores: [],
        k: 0,
        T: 0,
        denseComparisons: 0,
        sparseComparisons: 0,
        scale: 1,
      };
    }
    const topK = Math.max(1, Math.min(Math.round(k) || 1, T));
    const weights = Array.from({ length: T }, () => Array(T).fill(0));
    for (let i = 0; i < T; i++) {
      const cands = [];
      for (let j = 0; j <= i; j++) {
        cands.push({ j, s: dense.scores[i][j] });
      }
      cands.sort((a, b) => b.s - a.s);
      const keep = cands.slice(0, Math.min(topK, cands.length));
      const m = Math.max(...keep.map((c) => c.s));
      const ex = keep.map((c) => Math.exp(c.s - m));
      const z = ex.reduce((s, v) => s + v, 0) || 1;
      keep.forEach((c, idx) => {
        weights[i][c.j] = ex[idx] / z;
      });
    }
    return {
      weights,
      scores: dense.scores,
      k: topK,
      T,
      denseComparisons: T * T,
      sparseComparisons: T * topK,
      scale: dense.scale,
    };
  }

  function rotate2D(x0, x1, m, theta) {
    const a = m * theta;
    const c = Math.cos(a);
    const s = Math.sin(a);
    return [c * x0 - s * x1, s * x0 + c * x1];
  }

  function dot2(a, b) {
    return a[0] * b[0] + a[1] * b[1];
  }

  /**
   * RoPE 2D demo (formulas §10): relative distance enters the rotated score.
   */
  function ropeDemo({ X, i, j, theta }) {
    const T = X && X.length ? X.length : 0;
    let qPair;
    let kPair;
    if (T >= 2 && X[0].length >= 2) {
      const ii = Math.max(0, Math.min(T - 1, i | 0));
      const jj = Math.max(0, Math.min(T - 1, j | 0));
      qPair = [X[ii][0], X[ii][1]];
      kPair = [X[jj][0], X[jj][1]];
      i = ii;
      j = jj;
    } else {
      qPair = [1, 0];
      kPair = [0.8, 0.2];
      i = typeof i === "number" ? i : 0;
      j = typeof j === "number" ? j : 1;
    }
    const th = typeof theta === "number" ? theta : 1;
    const qR = rotate2D(qPair[0], qPair[1], i, th);
    const kR = rotate2D(kPair[0], kPair[1], j, th);
    const unrot = dot2(qPair, kPair);
    const rot = dot2(qR, kR);
    const deltaPos = j - i;

    // Sweep relative distances with fixed content vectors
    const sweep = [];
    for (let d = -3; d <= 3; d++) {
      const jj = i + d;
      const kRd = rotate2D(kPair[0], kPair[1], jj, th);
      const qRi = rotate2D(qPair[0], qPair[1], i, th);
      sweep.push({ delta: d, score: +dot2(qRi, kRd).toFixed(4) });
    }

    return {
      i,
      j,
      theta: th,
      deltaPos,
      qPair: qPair.map((x) => +x.toFixed(4)),
      kPair: kPair.map((x) => +x.toFixed(4)),
      qR: qR.map((x) => +x.toFixed(4)),
      kR: kR.map((x) => +x.toFixed(4)),
      unrotatedDot: +unrot.toFixed(4),
      rotatedDot: +rot.toFixed(4),
      sweep,
      note: "Causal mask blocks the future; RoPE makes near vs far enter the score.",
      formula: "(R_i q)·(R_j k) depends on (j−i)",
    };
  }

  /** DroPE extension factor cartoon (formulas §11). Algorithm under-specified — factor only. */
  function dropeExtension({ trainLen, serveLen }) {
    const train = Math.max(1, Math.round(trainLen) || 8192);
    const serve = Math.max(train, Math.round(serveLen) || 262144);
    const factor = serve / train;
    const reported = Math.abs(factor - 32) < 1e-9;
    return {
      trainLen: train,
      serveLen: serve,
      factor: +factor.toFixed(4),
      factorLabel: factor === Math.floor(factor) ? String(factor) : factor.toFixed(2),
      reportedV4: reported,
      formula: "ExtensionFactor = serve / train",
      note: "V4 story: train 8K → DroPE → report 256K (32×). Exact DroPE algorithm is VERIFY / under-specified.",
      flow: "Train short → DroPE recalibration → serve long",
    };
  }

  /**
   * Sequence compression: T → T/m block summaries (formulas §14).
   * Optional top-k blocks for read budget.
   */
  function sequenceCompression({ T, m, topKBlocks, sentenceT }) {
    const tokT = Math.max(0, Math.round(T) || 0);
    const block = Math.max(1, Math.round(m) || 4);
    const nBlocks = tokT === 0 ? 0 : Math.ceil(tokT / block);
    const kBlocks = Math.max(1, Math.min(nBlocks || 1, Math.round(topKBlocks) || 1));
    const fullSlots = tokT;
    const summarySlots = nBlocks;
    const readSlots = nBlocks === 0 ? 0 : Math.min(kBlocks, nBlocks);
    const reduction = fullSlots === 0 ? 0 : fullSlots / Math.max(1, summarySlots);
    // Toy block map from sentence length for UI chips
    const demoT = Math.max(tokT, Math.round(sentenceT) || 0);
    const demoBlocks = demoT === 0 ? 0 : Math.ceil(demoT / block);
    const blocks = [];
    for (let b = 0; b < demoBlocks; b++) {
      const start = b * block;
      const end = Math.min(demoT, start + block);
      blocks.push({ id: b, start, end, size: end - start });
    }
    return {
      T: tokT,
      m: block,
      nBlocks,
      topKBlocks: kBlocks,
      fullSlots,
      summarySlots,
      readSlots,
      reduction: +reduction.toFixed(2),
      formula: "StoredPositions: T → T/m",
      note: "Indexer top-k is optional; size formula is T/m summaries only.",
      blocks,
    };
  }

  /** V4 depth motif + two roads board (formulas §15 + session fork). */
  function scheduleAndFork({ road }) {
    const motif = ["D", "D", "D", "G", "D", "D", "D", "G"];
    const nD = motif.filter((x) => x === "D").length;
    const nG = motif.filter((x) => x === "G").length;
    const r = road === "native" ? "native" : "stretch";
    const roads = {
      stretch: {
        id: "stretch",
        title: "Road 1 — train short, stretch",
        buy: "Affordable train length; reuse RoPE/DroPE machinery",
        give: "Extension ceiling; needs recalibration evidence",
        when: "Budget-limited; 8K→256K-style targets",
      },
      native: {
        id: "native",
        title: "Road 2 — train at target",
        buy: "Position/cache matched to serve length",
        give: "Much higher train cost (GPU-days)",
        when: "Native long context is worth the train bill",
      },
    };
    return {
      motif,
      motifStr: motif.join(""),
      nD,
      nG,
      formula: "Motif = [D,D,D,G,D,D,D,G]  (6 D + 2 G)",
      note: "Working mixture, not a proven optimum. D=DeltaNet state, G=sparse attn.",
      road: r,
      roads,
      selected: roads[r],
      targetLabel: "Pedagogical target 256K (boards only — no tensor alloc)",
    };
  }

  function runPipeline(text, D) {
    const tokens = tokenize(text);
    const X = embedTokens(tokens, D);
    const attn = causalAttention(X);
    const bills = twoBills(tokens.length);
    return {
      text,
      tokens,
      T: tokens.length,
      D,
      X,
      attn,
      bills,
    };
  }

  global.DemoPipeline = {
    tokenize,
    embedTokens,
    causalAttention,
    twoBills,
    chunkGate,
    kvCacheBytes,
    gqaLayout,
    linearSoftmaxOff,
    deltaRuleWrite,
    sparseTopKAttention,
    ropeDemo,
    dropeExtension,
    sequenceCompression,
    scheduleAndFork,
    runPipeline,
  };
})(typeof window !== "undefined" ? window : globalThis);
