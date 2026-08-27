/**
 * Session 9 browser engine — loads exported NanoLM weights and runs forward + CE.
 * Mirrors src/llm/nano_lm.py + src/llm/chunked_ce.py (CPU, float64 for parity).
 */
(function (global) {
  "use strict";

  var BUNDLE = null;

  function tokenizeText(text) {
    return String(text).match(/[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]/g) || [];
  }

  function encodeText(text, bundle) {
    var ids = [bundle.bos_id];
    var toks = tokenizeText(text);
    for (var i = 0; i < toks.length; i++) {
      var id = bundle.token_to_id[toks[i]];
      ids.push(id !== undefined ? id : bundle.unk_id);
    }
    ids.push(bundle.eos_id);
    return { ids: ids, toks: toks };
  }

  function decodeId(id, bundle) {
    return bundle.id_to_token[id] || bundle.special.UNK;
  }

  function matVec(x, W) {
    var out = new Float64Array(W.length);
    for (var j = 0; j < W.length; j++) {
      var row = W[j], s = 0;
      for (var i = 0; i < x.length; i++) s += x[i] * row[i];
      out[j] = s;
    }
    return out;
  }

  function layerNorm(x, gamma, beta) {
    var n = x.length, m = 0, i;
    for (i = 0; i < n; i++) m += x[i];
    m /= n;
    var v = 0;
    for (i = 0; i < n; i++) { var d = x[i] - m; v += d * d; }
    v = Math.sqrt(v / n + 1e-5);
    var o = new Float64Array(n);
    for (i = 0; i < n; i++) o[i] = ((x[i] - m) / v) * gamma[i] + beta[i];
    return o;
  }

  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  }

  function softmaxRow(a) {
    var m = -Infinity, i;
    for (i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
    var e = new Float64Array(a.length), s = 0;
    for (i = 0; i < a.length; i++) {
      e[i] = a[i] === -Infinity ? 0 : Math.exp(a[i] - m);
      s += e[i];
    }
    for (i = 0; i < a.length; i++) e[i] /= s || 1;
    return e;
  }

  function causalSelfAttn(xSeq, blk, hp) {
    var T = xSeq.length, D = hp.d_model, NH = hp.n_heads, DH = D / NH;
    var qkvW = blk.attn_qkv, projW = blk.attn_proj;
    var Q = [], K = [], V = [];
    for (var t = 0; t < T; t++) {
      var qkv = matVec(xSeq[t], qkvW);
      Q.push(qkv.subarray(0, D));
      K.push(qkv.subarray(D, 2 * D));
      V.push(qkv.subarray(2 * D, 3 * D));
    }
    var outSeq = [];
    for (t = 0; t < T; t++) {
      var acc = new Float64Array(D);
      for (var h = 0; h < NH; h++) {
        var off = h * DH;
        var scores = new Float64Array(t + 1);
        for (var j = 0; j <= t; j++) {
          var s = 0;
          for (var d = 0; d < DH; d++) s += Q[t][off + d] * K[j][off + d];
          scores[j] = s / Math.sqrt(DH);
        }
        var w = softmaxRow(scores);
        var headOut = new Float64Array(DH);
        for (j = 0; j <= t; j++) {
          for (d = 0; d < DH; d++) headOut[d] += w[j] * V[j][off + d];
        }
        for (d = 0; d < DH; d++) acc[off + d] = headOut[d];
      }
      outSeq.push(matVec(acc, projW));
    }
    return outSeq;
  }

  function feedForward(x, blk) {
    var h = matVec(x, blk.ffn_w1);
    for (var i = 0; i < h.length; i++) h[i] = gelu(h[i]);
    return matVec(h, blk.ffn_w2);
  }

  function addVec(a, b) {
    var o = new Float64Array(a.length);
    for (var i = 0; i < a.length; i++) o[i] = a[i] + b[i];
    return o;
  }

  function lookupEmb(ids, table) {
    return ids.map(function (id) {
      return new Float64Array(table[id]);
    });
  }

  function prepareBlocks(state, nLayers) {
    var blocks = [];
    for (var b = 0; b < nLayers; b++) {
      var p = "blocks." + b + ".";
      blocks.push({
        ln1_g: new Float64Array(state[p + "ln1.weight"]),
        ln1_b: new Float64Array(state[p + "ln1.bias"]),
        ln2_g: new Float64Array(state[p + "ln2.weight"]),
        ln2_b: new Float64Array(state[p + "ln2.bias"]),
        attn_qkv: state[p + "attn.qkv.weight"].map(function (r) { return new Float64Array(r); }),
        attn_proj: state[p + "attn.proj.weight"].map(function (r) { return new Float64Array(r); }),
        ffn_w1: state[p + "ffn.net.0.weight"].map(function (r) { return new Float64Array(r); }),
        ffn_w2: state[p + "ffn.net.2.weight"].map(function (r) { return new Float64Array(r); }),
      });
    }
    return blocks;
  }

  function forwardIds(ids, bundle) {
    var hp = bundle.hyperparams;
    var state = bundle.state_dict;
    var tokEmb = state["tok_emb.weight"];
    var posEmb = state["pos_emb.weight"];
    var blocks = prepareBlocks(state, hp.n_layers);
    var lnF_g = new Float64Array(state["ln_f.weight"]);
    var lnF_b = new Float64Array(state["ln_f.bias"]);
    var headW = state["lm_head.proj.weight"].map(function (r) { return new Float64Array(r); });

    var x = lookupEmb(ids, tokEmb);
    for (var t = 0; t < ids.length; t++) {
      var pe = new Float64Array(posEmb[t]);
      x[t] = addVec(x[t], pe);
    }

    for (var bi = 0; bi < blocks.length; bi++) {
      var blk = blocks[bi];
      var xin = x.map(function (r) { return new Float64Array(r); });
      var h1 = xin.map(function (row) { return layerNorm(row, blk.ln1_g, blk.ln1_b); });
      var attnOut = causalSelfAttn(h1, blk, hp);
      x = x.map(function (row, i) { return addVec(row, attnOut[i]); });
      xin = x.map(function (r) { return new Float64Array(r); });
      h1 = xin.map(function (row) { return layerNorm(row, blk.ln2_g, blk.ln2_b); });
      var ffnOut = h1.map(function (row) { return feedForward(row, blk); });
      x = x.map(function (row, i) { return addVec(row, ffnOut[i]); });
    }

    var hidden = x.map(function (row) { return layerNorm(row, lnF_g, lnF_b); });
    var logits = hidden.map(function (row) { return matVec(row, headW); });
    return { hidden: hidden, logits: logits, vocab_size: bundle.vocab_size };
  }

  function shiftLogitsTargets(logits, ids) {
    return {
      shift_logits: logits.slice(0, -1),
      shift_targets: ids.slice(1),
    };
  }

  function crossEntropyRow(logits, target) {
    var V = logits.length, m = -Infinity, i;
    for (i = 0; i < V; i++) if (logits[i] > m) m = logits[i];
    var logZ = 0;
    for (i = 0; i < V; i++) logZ += Math.exp(logits[i] - m);
    logZ = m + Math.log(logZ);
    return logZ - logits[target];
  }

  function maskedMeanCE(shift_logits, shift_targets, mask) {
    var sum = 0, count = 0;
    for (var i = 0; i < shift_targets.length; i++) {
      if (!mask[i]) continue;
      sum += crossEntropyRow(shift_logits[i], shift_targets[i]);
      count++;
    }
    return { loss: count ? sum / count : 0, count: count };
  }

  function chunkedCE(shift_logits, shift_targets, mask, chunkSize) {
    var pairs = [];
    for (var i = 0; i < shift_targets.length; i++) {
      if (mask[i]) pairs.push({ lg: shift_logits[i], tg: shift_targets[i] });
    }
    var total = 0, count = 0;
    for (var s = 0; s < pairs.length; s += chunkSize) {
      var end = Math.min(s + chunkSize, pairs.length);
      for (var j = s; j < end; j++) {
        total += crossEntropyRow(pairs[j].lg, pairs[j].tg);
        count++;
      }
    }
    return count ? total / count : 0;
  }

  function buildPadMask(ids, padId) {
    var mask = [];
    for (var i = 1; i < ids.length; i++) mask.push(ids[i] !== padId ? 1 : 0);
    return mask;
  }

  function shiftPairs(ids, bundle, limit) {
    limit = limit || 8;
    var pairs = [];
    for (var i = 0; i < ids.length - 1 && pairs.length < limit; i++) {
      pairs.push([decodeId(ids[i], bundle), decodeId(ids[i + 1], bundle)]);
    }
    return pairs;
  }

  function estimateLogitsBytes(seqLen, vocab, bytesPer) {
    return seqLen * vocab * (bytesPer || 4);
  }

  function runPipeline(text, opts) {
    opts = opts || {};
    if (!BUNDLE) throw new Error("S9Engine: call loadBundle first");
    var enc = encodeText(text, BUNDLE);
    var fwd = forwardIds(enc.ids, BUNDLE);
    var sh = shiftLogitsTargets(fwd.logits, enc.ids);
    var mask = buildPadMask(enc.ids, BUNDLE.pad_id);
    var ce = maskedMeanCE(sh.shift_logits, sh.shift_targets, mask);
    var T = enc.ids.length;
    var V = BUNDLE.vocab_size;
    var D = BUNDLE.hyperparams.d_model;
    var nValid = ce.count;
    var chunkSize = Math.max(1, Math.floor(opts.chunkSize || 1024));
    var ceChunk = chunkedCE(sh.shift_logits, sh.shift_targets, mask, chunkSize);
    var nChunks = nValid ? Math.ceil(nValid / chunkSize) : 0;
    var peakFull = estimateLogitsBytes(Math.max(T - 1, 1), V, 4);
    var peakChunk = estimateLogitsBytes(Math.min(chunkSize, Math.max(nValid, 1)), V, 4);
    return {
      text: text,
      token_strings: enc.ids.map(function (id) { return decodeId(id, BUNDLE); }),
      ids: enc.ids,
      shapes: {
        tokens: [1, T],
        hidden: [1, T, D],
        logits: [1, T, V],
        targets: [1, T - 1],
        mask: [1, T - 1],
      },
      shift_pairs: shiftPairs(enc.ids, BUNDLE, opts.pairLimit || 8),
      loss0: ce.loss,
      ppl0: Math.exp(ce.loss),
      ln_v: Math.log(V),
      vocab_size: V,
      ce_chunked: ceChunk,
      chunk_size: chunkSize,
      n_chunks: nChunks,
      mask_count: nValid,
      peak_full_mib: peakFull / (1024 * 1024),
      peak_chunk_mib: peakChunk / (1024 * 1024),
      tied_params: BUNDLE.tied_params,
      untied_params: BUNDLE.untied_params,
      top_logits_last: topK(sh.shift_logits[sh.shift_logits.length - 1], BUNDLE, 5),
    };
  }

  function topK(logits, bundle, k) {
    var order = [];
    for (var i = 0; i < logits.length; i++) order.push(i);
    order.sort(function (a, b) { return logits[b] - logits[a]; });
    var out = [];
    for (var j = 0; j < Math.min(k, order.length); j++) {
      var id = order[j];
      out.push({ token: decodeId(id, bundle), id: id, logit: logits[id] });
    }
    return out;
  }

  function loadBundle(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("Failed to load " + url);
      return r.json();
    }).then(function (data) {
      BUNDLE = data;
      return data;
    });
  }

  function getBundle() { return BUNDLE; }

  function setBundle(data) { BUNDLE = data; }

  global.S9Engine = {
    loadBundle: loadBundle,
    getBundle: getBundle,
    setBundle: setBundle,
    tokenizeText: tokenizeText,
    encodeText: encodeText,
    forwardIds: forwardIds,
    runPipeline: runPipeline,
    crossEntropyRow: crossEntropyRow,
    chunkedCE: chunkedCE,
    estimateLogitsBytes: estimateLogitsBytes,
    buildPadMask: buildPadMask,
    shiftPairs: shiftPairs,
    decodeId: decodeId,
  };
})(typeof window !== "undefined" ? window : globalThis);
