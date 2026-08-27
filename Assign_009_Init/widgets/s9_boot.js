/** Shared boot helpers: detect file:// and resolve data paths for widgets. */
(function (global) {
  "use strict";

  function isFileProtocol() {
    return global.location && global.location.protocol === "file:";
  }

  /** Widget HTML lives in dist/widgets/ → data is ../data/ */
  function dataUrl(name) {
    if (isFileProtocol()) {
      throw new Error(
        "Cannot load " + name + " via file://. From Assign_009_Init run: " +
        "python -m src.pipeline.serve_dist  then open http://127.0.0.1:8765/"
      );
    }
    var base = global.location.pathname.replace(/\\/g, "/");
    if (base.indexOf("/widgets/") !== -1) {
      return "../data/" + name;
    }
    return "data/" + name;
  }

  function reportsUrl(name) {
    if (isFileProtocol()) {
      throw new Error("Open via python -m src.pipeline.serve_dist (not file://)");
    }
    var base = global.location.pathname.replace(/\\/g, "/");
    if (base.indexOf("/widgets/") !== -1) {
      return "../reports/" + name;
    }
    return "reports/" + name;
  }

  function showFileWarning(containerId) {
    if (!isFileProtocol()) return;
    var el = document.getElementById(containerId || "s9-file-warn");
    if (!el) {
      el = document.createElement("div");
      el.id = "s9-file-warn";
      el.style.cssText =
        "margin:0 0 10px;padding:10px 12px;background:#fef2f2;border:1px solid #fecaca;" +
        "border-radius:8px;color:#991b1b;font-size:12px;line-height:1.45";
      var main = document.querySelector("main") || document.body;
      main.insertBefore(el, main.firstChild);
    }
    el.innerHTML =
      "<b>Widgets need a local web server.</b> Double-clicking HTML blocks JSON load.<br>" +
      "In terminal: <code>cd Assign_009_Init</code> then " +
      "<code>python -m src.pipeline.serve_dist</code> → open " +
      "<a href=\"http://127.0.0.1:8765/\">http://127.0.0.1:8765/</a>";
  }

  global.S9Boot = {
    isFileProtocol: isFileProtocol,
    dataUrl: dataUrl,
    reportsUrl: reportsUrl,
    showFileWarning: showFileWarning,
  };
})(typeof window !== "undefined" ? window : globalThis);
