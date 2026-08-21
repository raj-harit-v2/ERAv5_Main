(function () {
  "use strict";

  const VERIFY = "VERIFY_PRIMARY_SOURCE";

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function sortCards(data) {
    // Tie-break only: Assignment §18 cover list — NOT teaching_order.
    const cover =
      data.assignment_cover_order || data.provisional_order || [];
    const rank = Object.fromEntries(cover.map((id, i) => [id, i]));
    return (data.cards || []).slice().sort((a, b) => {
      const ya = a.year_sort != null ? a.year_sort : 99999999;
      const yb = b.year_sort != null ? b.year_sort : 99999999;
      if (ya !== yb) return ya - yb;
      return (rank[a.id] != null ? rank[a.id] : 9999) - (rank[b.id] != null ? rank[b.id] : 9999);
    });
  }

  function renderCard(card) {
    const isVerify = card.year_display === VERIFY;
    const yearClass = isVerify ? "verify" : "verified";
    const src = card.source || {};
    const srcLabel = escapeHtml(src.label || "");
    const srcUrl = (src.url || "").trim();
    const srcLink = srcUrl
      ? '<a href="' + escapeHtml(srcUrl) + '" target="_blank" rel="noopener">' + srcLabel + "</a>"
      : srcLabel;
    const stampText = (src.pdf_verified || src.date_note || "").trim();
    const pdfHref = (src.pdf_v1 || src.url || "").trim();
    let stampPart = "";
    if (stampText) {
      stampPart = " <strong>Stamp:</strong> " + escapeHtml(stampText);
    }
    if (pdfHref) {
      stampPart +=
        ' <a href="' +
        escapeHtml(pdfHref) +
        '" target="_blank" rel="noopener">pdf_v</a>';
    }

    let widget = "";
    if (card.widget_iframe) {
      widget =
        '<div class="widget-frame"><iframe title="' +
        escapeHtml(card.title) +
        '" src="' +
        escapeHtml(card.widget_iframe) +
        '" loading="lazy"></iframe></div>';
    }

    return (
      '<article class="chrono-card' +
      (isVerify ? " verify-year" : "") +
      '" data-id="' +
      escapeHtml(card.id) +
      '">' +
      '<div class="year-rail ' +
      yearClass +
      '">' +
      escapeHtml(card.year_display) +
      "</div>" +
      "<h2>" +
      escapeHtml(card.title) +
      "</h2>" +
      '<div class="story-block"><strong>Problem</strong>' +
      escapeHtml(card.problem) +
      "</div>" +
      '<div class="story-block"><strong>Mechanism</strong>' +
      escapeHtml(card.mechanism) +
      "</div>" +
      '<div class="trade-grid">' +
      '<div class="trade-cell buy"><b>Buy</b>' +
      escapeHtml(card.buy) +
      "</div>" +
      '<div class="trade-cell give"><b>Give up</b>' +
      escapeHtml(card.give_up) +
      "</div>" +
      '<div class="trade-cell when"><b>When</b>' +
      escapeHtml(card.when) +
      "</div>" +
      "</div>" +
      '<div class="source-line"><strong>Source:</strong> ' +
      srcLink +
      stampPart +
      "</div>" +
      widget +
      "</article>"
    );
  }

  function loadData() {
    if (window.CHRONOLOGY_DATA && Array.isArray(window.CHRONOLOGY_DATA.cards)) {
      return Promise.resolve(window.CHRONOLOGY_DATA);
    }
    return fetch("data/chronology.json").then((res) => {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    });
  }

  function boot() {
    const root = document.getElementById("timeline");
    if (!root) return;
    loadData()
      .then((data) => {
        const cards = sortCards(data);
        root.innerHTML = cards.map(renderCard).join("");
      })
      .catch((err) => {
        root.innerHTML =
          '<div class="error-box">Could not load chronology.json: ' +
          escapeHtml(err.message) +
          ". Open via Live Server, or keep data/chronology.embed.js next to this page.</div>";
      });
  }

  document.addEventListener("DOMContentLoaded", boot);

  window.ChronoApp = { sortCards: sortCards, VERIFY: VERIFY };
})();
