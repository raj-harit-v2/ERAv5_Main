"""Import Chrono_verify CSV stamps into chronology.json."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chronology.json"
DEFAULT_CSV = (
    Path(__file__).resolve().parents[3]
    / "Asgn_08_All_docs_xfr"
    / "Chrono_verify_0818a.csv"
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

LABELS: dict[str, dict[str, str]] = {
    "standard_attention": {
        "label": "Vaswani et al. — Attention Is All You Need",
        "citation": "arXiv:1706.03762",
    },
    "sinusoidal_pe": {
        "label": "Vaswani et al. — sinusoidal PE formulation (§3.5)",
        "citation": "arXiv:1706.03762v3",
    },
    "absolute_learned_pe": {
        "label": "Vaswani et al. — learned PE variant (Transformer §3.5)",
        "citation": "arXiv:1706.03762v3",
    },
    "rope": {
        "label": "Su et al. — RoFormer / Rotary Position Embedding",
        "citation": "arXiv:2104.09864",
    },
    "alibi": {
        "label": "Press, Smith, Lewis — Train Short, Test Long (ALiBi)",
        "citation": "arXiv:2108.12409",
    },
    "mqa": {
        "label": "Shazeer — Fast Transformer Decoding (MQA)",
        "citation": "arXiv:1911.02150",
    },
    "gqa": {
        "label": "Ainslie et al. — GQA: Training Generalized Multi-Query Attention",
        "citation": "arXiv:2305.13245",
    },
    "mla": {
        "label": "DeepSeek-V2 — Multi-head Latent Attention (MLA)",
        "citation": "arXiv:2405.04434",
    },
    "sliding_window": {
        "label": "Beltagy et al. — Longformer (sliding-window attention)",
        "citation": "arXiv:2004.05150",
    },
    "attention_sinks": {
        "label": "Xiao et al. — Efficient Streaming Language Models with Attention Sinks",
        "citation": "arXiv:2309.17453",
    },
    "ntk_scaling": {
        "label": "bloc97 — NTK-Aware Scaled RoPE",
        "citation": "Reddit 14lz7j5 (29 Jun 2023)",
    },
    "yarn": {
        "label": "Peng et al. — YaRN: Efficient Context Window Extension",
        "citation": "arXiv:2309.00071",
    },
    "linear_attention": {
        "label": "Katharopoulos et al. — Transformers are RNNs (linear attention)",
        "citation": "arXiv:2006.16236",
    },
    "delta_rule": {
        "label": "Yang et al. — Parallelizing Linear Transformers with the Delta Rule",
        "citation": "arXiv:2406.06484",
    },
    "gated_deltanet": {
        "label": "Yang et al. — Gated Delta Networks",
        "citation": "arXiv:2412.06464",
    },
    "sparse_topk": {
        "label": "Roy et al. — Routing Transformer (content-based sparse / k-means routing)",
        "citation": "arXiv:2003.05997",
    },
    "deepseek_compressed_sparse": {
        "label": "DeepSeek — Native Sparse Attention (compression + sparse)",
        "citation": "arXiv:2502.11089",
    },
    "drope": {
        "label": "YaRN paper — V4 DroPE extension-factor story (not Sakana drop-PE)",
        "citation": "arXiv:2309.00071 (course DroPE / extension factor)",
    },
}


def clean_id(raw: str) -> str:
    return raw.strip().strip("`").strip()


def clean_url(raw: str) -> str:
    s = raw.strip()
    m = re.search(r"https?://[^\s,\]]+", s)
    return m.group(0).rstrip("/") if m else s


def parse_date(raw: str) -> tuple[str, int]:
    s = raw.strip()
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"^(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{2,4})$", s)
        if not m:
            raise ValueError(f"Unrecognized date: {raw!r}")
        day = int(m.group(1))
        month = MONTHS[m.group(2).lower()]
        year = int(m.group(3))
        if year < 100:
            year += 2000
    mon_abbr = list(MONTHS.keys())[list(MONTHS.values()).index(month)].title()
    display = f"{day:02d}-{mon_abbr}-{year}"
    sort_key = year * 10000 + month * 100 + day
    return display, sort_key


def row_field(row: dict, *names: str) -> str:
    norm = {k.strip(): v for k, v in row.items()}
    for name in names:
        if name in norm:
            return norm[name] or ""
    return ""


def load_csv_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open(encoding=enc, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cid = clean_id(row_field(row, "Card"))
                    if not cid:
                        continue
                    date_display, year_sort = parse_date(row_field(row, "Date_Verified"))
                    url = clean_url(row_field(row, "URL"))
                    pdf_v1 = clean_url(row_field(row, "PDF_v1"))
                    pdf_verified = row_field(row, "PDF_Verified").strip()
                    if cid == "ntk_scaling":
                        pdf_verified = "Reddit u/bloc97 r/LocalLLaMA 14lz7j5 · 29 Jun 2023"
                    meta = LABELS.get(cid, {"label": cid, "citation": cid})
                    rows[cid] = {
                        "year_display": date_display,
                        "year_sort": year_sort,
                        "url": url,
                        "pdf_verified": pdf_verified,
                        "pdf_v1": pdf_v1,
                        "label": meta["label"],
                        "citation": meta["citation"],
                    }
            return rows
        except UnicodeDecodeError as exc:
            last_err = exc
            rows.clear()
            continue
    raise last_err or UnicodeDecodeError("csv", b"", 0, 1, "could not decode CSV")


def main() -> None:
    csv_path = DEFAULT_CSV
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    stamps = load_csv_rows(csv_path)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    missing = []
    for card in data.get("cards") or []:
        cid = card.get("id")
        stamp = stamps.get(cid)
        if not stamp:
            missing.append(cid)
            continue
        card["year_display"] = stamp["year_display"]
        card["year_sort"] = stamp["year_sort"]
        src = card.setdefault("source", {})
        src["label"] = stamp["label"]
        src["url"] = stamp["url"]
        src["citation"] = stamp["citation"]
        src["date_verified"] = True
        src["pdf_verified"] = stamp["pdf_verified"]
        src["pdf_v1"] = stamp["pdf_v1"]
        src["date_note"] = stamp["pdf_verified"]
    if missing:
        raise SystemExit(f"No CSV row for: {missing}")
    extra = set(stamps) - {c.get("id") for c in data["cards"]}
    if extra:
        raise SystemExit(f"Unused CSV rows: {extra}")
    DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Imported {len(stamps)} cards from {csv_path.name} into {DATA}")


if __name__ == "__main__":
    main()
