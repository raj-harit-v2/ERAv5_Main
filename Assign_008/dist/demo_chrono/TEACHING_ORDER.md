# Teaching order vs chapter vs launch date

Authority for **Te_No** / **Sec_No** / **Sr_No** in this folder.  
Sources: `Session 8 Modern Attention Variants Web.htm`, Transcript FULL, Assignment §18.

## Why the old Te_No was wrong

`provisional_order` copied Assignment §18 **“What to cover, at minimum”** — a **coverage checklist**, not class order.

That checklist puts RoPE 4th (after standard / abs PE / sin PE), so export marked **Te_No=4**.  
In the Full Document, RoPE’s **major** beat is **§8** (`s8_widget_7_rope`), after linear / delta / sparse (§4–7).

Coverage list ≠ teaching. Do not use `assignment_cover_order` as Te_No.

## Definitions

| Rank | Formula |
| :--- | :--- |
| **Te_No** | 1-based index in `data/chronology.json` → `teaching_order` |
| **Sec_No** | Full Document `<h1>` chapter (`card.sec_no`) |
| **Sr_No** | Rank by `year_sort` ascending (launch date) |

Te_No need not equal Sec_No for every id (shared chapters; some have no own h1).

## RoPE proof

| Rank | Value | Why |
| :--- | ---: | :--- |
| Te_No | **8** | Eight teaching beats: attn → linear → delta → gated → sparse → window → abs PE → **RoPE** |
| Sec_No | **8** | Web.htm `<h1 id="8-position-and-the-part-that-is-usually-skipped">` + widget `s8_widget_7_rope` |
| Sr_No | **8** | Launch 20-Apr-2021 is 8th among dated wall + Q2 extras by `year_sort` |

## Locked `teaching_order`

| Te_No | id | Sec_No | Widget (if any) | Cue |
| ---: | :--- | ---: | :--- | :--- |
| 1 | standard_attention | 2 | s8_widget_0b / 0 | §2 Attention |
| 2 | linear_attention | 4 | 3, 4 | §4–5 softmax off / state |
| 3 | delta_rule | 6 | 5 | §6 delta |
| 4 | gated_deltanet | 6 | — | After delta; §13 schedule also |
| 5 | sparse_topk | 7 | 6 | §7 sparse |
| 6 | sliding_window | 7 | — | §7 window family |
| 7 | absolute_learned_pe | 8 | — | §8 pre-RoPE PE |
| **8** | **rope** | **8** | **s8_widget_7_rope** | **§8 major** |
| 9 | sinusoidal_pe | 8 | — | §8 PE cluster (after RoPE in Te_No so rope stays 8) |
| 10 | alibi | 8 | — | Position/bias; no own h1 |
| 11 | drope | 9 | 8 | §9 DroPE |
| 12 | mqa | 11 | 10 | §11 |
| 13 | gqa | 11 | 10 | §11 |
| 14 | mla | 12 | 11 | §12 compression |
| 15 | deepseek_compressed_sparse | 12 | — | §12+ / late paper |
| 16 | ntk_scaling | 15 | — | §15 mentions |
| 17 | yarn | 15 | — | §15 |
| 18 | attention_sinks | 15 | — | §15 |

## Assignment cover list

`assignment_cover_order` (= legacy `provisional_order`) remains the §18 checklist for “did we ship a card?” — **not** Te_No.

## Graded wall

`demo_chrono/index.html` timeline is sorted by **Sr_No / launch date** (Assignment: chronological story).  
Teaching ranks live in [`CHRONOLOGY_Compare.md`](CHRONOLOGY_Compare.md).
