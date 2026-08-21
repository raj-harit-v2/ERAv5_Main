# Assignment Q2 — What the timeline shows

Once the mechanisms are sorted by date, the field’s priorities read as a sequence of bills, not a taxonomy.

- **2017:** exact global attention and position encoding arrive together (Transformer + learned/sinusoidal PE).
- **2019–2023:** the bill shifts to **KV memory at decode** (MQA → GQA → MLA) while **position** forks (RoPE, ALiBi) and **compute** forks (Longformer window, linear attention, sparse routing).
- **2023:** a cluster of **RoPE stretch** fixes (NTK, YaRN, attention sinks, DroPE/V4 extension story) appears after long-context pain is already visible — you would not see that cluster as one “family” in a list.
- **2024–2025:** **stateful recurrence** returns (delta rule, Gated DeltaNet) alongside **aggressive sparse compression** (DeepSeek NSA).

A list tells you what exists; the timeline shows **what problem was urgent when** and why the next paper existed.

**Extra mechanism:** none added (minimum 18 only).

Sources for dates: [`demo_chrono/CHRONOLOGY_SOURCES_BY_DATE.md`](../demo_chrono/CHRONOLOGY_SOURCES_BY_DATE.md) (18 wall cards). The two extras below were considered for Q2 bonus only and are documented here (not on the wall).

## Extra-mechanism options considered (not added)

These were the earlier Question 2 bonus choices. The wall stays at the minimum 18 cards.

### A — Position Interpolation (Chen et al., 27 Jun 2023)

Extends a RoPE model’s context by interpolating position indices rather than extrapolating them. Sits in the 2023 RoPE-stretch cluster (next to NTK / YaRN) but was not on the class minimum list.

- Abs: [https://arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)
- PDF: [https://arxiv.org/pdf/2306.15595](https://arxiv.org/pdf/2306.15595)
- Stamp: `arXiv:2306.15595` (Chen, Han, Yu, Bick, Bai, Wang — *Extending Context Window of Large Language Models via Positional Interpolation*)

### C — FlashAttention (Dao et al., 27 May 2022)

IO-aware exact attention: tiles Q/K/V through SRAM so the full N×N score matrix is never materialized in HBM. A compute/memory mechanism, not a new attention *formula* — which is why it was optional rather than one of the 18 cards.

- Abs: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- PDF: [https://arxiv.org/pdf/2205.14135](https://arxiv.org/pdf/2205.14135)
- Stamp: `arXiv:2205.14135` (Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*)
