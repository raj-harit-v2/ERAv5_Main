# Chronology compare — launch date vs teaching order

- **Sr_No** = launch-date rank (`year_sort`).
- **Te_No** = teaching rank (`teaching_order`).

See [`TEACHING_ORDER.md`](TEACHING_ORDER.md).

## Chronology vs teaching order

| Sr_No | Te_No | id | year_display | source.label | source.url |
| ---: | ---: | :--- | :--- | :--- | :--- |
| 1 | 1 | standard_attention | 12-Jun-2017 | Vaswani et al. — Attention Is All You Need | [url](https://arxiv.org/abs/1706.03762) |
| 2 | 7 | absolute_learned_pe | 20-Jun-2017 | Vaswani et al. — learned PE variant (Transformer §3.5) | [url](https://arxiv.org/abs/1706.03762v3) |
| 3 | 9 | sinusoidal_pe | 20-Jun-2017 | Vaswani et al. — sinusoidal PE formulation (§3.5) | [url](https://arxiv.org/abs/1706.03762v3) |
| 4 | 12 | mqa | 07-Nov-2019 | Shazeer — Fast Transformer Decoding (MQA) | [url](https://arxiv.org/abs/1911.02150) |
| 5 | 5 | sparse_topk | 12-Mar-2020 | Roy et al. — Routing Transformer (content-based sparse / k-means routing) | [url](https://arxiv.org/abs/2003.05997v1) |
| 6 | 6 | sliding_window | 10-Apr-2020 | Beltagy et al. — Longformer (sliding-window attention) | [url](https://arxiv.org/abs/2004.05150) |
| 7 | 2 | linear_attention | 29-Jun-2020 | Katharopoulos et al. — Transformers are RNNs (linear attention) | [url](https://arxiv.org/abs/2006.16236) |
| 8 | 8 | rope | 20-Apr-2021 | Su et al. — RoFormer / Rotary Position Embedding | [url](https://arxiv.org/abs/2104.09864) |
| 9 | 10 | alibi | 27-Aug-2021 | Press, Smith, Lewis — Train Short, Test Long (ALiBi) | [url](https://arxiv.org/abs/2108.12409) |
| 10 |  | flash_attention <span style="background:#fde047;color:#0c1018;font-weight:700">    **missing** </span> | 27-May-2022 | Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [url](https://arxiv.org/abs/2205.14135) |
| 11 | 13 | gqa | 22-May-2023 | Ainslie et al. — GQA: Training Generalized Multi-Query Attention | [url](https://arxiv.org/abs/2305.13245) |
| 12 |  | position_interpolation <span style="background:#fde047;color:#0c1018;font-weight:700">    **missing** </span> | 27-Jun-2023 | Chen et al. — Extending Context Window of Large Language Models via Positional Interpolation | [url](https://arxiv.org/abs/2306.15595) |
| 13 | 16 | ntk_scaling | 29-Jun-2023 | bloc97 — NTK-Aware Scaled RoPE | [url](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have) |
| 14 | 17 | yarn | 31-Aug-2023 | Peng et al. — YaRN: Efficient Context Window Extension | [url](https://arxiv.org/abs/2309.00071) |
| 15 | 11 | drope | 31-Aug-2023 | YaRN paper — V4 DroPE extension-factor story (not Sakana drop-PE) | [url](https://arxiv.org/abs/2309.00071) |
| 16 | 18 | attention_sinks | 29-Sep-2023 | Xiao et al. — Efficient Streaming Language Models with Attention Sinks | [url](https://arxiv.org/abs/2309.17453) |
| 17 | 14 | mla | 07-May-2024 | DeepSeek-V2 — Multi-head Latent Attention (MLA) | [url](https://arxiv.org/abs/2405.04434) |
| 18 | 3 | delta_rule | 10-Jun-2024 | Yang et al. — Parallelizing Linear Transformers with the Delta Rule | [url](https://arxiv.org/abs/2406.06484) |
| 19 | 4 | gated_deltanet | 09-Dec-2024 | Yang et al. — Gated Delta Networks | [url](https://arxiv.org/abs/2412.06464) |
| 20 | 15 | deepseek_compressed_sparse | 16-Feb-2025 | DeepSeek — Native Sparse Attention (compression + sparse) | [url](https://arxiv.org/abs/2502.11089) |
