# Assignment 06 Evidence Bundle

Run ID: `run_29f41fa697`
Overall: **PASS**

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | manifests/ |
| Evaluation firewall | PASS | ledgers/firewall.json |
| Packing correctness | PASS | packed_batch_reports/|consumption_ledger.jsonl |
| Mixture compliance | PASS | ledgers/consumption_ledger.jsonl |
| OPUS audit trail | PASS | ledgers/opus_ledger.jsonl |
| Crash recovery | PASS | checkpoints/ |
| Replay | PASS | ledgers/consumption_ledger.jsonl |
| Learning trace | PASS | ledgers/learning_ledger.jsonl |
| Throughput | PASS | performance.json |

## Details
- **Tokenizer integrity**: n_manifests=28; hashes_ok
- **Evaluation firewall**: blocked_events=1
- **Packing correctness**: max_utilization=1.000
- **Mixture compliance**: actual_share={agentic:0.025, code:0.375, indic:0.084, long_context:0.088, reasoning:0.025, stem:0.096, web:0.307}; max_abs_delta=0.12257344968611883
- **OPUS audit trail**: statuses=['accepted', 'deferred', 'floor_override', 'rejected']
- **Crash recovery**: expected=run_29f41fa697:12:0 actual=run_29f41fa697:12:0
- **Replay**: compared=5 original=409cbabc529b63113bfbc9250e5d703207a5b3dd7fb74871310935259c2a543a replay=409cbabc529b63113bfbc9250e5d703207a5b3dd7fb74871310935259c2a543a
- **Learning trace**: n_learning_events=26
- **Throughput**: useful_tokens_per_second=4043.10
