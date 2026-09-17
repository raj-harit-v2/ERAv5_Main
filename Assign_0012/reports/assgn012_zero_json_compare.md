# Assgn012 DeepSpeed JSON vs Lab Accounting

JSON configs select `zero_optimization.stage` only. GiB is **computed** by Session 12 formulas
and compared to `assgn012_zero_table.csv` at W=32. This is not measured DeepSpeed VRAM.

- Catalog: `configs/deepspeed/zero_stages_catalog.json`
- Lab CSV: `reports/assgn012_zero_table.csv`
- Overlay: N=3e+10, W=32
- Overall: **PASS** (ok=True)

| Stage | JSON | Implied GiB | Lab GiB | Δ | Comm | Fit | Verdict |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- | :--- |
| ZeRO-1 | `ds_zero1.json` | 122.24 | 122.24 | 0.000 | 2P | NEVER_FIT | **PASS** |
| ZeRO-2 | `ds_zero2.json` | 68.10 | 68.10 | 0.000 | 2P | FITS | **PASS** |
| ZeRO-3 | `ds_zero3.json` | 13.97 | 13.97 | 0.000 | 3P | FITS | **PASS** |

## Details

- **ZeRO-1** (`stage=1`): bpp=4.3750; lab_fit=NEVER_FIT; lab_comm=2P; stage maps to lab W=32 row within tolerance
- **ZeRO-2** (`stage=2`): bpp=2.4375; lab_fit=FITS; lab_comm=2P; stage maps to lab W=32 row within tolerance
- **ZeRO-3** (`stage=3`): bpp=0.5000; lab_fit=FITS; lab_comm=3P; stage maps to lab W=32 row within tolerance

## Formulas

```text
ZeRO-1: 4 + 12/W
ZeRO-2: 2 + 14/W
ZeRO-3: 16/W
```

## Note on root GPU_example.json

Root `GPU_example.json` is a ZeRO-3 + CPU offload sketch (HF-style `auto` fields).
It is **not** the compare baseline; use `configs/deepspeed/ds_zero{1,2,3}.json`.
