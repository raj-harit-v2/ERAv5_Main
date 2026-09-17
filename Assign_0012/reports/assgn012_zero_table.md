# Assgn012 ZeRO Memory Table (30B overlay)

Card capacity = **74.51 GiB**; replicated floor (4 B/weight) = **111.76 GiB**.

| W | Stage | Bytes/param | GiB/GPU | Cluster GiB | Comm | Fit |
| ---: | :--- | ---: | ---: | ---: | :---: | :--- |
| 8 | DP | 16.0000 | 447.03 | 3576.3 | 2P | NEVER_FIT |
| 8 | ZeRO-1 | 5.5000 | 153.67 | 1229.3 | 2P | NEVER_FIT |
| 8 | ZeRO-2 | 3.7500 | 104.77 | 838.2 | 2P | OOM |
| 8 | ZeRO-3 | 2.0000 | 55.88 | 447.0 | 3P | FITS |
| 16 | DP | 16.0000 | 447.03 | 7152.6 | 2P | NEVER_FIT |
| 16 | ZeRO-1 | 4.7500 | 132.71 | 2123.4 | 2P | NEVER_FIT |
| 16 | ZeRO-2 | 2.8750 | 80.33 | 1285.2 | 2P | OOM |
| 16 | ZeRO-3 | 1.0000 | 27.94 | 447.0 | 3P | FITS |
| 32 | DP | 16.0000 | 447.03 | 14305.1 | 2P | NEVER_FIT |
| 32 | ZeRO-1 | 4.3750 | 122.24 | 3911.6 | 2P | NEVER_FIT |
| 32 | ZeRO-2 | 2.4375 | 68.10 | 2179.3 | 2P | FITS |
| 32 | ZeRO-3 | 0.5000 | 13.97 | 447.0 | 3P | FITS |
| 64 | DP | 16.0000 | 447.03 | 28610.2 | 2P | NEVER_FIT |
| 64 | ZeRO-1 | 4.1875 | 117.00 | 7487.8 | 2P | NEVER_FIT |
| 64 | ZeRO-2 | 2.2188 | 61.99 | 3967.4 | 2P | FITS |
| 64 | ZeRO-3 | 0.2500 | 6.98 | 447.0 | 3P | FITS |

## Session 12 ladder match gate

- Mismatches: **0**
- PASS: all ladder cells within 0.2 GiB; W=8 bytes/param match.

## Formulas (Session 12)

```text
DP:     16
ZeRO-1: 4 + 12/W
ZeRO-2: 2 + 14/W
ZeRO-3: 16/W
```
