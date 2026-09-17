# Demo-scale shard plan (W=32, tiny model)

N_demo = 49920

| Stage | bytes/param | local model | local grad | local opt | est MiB | overlay GiB@30B |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP | 16.0 | 49920 | 49920 | 49920 | 0.7617 | 447.0348 |
| ZeRO-1 | 4.375 | 49920 | 49920 | 8192 | 0.2842 | 122.2361 |
| ZeRO-2 | 2.4375 | 49920 | 8192 | 8192 | 0.2046 | 68.103 |
| ZeRO-3 | 0.5 | 8192 | 8192 | 8192 | 0.125 | 13.9698 |
