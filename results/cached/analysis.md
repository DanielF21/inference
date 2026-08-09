# cached: preallocated KV cache

## Results

| prompt | ptok | ttft | total | decode tok/s | itl p95 | per-step | peak mem |
|---|---|---|---|---|---|---|---|
| p16 | 16 | 0.034s | 5.821s | 44.05 | 24.0 ms | 22.7 ms | 2.89 GiB |
| p64 | 64 | 0.027s | 5.606s | 45.70 | 22.3 ms | 21.9 ms | 2.90 GiB |
| p256 | 256 | 0.025s | 5.543s | 46.21 | 23.0 ms | 21.7 ms | 2.91 GiB |
| p1024 | 1024 | 0.055s | 5.639s | 45.67 | 22.4 ms | 22.0 ms | 2.98 GiB |
| p4096 | 4096 | 0.248s | 5.815s | 45.80 | 22.3 ms | 22.7 ms | 3.25 GiB |

`total` is the mean wall clock per run. `per-step` is `total / 256`.
