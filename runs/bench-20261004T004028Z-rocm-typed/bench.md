# Laya bench `rocm-typed` (2026-10-04T00:36:45+00:00)

- server: `http://127.0.0.1:8000/v1/systemone`, model `typed-decisions`, state `raw`
- device: {'typed-decisions': 'cuda', 'multilingual': 'cuda', 'english': 'cuda'}
- emails: 74 x 1 repeat(s); errors: 0
- memory idle: {'gpu_shared_mb': 113, 'pid': 2280, 'rss_mb': 2522, 'gpu_dedicated_mb': 22444, 'private_mb': 11777}
- memory peak (after phases): {'gpu_shared_mb': 113, 'rss_mb': 2523, 'gpu_dedicated_mb': 59385, 'private_mb': 48716}

| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 1 | 3.3 | 301.8 | 301.8 | 301.8 | 298.1 | 298.1 |
| seq | 74 | 3.4 | 279.2 | 300.9 | 730.3 | 276.0 | 297.7 |
| conc-4 | 74 | 3.4 | 1113.1 | 1705.2 | 2677.3 | 277.9 | 302.2 |
| batch-8 | 10 | 2.3 | 286.2 | 1077.1 | 1589.3 | 285.7 | 1076.7 |
| batch-32 | 3 | 1.1 | 288.7 | 1575.6 | 1690.0 | 288.6 | 1575.5 |
| batch-64 | 2 | 1.1 | 630.0 | 957.2 | 986.3 | 629.7 | 957.1 |

Latencies in ms. Batch rows are per email (call latency / batch size).
