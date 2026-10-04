# Laya bench `cpu-english` (2026-10-04T00:18:07+00:00)

- server: `http://127.0.0.1:8000/v1/systemone`, model `english`, state `raw`
- device: {'multilingual': 'cpu', 'typed-decisions': 'cpu', 'english': 'cpu'}
- emails: 74 x 1 repeat(s); errors: 0
- memory idle: {'gpu_shared_mb': 0, 'pid': 28332, 'rss_mb': 5082, 'gpu_dedicated_mb': 0, 'private_mb': 7052}
- memory peak (after phases): {'gpu_shared_mb': 0, 'rss_mb': 5312, 'gpu_dedicated_mb': 0, 'private_mb': 7235}

| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 1 | 0.4 | 2557.5 | 2557.5 | 2557.5 | 2534.0 | 2534.0 |
| seq | 74 | 0.4 | 2397.2 | 2548.2 | 2995.5 | 2395.0 | 2546.2 |
| conc-4 | 74 | 0.4 | 9594.3 | 10395.8 | 10766.1 | 2400.5 | 2573.4 |
| batch-8 | 10 | 0.4 | 2084.1 | 3095.5 | 3678.0 | 2083.8 | 3095.2 |
| batch-32 | 3 | 0.3 | 2445.5 | 3694.1 | 3805.1 | 2445.4 | 3694.0 |

Latencies in ms. Batch rows are per email (call latency / batch size).
