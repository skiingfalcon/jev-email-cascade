# Laya bench `rocm-english` (2026-10-04T00:33:15+00:00)

- server: `http://127.0.0.1:8000/v1/systemone`, model `english`, state `raw`
- device: {'english': 'cuda', 'typed-decisions': 'cuda', 'multilingual': 'cuda'}
- emails: 74 x 2 repeat(s); errors: 0
- memory idle: {'gpu_shared_mb': 113, 'pid': 2280, 'rss_mb': 2543, 'gpu_dedicated_mb': 7459, 'private_mb': 3593}
- memory peak (after phases): {'gpu_shared_mb': 113, 'rss_mb': 2544, 'gpu_dedicated_mb': 22444, 'private_mb': 11777}

| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 1 | 3.7 | 271.2 | 271.2 | 271.2 | 267.9 | 267.9 |
| seq | 148 | 4.0 | 247.6 | 268.2 | 446.4 | 244.5 | 265.3 |
| conc-2 | 74 | 4.0 | 491.6 | 529.5 | 818.3 | 245.4 | 265.4 |
| conc-4 | 74 | 4.0 | 987.9 | 1133.5 | 1301.9 | 245.7 | 264.2 |
| conc-8 | 74 | 4.0 | 1985.9 | 2203.5 | 2214.7 | 245.9 | 264.6 |
| conc-16 | 74 | 4.0 | 4000.6 | 4136.2 | 4142.1 | 247.0 | 264.0 |
| batch-8 | 10 | 3.5 | 251.2 | 428.4 | 533.7 | 250.8 | 427.9 |
| batch-32 | 3 | 2.6 | 266.9 | 521.3 | 544.0 | 266.8 | 521.2 |
| batch-64 | 2 | 2.6 | 320.2 | 393.9 | 400.5 | 319.9 | 393.8 |

Latencies in ms. Batch rows are per email (call latency / batch size).
