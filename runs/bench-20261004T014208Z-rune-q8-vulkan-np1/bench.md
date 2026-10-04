# rune bench `rune-q8-vulkan-np1` (2026-10-04T01:34:13+00:00)

- server: `http://127.0.0.1:8001/v1/systemone`, model `Rune-26B-A4B-v3-Q8_0.gguf`, state `raw`
- device: {'model_path': 'C:\\Users\\kghosh\\projects\\local-decision-model\\state\\hf\\hub\\models--owao--surogate-rune-26b-a4b-systemone\\snapshots\\f6cf5c19a05713cb9b2c59a57cd79e85ab8d2ffd\\Rune-26B-A4B-v3-Q8_0.gguf', 'build_info': 'b11382-11fe02151', 'total_slots': 1}
- emails: 74 x 1 repeat(s); errors: 0
- memory idle: {'gpu_shared_mb': 780, 'pid': 15116, 'rss_mb': 1002, 'gpu_dedicated_mb': 26889, 'private_mb': 27937}
- memory peak (after phases): {'gpu_shared_mb': 780, 'rss_mb': 1003, 'gpu_dedicated_mb': 26889, 'private_mb': 27938}

| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 1 | 0.5 | 2132.5 | 2132.5 | 2132.5 | - | - |
| seq | 74 | 0.5 | 2082.4 | 2148.5 | 2263.0 | - | - |
| conc-2 | 74 | 0.5 | 4174.1 | 4312.2 | 4623.6 | - | - |
| conc-4 | 74 | 0.5 | 8378.4 | 8678.7 | 8741.8 | - | - |

Latencies in ms. Batch rows are per email (call latency / batch size).
