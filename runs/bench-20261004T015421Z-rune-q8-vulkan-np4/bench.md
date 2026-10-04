# rune bench `rune-q8-vulkan-np4` (2026-10-04T01:42:38+00:00)

- server: `http://127.0.0.1:8001/v1/systemone`, model `Rune-26B-A4B-v3-Q8_0.gguf`, state `raw`
- device: {'model_path': 'C:\\Users\\kghosh\\projects\\local-decision-model\\state\\hf\\hub\\models--owao--surogate-rune-26b-a4b-systemone\\snapshots\\f6cf5c19a05713cb9b2c59a57cd79e85ab8d2ffd\\Rune-26B-A4B-v3-Q8_0.gguf', 'build_info': 'b11382-11fe02151', 'total_slots': 4}
- emails: 74 x 1 repeat(s); errors: 0
- memory idle: {'gpu_shared_mb': 782, 'pid': 7532, 'rss_mb': 969, 'gpu_dedicated_mb': 28673, 'private_mb': 29678}
- memory peak (after phases): {'gpu_shared_mb': 783, 'rss_mb': 1012, 'gpu_dedicated_mb': 28763, 'private_mb': 29819}

| phase | requests | emails/s | client p50 | p95 | p99 | server p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 1 | 0.3 | 2990.8 | 2990.8 | 2990.8 | - | - |
| seq | 74 | 0.4 | 2317.0 | 2366.4 | 2654.7 | - | - |
| conc-2 | 74 | 0.4 | 3677.2 | 6012.5 | 7208.4 | - | - |
| conc-4 | 74 | 0.4 | 4664.6 | 12474.2 | 144199.7 | - | - |
| conc-8 | 74 | 0.4 | 15194.2 | 37375.6 | 42286.7 | - | - |

Latencies in ms. Batch rows are per email (call latency / batch size).
