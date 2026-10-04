# Decision models for email triage: what we tried and what we found

*Measured September–October 2026 on the same 74 labelled business emails, with the same 8
questions and the same scoring code for every model.*

## The short version

We need software that reads a business email and answers a fixed set of questions about it: what
kind of email it is, how urgent it is, whether the sender expects a reply, and so on. We compared
five models for that job.

**The best fully local option is Rune.** It runs on our own machine, costs nothing per email, and
keeps the email on our hardware. It is as accurate as the paid service we use today (Jev): both
got 71 of 74 emails into the right category. The trade-off is speed. Rune takes about 2 seconds
per email, where Jev takes about a quarter of a second. That is still roughly 1,700 emails an
hour, enough for most mailboxes.

**Laya is fast and free but not accurate enough.** It runs locally, as fast as Jev, but it filed
about 1 in 5 emails in the wrong category.

**A large general-purpose AI model (gpt-5.6-terra) is the most accurate, but the slowest and
most expensive option.** It costs about 65 times more than Jev.

## The models

| model | made by | where it runs | cost per 1,000 emails | data leaves our machine? |
| --- | --- | --- | ---: | --- |
| **Jev 1.13** | TypeSafe | their cloud service | $0.05 | yes |
| **gpt-5.6-terra** | OpenAI | their cloud service | $3.51 | yes |
| **Rune 26B-A4B v3** | Invergent | our Strix Halo box | $0 | no |
| **Laya** | Convai Innovations | our Strix Halo box | $0 | no |
| **GLiNER 2.5** | Fastino | our DGX Spark | $0 | no |

Jev, Rune, Laya and GLiNER are **decision models**. They don't write text. They pick from the
answers you allow and say how sure they are. That makes them fast, cheap and easy to check.
gpt-5.6-terra is a general chat model that we prompt to answer the same questions.

"$0" means no per-email charge. The local models still need the hardware and its electricity.

## How accurate was each model?

Every model answered the same 8 questions about each email:
- **Category**: one of 8 kinds (support, sales, billing, HR, vendor, internal, spam, other).
- **Priority**: four levels, from "no action needed" to "act within two days".
- **Six yes/no flags**, for example "is the sender waiting for a reply?" and "does this mention a
  deadline?".

| | category correct | priority exactly right | priority within one level | yes/no flags correct | time per email |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-5.6-terra | **97%** | **91%** | **100%** | **91%** | 1.6 s |
| Jev | 96% | 78% | 92% | 89% | 0.3 s |
| **Rune** | **96%** | 74% | **100%** | **90%** | 2.1 s |
| Laya | 78% | 41% | 81% | 73% | 0.3 s |
| GLiNER | 46% | 16% | 70% | 46% | 0.1 s |

In plain terms:
- **Rune and Jev are tied on category.** Each missed the same number of emails, 3 of 74.
- **On priority, Rune is never far off.** It was off by at most one level on every email. Jev
  was off by two or more levels on 6 emails.
- **The yes/no flags are a near-tie between Rune and Jev.** Rune's weak spot is "does this need a
  decision from us?", which it got right 61% of the time against Jev's 74%. Everything else is
  close.
- **Laya struggles with these questions.** It confuses support with internal email and sales
  with vendor email. It also gets "is the sender waiting for a reply?" backwards more often than
  not.
- **GLiNER is no better than a coin flip on the yes/no flags.** It isn't usable here as
  configured.

## Does the model know when it's unsure?

This matters as much as accuracy: you can only automate the emails where the model is confident.
We checked whether each model's confidence matched how often it was actually right:
- **Jev, Rune and gpt-5.6-terra**: on their most confident 70% of emails, all three were 100%
  right on category. They are safe to automate for the high-confidence cases.
- **Laya**: on its most confident 70%, it was still wrong 21% of the time. Its confidence doesn't
  separate good answers from bad ones.

## What we recommend

1. **If data must stay in-house, use Rune.** It matches Jev's accuracy with no per-email cost and
   no data leaving the building. It's already running on the Strix Halo box.
2. **If speed matters most and cloud is acceptable, keep Jev.** It's 8 times faster than Rune and
   still very cheap.
3. **Don't use Laya or GLiNER for this task** without further work. They're quick, but too often
   wrong.
4. **Don't use the general chat model for bulk triage.** It's only slightly more accurate than
   Jev or Rune, and about 65 times more expensive than Jev.

## Caveats

- **The test set is small.** 74 synthetic emails is enough to separate good models from bad
  ones. It isn't enough to rank models within a couple of percentage points of each other.
- **Routing to human review is conservative for every model.** The current routing policy sent
  70 of 74 Jev emails, 71 Rune emails and all 74 Laya emails to human review. Before anything is
  auto-filed, the thresholds need tuning on a larger, held-out set.
- **We ran Rune in its standard mode.** Its optional "thinking" mode, which reasons briefly when
  unsure, only exists in Invergent's own server, and that server targets NVIDIA GPUs.

---

# Technical details

## Where the runs are

All runs used the 8-question contract in [`questions.py`](../src/jev_email_cascade/questions.py)
and `data/emails.jsonl`, and were scored by [`report.py`](../src/jev_email_cascade/report.py).

| backend | run | configuration |
| --- | --- | --- |
| jev | [`20260919T131239Z`](../runs/20260919T131239Z) | typesafe/jev-1.13-20260917 |
| frontier | [`20260919T184646Z`](../runs/20260919T184646Z) | gpt-5.6-terra |
| gliner | [`20260920T012913Z`](../runs/20260920T012913Z) | fastino/gliner2.5-multi-v1, GB10 |
| rune | [`20261004T013051Z`](../runs/20261004T013051Z) | Rune-26B-A4B-v3-Q8_0.gguf, llama.cpp b11382 Vulkan, raw state |
| rune | [`20261004T013328Z`](../runs/20261004T013328Z) | same, rendered state |
| laya | [`20261004T003302Z`](../runs/20261004T003302Z) | `typed-decisions`, rendered state, ROCm |
| laya | [`20261004T003240Z`](../runs/20261004T003240Z) / [`003217Z`](../runs/20261004T003217Z) / [`003314Z`](../runs/20261004T003314Z) | typed-decisions raw / english / multilingual, ROCm |
| laya | [`20261004T001120Z`](../runs/20261004T001120Z) / [`001500Z`](../runs/20261004T001500Z) / [`001752Z`](../runs/20261004T001752Z) | english / typed-decisions / english rendered, CPU |

```bash
uv run cascade compare runs/20260919T131239Z runs/20260919T184646Z runs/20261004T013051Z \
  runs/20261004T003302Z runs/20260920T012913Z --markdown
```

| backend | category acc | macro F1 | acc @70% coverage | priority exact / ±1 | multi-level priority errors | noul Brier / ECE | p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frontier | 0.973 | 0.967 | 1.000 | 0.905 / 1.000 | 0 | 0.071 / 0.072 | 1,600 / 3,710 |
| jev | 0.959 | 0.955 | 1.000 | 0.784 / 0.919 | 6 | 0.079 / 0.097 | 272 / 449 |
| rune Q8_0 | 0.959 | 0.958 | 1.000 | 0.743 / 1.000 | 0 | 0.090 / 0.113 | 2,080 / 2,150 |
| laya typed-decisions | 0.784 | 0.789 | 0.788 | 0.405 / 0.811 | 14 | 0.165 / 0.149 | 274 / 296 |
| gliner | 0.459 | 0.405 | 0.558 | 0.162 / 0.703 | 22 | 0.318 / 0.330 | 79 / 94 |

Per-flag raw accuracy (P(true) ≥ 0.5 counted as "yes"):

| flag | jev | frontier | rune | laya | gliner |
| --- | ---: | ---: | ---: | ---: | ---: |
| awaiting_reply | 0.85 | 0.81 | 0.88 | 0.42 | 0.46 |
| deadline_present | 0.86 | 0.99 | 0.96 | 0.82 | 0.53 |
| dissatisfied | 0.88 | 0.86 | 0.92 | 0.85 | 0.36 |
| needs_decision | 0.74 | 0.77 | 0.61 | 0.49 | 0.49 |
| opportunity | 1.00 | 1.00 | 1.00 | 0.81 | 0.50 |
| injection_suspected | 1.00 | 1.00 | 1.00 | 0.97 | 0.39 |

## How the local models are served

Both local models live in
[local-decision-model-halo](https://github.com/skiingfalcon/local-decision-model-halo)
(`C:\Users\kghosh\projects\local-decision-model`). Each runs as a supervised per-user scheduled
task that restarts its server if it exits. Both speak Jev's own `POST /v1/systemone` protocol,
so one client, [`laya_client.py`](../src/jev_email_cascade/laya_client.py), serves `--backend laya`
and `--backend rune` with the same request body `JevClient` sends.

| | Rune 26B-A4B v3 | Laya 0.3.26 |
| --- | --- | --- |
| architecture | Gemma 4 MoE: 26.5B parameters, 4B active (8 of 128 experts) | ModernBERT-large encoder with decision heads, 421M |
| weights | `owao/surogate-rune-26b-a4b-systemone` @ `f6cf5c19`, Q8_0 GGUF, 25 GB | `convaiinnovations/laya`, laya's reviewed revision |
| server | llama.cpp `llama-server` b11382 (PR #29818 added `/v1/systemone`) | `laya-serve` (FastAPI) |
| GPU path | Vulkan | ROCm: AMD's `torch 2.11.0+rocm7.13.0` for gfx1151 |
| port / task | 8001 / `RuneServe` | 8000 / `LayaServe` |
| how it decides | renders the surogate prompt; softmax over the option letters' logits at T = 2 | typed decision heads; one forward pass per question set |

How Rune's GGUF works: owao's file is Invergent's weights, unmodified, with five GGUF header keys
added: `gemma4.decision.type = openjev`, temperature 2 for each question type, and the decision
prompt template. With those keys, `llama-server` serves decisions with no extra flags. The
official repo (`surogate/rune-26b-a4b-GGUF`) is gated, holds bf16 safetensors, and is meant for
Invergent's surogate engine on NVIDIA.

## Speed, throughput and memory on the Strix Halo box

Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB with 96 GB carved out for the GPU.
`scripts/bench_laya.py`, 74 emails at about 1,500–2,300 prompt tokens each across the 8 questions:

| configuration | p50 / p95 per email | throughput | resident memory | bench |
| --- | ---: | ---: | --- | --- |
| Rune Q8_0, Vulkan, 1 slot | 2,082 / 2,149 ms | 0.48 emails/s | 26.9 GB GPU, 1 GB RAM | [`bench-…014208Z`](../runs/bench-20261004T014208Z-rune-q8-vulkan-np1/bench.md) |
| Rune Q8_0, Vulkan, 4 slots | 2,317 ms seq | 0.43 emails/s | | [`bench-…015421Z`](../runs/bench-20261004T015421Z-rune-q8-vulkan-np4/bench.md) |
| Laya english, ROCm | 248 / 268 ms | 4.0 emails/s | 7.5 GB GPU; up to 59 GB cached after batch-64 | [`bench-…003644Z`](../runs/bench-20261004T003644Z-rocm-english/bench.md) |
| Laya typed-decisions, ROCm | 279 / 301 ms | 3.4 emails/s | | [`bench-…004028Z`](../runs/bench-20261004T004028Z-rocm-typed/bench.md) |
| Laya english, CPU (16 threads) | 2,397 / 2,548 ms | 0.4 emails/s | 5 GB RAM | [`bench-…003036Z`](../runs/bench-20261004T003036Z-cpu-english/bench.md) |

- **Rune is limited by prompt processing.** Extra llama-server slots only add queueing.
- **Laya runs one inference at a time.** Neither concurrency nor its `/batch` route raised
  throughput.
- **On the first request after a start**, Laya takes about 1.9 s while ROCm kernels load, and
  Rune about 2.1 s.

## Things that didn't work, and the workarounds

- **Rune BF16 (47 GB)** loads, but every request fails with `vk::Queue::submit: ErrorDeviceLost`
  on b11382 Vulkan with driver 32.0.31041, even at `-ub 128`. Livesport measured Q8_0 at 98.7%
  top-option agreement with bf16, so Q8_0 is the configured default.
- **llama.cpp's official `win-rocm` build** lists no GPU on this driver, so Rune runs on Vulkan.
- **ONNX/DirectML for Laya** isn't wired into `laya-serve` 0.3.26, which runs torch only. AMD's
  ROCm torch wheels for gfx1151 give the GPU path instead. They crash in rocBLAS unless the venv
  path is short, because Windows' 260-character path limit is exceeded otherwise.
- **The corporate proxy (Cisco Umbrella)** re-signs huggingface.co and GitHub downloads and
  redirects every file through a session handshake. local-decision-model handles this with three
  pieces:
  - a CA bundle that includes Cisco's published root,
  - a Python 3.13 hook that relaxes only the strict X.509 flag,
  - an httpx transport that completes the handshake.

  After the one-time weight download, both servers run offline.

## Reproducing

```bash
# servers: see local-decision-model-halo's README (setup-tls, setup-rocm, setup-llama, fetch-*, install-task)
uv run cascade run --backend rune                       # RUNE_URL defaults to http://127.0.0.1:8001/v1/systemone
LAYA_MODEL=typed-decisions LAYA_STATE_MODE=rendered uv run cascade run --backend laya
uv run python scripts/bench_laya.py --backend rune --label rune-q8
make test-live                                          # round trips against both servers
```
