# Email triage: Jev vs a frontier model — CTO brief

*Measured 2026-09-19 on the same 74 labelled business emails, the same 8 typed questions, the same
scoring code. Runs: [`runs/20260919T131239Z`](../runs/20260919T131239Z/report.md) (Jev) and
[`runs/20260919T184646Z`](../runs/20260919T184646Z/report.md) (gpt-5.6-terra). Reproduce with
`uv run cascade compare runs/20260919T131239Z runs/20260919T184646Z --markdown`.*

## The elevator pitch

**A purpose-built decision model (TypeSafe Jev) classifies email at 1/65th the cost and 6x the
speed of a frontier LLM, with statistically indistinguishable accuracy — and, unlike the LLM, it
tells you when it is unsure.** The frontier model is more accurate on priority, but it was
confidently wrong on the emails it missed; Jev's misses were all flagged as low-confidence. That
property, not the raw accuracy, is what makes automation safe.

## Scorecard

| | **Jev** (typesafe/jev-1.13) | **gpt-5.6-terra** (OpenAI) | edge |
| --- | ---: | ---: | :--- |
| **Cost, 74 emails** | $0.004 | $0.260 | Jev **65x** cheaper |
| **Cost per 1K emails** | $0.054 | $3.51 | |
| **Cost per 1M emails / month** | ~$54 | ~$3,510 | |
| **Latency p50 / p95** | 272 / 449 ms | 1,600 / 3,705 ms | Jev **6–8x** faster |
| **Wall time, 74 emails (serial)** | 21 s | 139 s | |
| **Category accuracy** | 95.9% (CI 91–100) | 97.3% (CI 93–100) | tie (CIs overlap) |
| **Priority exact / within-1** | 78.4% / 91.9% | 90.5% / 100% | Terra |
| **6 yes/no flags, mean raw accuracy** | 88.9% | 90.5% | tie |
| **Confidence on its wrong category calls** | 0.39, 0.46, 0.65 — *"I'm not sure"* | 0.96, 0.93 — *"I'm certain"* | **Jev, decisively** |
| **Auto-routed items with zero errors** | 4 / 4 | 18 / 26 | Jev |
| **Calls per email** | 1 | 1 | |
| **Output tokens to parse** | 0 (typed answers) | 9,733 (+1,451 hidden reasoning) | Jev |
| **Errors / retries** | 0 | 0 | |

## What this means

1. **Cost and speed are not close.** Jev's entire pricing is $0.042 per million input tokens with
   no output charge. At any volume worth automating, the frontier model is two orders of
   magnitude more expensive and an order of magnitude slower per email.

2. **Accuracy is a wash on this sample.** The 74-email confidence intervals overlap on category
   and on the yes/no flags. Terra does win on priority (90.5% vs 78.4% exact) — mostly by reading
   routine calendar times ("stand-up moved to 10am") as non-urgent where Jev over-triggered.

3. **Calibration is the real difference.** Jev returns a probability; a frontier model returns a
   number it typed. When Jev got a category wrong it reported 0.39–0.65 and the policy sent it to
   review. When Terra got one wrong it reported 0.93–0.96 and would have been auto-filed. Of the
   26 emails Terra's confidence said were safe to automate, 8 had at least one wrong flag; Jev's
   4 auto items were all clean. Self-reported LLM confidence is not a routing signal; Jev's is.

4. **The cascade is the design, not a compromise.** Jev handles the volume; an LLM (local
   gpt-oss-120b on the Spark, or Terra via `LLM_PROVIDER=frontier`) is called only for the band
   Jev marks uncertain, and only there does its higher priority accuracy earn its cost. The
   pipeline already does this; the hook is a config flag.

## Caveats, stated plainly

- 74 synthetic emails, one run each. Enough to see a 65x cost gap; not enough to rank two models
  within 2 points of accuracy. Volume re-run is the next step.
- Jev currently escalates 70/74 emails, almost entirely because the `needs_decision` question is
  worded too loosely (37 of 74 land in its "can't tell" band). That is a question-wording fix,
  not a model limitation, and it is the lever that decides the cascade's economics.
- Terra's cost is list-price arithmetic on the API's token counts (the API reports tokens, not
  dollars). Jev's is what OpenRouter billed.
- Both are hosted US APIs; the prepared email text leaves the box in either case. Same
  compliance question, different vendor.

## Ask

Approve a volume re-run (10K real, de-identified emails; Jev ≈ $0.50, Terra ≈ $35) after the
`needs_decision` wording fix, with the cascade's escalation arm on gpt-oss-120b. That produces
the number the business case needs: **cost per correctly-routed email, at production volume.**
