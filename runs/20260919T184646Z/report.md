### runs/20260919T184646Z
backend=`frontier` model=`gpt-5.6-terra` n=74

## Input & output

**Input:** 74 emails -- subject, sender, and body (quoted history and signatures stripped, body capped at 6,000 characters) -- each sent to the `frontier` backend along with the same 8 typed questions below (1 call(s) per email).

**Output:** one typed answer per question -- a Noul, Choice, or Score, explained next -- turned by a threshold policy into a final category, priority, six yes/no flags, and a route (`auto` / `review` / `llm`). Written to `results.jsonl`; the tables below score those answers against each email's true labels.

## Question types

Every backend answers the same 8 questions, each typed as one of three kinds (`src/jev_email_cascade/questions.py`):

| type | what it means | used for |
| --- | --- | --- |
| **Noul** | A yes/no question answered as a *probability* (0-1), not a boolean. 0.5 means "cannot tell", not "somewhat true" -- the policy treats anything between its thresholds as uncertain rather than rounding it. | `awaiting_reply`, `deadline_present`, `dissatisfied`, `needs_decision`, `opportunity`, `injection_suspected` |
| **Choice** | Pick exactly one option from a fixed, named list, with a confidence score. Every Choice needs an escape option (`other`) so an out-of-taxonomy email gets a low-confidence answer instead of a confident wrong one. | `category` (8 options) |
| **Score** | A position on an ordered scale of levels, each described in plain language, not just a number. The reported value is a probability-weighted expectation across the levels, so "2.7" means real uncertainty between levels 2 and 3, not a fractional level. | `priority` (4 levels, 0-3) |

## Summary

- category accuracy: 0.973 (CI 0.93-1.00)
- priority exact: 0.905 (CI 0.84-0.96), within-1: 1 (CI 1.00-1.00)
- routes: {'review': 48, 'auto': 26}
- cost: $0.26 total, $3.51/1K emails, 1 calls/email
- latency p50/p95: 1.6e+03/3.71e+03 ms
- errors: 0
- tokens: in 71,654 (cached 0) / out 9,733 (reasoning 1,451)
- pricing: list price $2.0/M in, $0.2/M cached, $12.0/M out

| question | raw acc | 95% CI | acted acc (n) | mean conf on wrong |
| --- | ---: | ---: | ---: | ---: |
| awaiting_reply | 0.811 | 0.72-0.89 | 0.843 (70/74) | 0.89 |
| deadline_present | 0.986 | 0.96-1.00 | 0.986 (74/74) | 0.98 |
| dissatisfied | 0.865 | 0.78-0.93 | 0.877 (73/74) | 0.888 |
| needs_decision | 0.77 | 0.68-0.86 | 0.873 (63/74) | 0.795 |
| opportunity | 1 | 1.00-1.00 | 1 (74/74) | - |
| injection_suspected | 1 | 1.00-1.00 | 1 (74/74) | - |

**Category confusion** (rows = true, cols = predicted)

| true \ pred | billing | hr | internal | other | sales | spam | support | vendor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| billing | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hr | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| internal | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| sales | 0 | 0 | 0 | 1 | 9 | 0 | 0 | 0 |
| spam | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| support | 0 | 0 | 0 | 1 | 0 | 0 | 11 | 0 |
| vendor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

**Noul calibration** (observed positive rate by predicted-probability band)

| question | 0.0-0.1 | 0.1-0.3 | 0.3-0.7 | 0.7-0.9 | 0.9-1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| awaiting_reply | 0 (n=8) | 0.6 (n=5) | 1 (n=1) | 0 (n=1) | 0.847 (n=59) |
| deadline_present | 0.0179 (n=56) | - (n=0) | - (n=0) | 1 (n=1) | 1 (n=17) |
| dissatisfied | 0 (n=55) | 0.25 (n=4) | - (n=0) | 0 (n=4) | 0.545 (n=11) |
| needs_decision | 0 (n=37) | 0.1 (n=10) | - (n=0) | 0.118 (n=17) | 0.9 (n=10) |
| opportunity | 0 (n=65) | - (n=0) | - (n=0) | - (n=0) | 1 (n=9) |
| injection_suspected | 0 (n=72) | - (n=0) | - (n=0) | - (n=0) | 1 (n=2) |
