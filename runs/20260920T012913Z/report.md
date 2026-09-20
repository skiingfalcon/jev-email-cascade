### runs/20260920T012913Z
backend=`gliner` model=`fastino/gliner2.5-multi-v1` n=74

## Input & output

**Input:** 74 emails -- subject, sender, and body (quoted history and signatures stripped, body capped at 6,000 characters) -- each sent to the `gliner` backend along with the same 8 typed questions below (1 call(s) per email).

**Output:** one typed answer per question -- a Noul, Choice, or Score, explained next -- turned by a threshold policy into a final category, priority, six yes/no flags, and a route (`auto` / `review` / `llm`). Written to `results.jsonl`; the tables below score those answers against each email's true labels.

## Question types

Every backend answers the same 8 questions, each typed as one of three kinds (`src/jev_email_cascade/questions.py`):

| type | what it means | used for |
| --- | --- | --- |
| **Noul** | A yes/no question answered as a *probability* (0-1), not a boolean. 0.5 means "cannot tell", not "somewhat true" -- the policy treats anything between its thresholds as uncertain rather than rounding it. | `awaiting_reply`, `deadline_present`, `dissatisfied`, `needs_decision`, `opportunity`, `injection_suspected` |
| **Choice** | Pick exactly one option from a fixed, named list, with a confidence score. Every Choice needs an escape option (`other`) so an out-of-taxonomy email gets a low-confidence answer instead of a confident wrong one. | `category` (8 options) |
| **Score** | A position on an ordered scale of levels, each described in plain language, not just a number. The reported value is a probability-weighted expectation across the levels, so "2.7" means real uncertainty between levels 2 and 3, not a fractional level. | `priority` (4 levels, 0-3) |

## Summary

- category accuracy: 0.459 (CI 0.35-0.58)
- priority exact: 0.162 (CI 0.08-0.26), within-1: 0.703 (CI 0.59-0.80)
- routes: {'review': 73, 'auto': 1}
- cost: $0 total, $0/1K emails, 1 calls/email
- latency p50/p95: 79/93.6 ms
- errors: 0
- tokens: in 6,246 / out 0

| question | raw acc | 95% CI | acted acc (n) | mean conf on wrong |
| --- | ---: | ---: | ---: | ---: |
| awaiting_reply | 0.459 | 0.35-0.57 | 0.208 (24/74) | 0.775 |
| deadline_present | 0.527 | 0.42-0.64 | 0.8 (15/74) | 0.68 |
| dissatisfied | 0.365 | 0.26-0.49 | 0.5 (20/74) | 0.702 |
| needs_decision | 0.486 | 0.38-0.59 | 0.619 (21/74) | 0.691 |
| opportunity | 0.5 | 0.39-0.61 | 0.75 (20/74) | 0.671 |
| injection_suspected | 0.392 | 0.28-0.50 | 0.522 (23/74) | 0.72 |

**Category confusion** (rows = true, cols = predicted)

| true \ pred | billing | hr | internal | sales | spam | support | vendor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| billing | 9 | 0 | 0 | 0 | 1 | 1 | 0 |
| hr | 0 | 3 | 6 | 0 | 0 | 0 | 0 |
| internal | 0 | 0 | 6 | 0 | 0 | 0 | 3 |
| other | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| sales | 0 | 0 | 1 | 6 | 0 | 0 | 3 |
| spam | 0 | 0 | 8 | 0 | 0 | 0 | 1 |
| support | 0 | 1 | 10 | 0 | 0 | 1 | 0 |
| vendor | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

**Noul calibration** (observed positive rate by predicted-probability band)

| question | 0.0-0.1 | 0.1-0.3 | 0.3-0.7 | 0.7-0.9 | 0.9-1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| awaiting_reply | 0.846 (n=13) | 0.692 (n=13) | 0.889 (n=36) | 0.167 (n=12) | - (n=0) |
| deadline_present | 0.231 (n=13) | 0 (n=12) | 0.432 (n=37) | 0 (n=12) | - (n=0) |
| dissatisfied | 0.308 (n=13) | 0.25 (n=12) | 0 (n=33) | 0 (n=16) | - (n=0) |
| needs_decision | 0.231 (n=13) | 0 (n=13) | 0.212 (n=33) | 0.133 (n=15) | - (n=0) |
| opportunity | 0 (n=13) | 0.231 (n=13) | 0.121 (n=33) | 0.133 (n=15) | - (n=0) |
| injection_suspected | 0.167 (n=12) | 0 (n=10) | 0 (n=27) | 0 (n=25) | - (n=0) |
