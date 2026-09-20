### runs/20260919T131239Z
backend=`jev` model=`typesafe/jev-1.13-20260917` n=74

## Input & output

**Input:** 74 emails -- subject, sender, and body (quoted history and signatures stripped, body capped at 6,000 characters) -- each sent to the `jev` backend along with the same 8 typed questions below (1 call(s) per email).

**Output:** one typed answer per question -- a Noul, Choice, or Score, explained next -- turned by a threshold policy into a final category, priority, six yes/no flags, and a route (`auto` / `review` / `llm`). Written to `results.jsonl`; the tables below score those answers against each email's true labels.

## Question types

Every backend answers the same 8 questions, each typed as one of three kinds (`src/jev_email_cascade/questions.py`):

| type | what it means | used for |
| --- | --- | --- |
| **Noul** | A yes/no question answered as a *probability* (0-1), not a boolean. 0.5 means "cannot tell", not "somewhat true" -- the policy treats anything between its thresholds as uncertain rather than rounding it. | `awaiting_reply`, `deadline_present`, `dissatisfied`, `needs_decision`, `opportunity`, `injection_suspected` |
| **Choice** | Pick exactly one option from a fixed, named list, with a confidence score. Every Choice needs an escape option (`other`) so an out-of-taxonomy email gets a low-confidence answer instead of a confident wrong one. | `category` (8 options) |
| **Score** | A position on an ordered scale of levels, each described in plain language, not just a number. The reported value is a probability-weighted expectation across the levels, so "2.7" means real uncertainty between levels 2 and 3, not a fractional level. | `priority` (4 levels, 0-3) |

## Summary

- category accuracy: 0.959 (CI 0.91-1.00)
- priority exact: 0.784 (CI 0.69-0.88), within-1: 0.919 (CI 0.85-0.97)
- routes: {'review': 70, 'auto': 4}
- cost: $0.00398 total, $0.0538/1K emails, 1 calls/email
- latency p50/p95: 272/449 ms
- errors: 0

| question | raw acc | 95% CI | acted acc (n) | mean conf on wrong |
| --- | ---: | ---: | ---: | ---: |
| awaiting_reply | 0.851 | 0.77-0.93 | 0.866 (67/74) | 0.89 |
| deadline_present | 0.865 | 0.78-0.93 | 0.952 (63/74) | 0.748 |
| dissatisfied | 0.878 | 0.80-0.95 | 0.912 (68/74) | 0.806 |
| needs_decision | 0.743 | 0.65-0.84 | 0.919 (37/74) | 0.668 |
| opportunity | 1 | 1.00-1.00 | 1 (74/74) | - |
| injection_suspected | 1 | 1.00-1.00 | 1 (73/74) | - |

**Category confusion** (rows = true, cols = predicted)

| true \ pred | billing | hr | internal | other | sales | spam | support | vendor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| billing | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hr | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| internal | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| sales | 0 | 0 | 0 | 1 | 9 | 0 | 0 | 0 |
| spam | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| support | 0 | 1 | 0 | 1 | 0 | 0 | 10 | 0 |
| vendor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |

**Noul calibration** (observed positive rate by predicted-probability band)

| question | 0.0-0.1 | 0.1-0.3 | 0.3-0.7 | 0.7-0.9 | 0.9-1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| awaiting_reply | 0 (n=8) | 0 (n=1) | 0.667 (n=6) | 1 (n=1) | 0.845 (n=58) |
| deadline_present | 0 (n=45) | 0.364 (n=11) | 1 (n=3) | 0.75 (n=12) | 1 (n=3) |
| dissatisfied | 0 (n=55) | 0 (n=3) | 0 (n=3) | 0 (n=3) | 0.7 (n=10) |
| needs_decision | 0 (n=10) | 0 (n=19) | 0 (n=26) | 0.3 (n=10) | 1 (n=9) |
| opportunity | 0 (n=57) | 0 (n=8) | - (n=0) | - (n=0) | 1 (n=9) |
| injection_suspected | 0 (n=71) | 0 (n=1) | - (n=0) | - (n=0) | 1 (n=2) |

## Analysis

First live call to Jev (via OpenRouter), replacing the mock backend used in earlier runs.
Category (95.9%) and the two sharpest Nouls (opportunity, injection_suspected, both 100%) are
strong -- category alone beats the mock's 86.5% on the same data. Two findings below are worth
acting on before drawing conclusions from thresholds.

### Jev mishandles time references in both directions

- **Explicit numeric deadlines under-detected.** `support-20/21/22` ("...need this fixed
  immediately... within 24 hours") and `billing-20/21/22` ("respond within 48 hours or the
  payment will be delayed") scored `deadline_present` **0.25-0.35** -- a confident miss on
  unambiguous phrasing.
- **Routine scheduling over-triggers it.** `internal-00/01/02` ("stand-up moved to 10am", no
  deadline) scored `deadline_present` **0.86-0.89** and `priority` **1.66-1.71** against a true
  label of 0. `internal-10/11/12` ("let's sync tomorrow") scored `priority` **2.67-2.75** against
  a true label of 1. Any specific clock time or near-term date seems to move the needle
  regardless of whether it is framed as a deadline.
- **Padding buries a real deadline.** `special-long-01` -- "URGENT: response needed within 24
  hours" at the very top of a ~12K-char body, followed by ~11KB of repeated filler -- scored
  `deadline_present` **0.29** and missed category too (`other`, confidence 0.39). This is the
  README's "retrieve first, judge second" caveat, measured rather than asserted.

### `needs_decision` drives almost all the routing, and it is soft

37 of 74 emails land `needs_decision` in the 0.2-0.8 "can't tell" band (see the calibration
table), and its raw accuracy (74.3%) is the weakest of any question -- this is why 70/74 emails
escalate to `review` even with no LLM configured. Compare to `opportunity` and
`injection_suspected`, both crisp and both 100%.

### Caveat: some of the "misses" above are probably my labels, not Jev's errors

- `billing-00/01/02` ("charged twice... need a refund processed") scored `needs_decision`
  **0.85-0.88**; labelled `False`. Our own question text says "approve... or commit money or
  resources" -- approving a refund plausibly is that.
- `support-20/21/22` and `hr-20/21/22` (outage, payroll error) scored `dissatisfied`
  **0.57-0.96** against a `False` label -- an outage demand and "escalate this" plausibly carry
  frustration without an explicit "frustrated" keyword.
- `special-negation-03` ("I'm not frustrated at all... onboarding went smoothly") got category
  `hr` (confidence 0.65) against a `support` label -- "onboarding" is genuinely ambiguous between
  the two in this taxonomy.

### Next steps

1. Re-run at volume before retuning thresholds -- single-run noise plus the label ambiguity
   above make 74 items too few to calibrate on.
2. Fix the label issues identified above and re-score to see how much of the miss count survives.
3. The real lever is `needs_decision`'s wording, not its threshold: narrow the instructions (e.g.
   explicitly exclude "approve a refund/routine correction") or stop using it as an automatic
   escalation trigger.
4. Re-run with `--llm` pointed at the Spark's gpt-oss-120b to see whether the 70 escalated items
   actually need a human, or whether the LLM hook resolves most of them cleanly.
