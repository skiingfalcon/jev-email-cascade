---
marp: true
paginate: true
title: Jev for email triage
---

# Decision models, not chat models

Email triage with TypeSafe Jev

<!--
Title only. One sentence: this is a proof that a decision model can label and route email without generating text.
-->

---

# Three jobs for a transformer

| | Role | Examples |
| --- | --- | --- |
| **Writers** | next token | GPT, Claude, Llama |
| **Readers** | whole sequence → vector | BERT, embeddings |
| **Deciders** | whole sequence → judgments | Jev |

Same family of model. Different last layer.

<!--
All three are transformers. Same scaled-dot-product attention. Difference is the mask and the head.
Writers: causal mask + vocab head. Readers: open mask + embedding. Deciders: open mask + classification heads.
-->

---

# Same math

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}+M}{\sqrt{d_k}}\right)V$$

$M$ is the mask. The **head** is what you read out.

<!--
Walk the equation if the room wants it. Otherwise: mask decides who can look at whom; the head decides whether you get a word, a vector, or a probability.
-->

---

# How they differ

- **Writer** — causal mask. Guess the next word.
- **Reader** — open mask. Emit an embedding.
- **Decider** — open mask. Emit Choice / Score / Noul.

A decider is an encoder used for a decision, not for search.

<!--
Don't overclaim a third architecture. Structurally: decoder vs encoder. "System One" is what you do with the encoder's last layer.
Hedge if asked: TypeSafe has not published Jev's internals. "Single forward pass, typed heads" is what they say publicly; "encoder" is our reading of that.
You can strap a classification head on a decoder like Qwen. That is one forward pass too, not generation -- the cost is that you carry a model sized to write, and a causal mask that only looks backward.
-->

---

# Why this exists

Inbox volume is **routing**, not drafting.

A chat model can do it.  
Slow, expensive, and the confidence it types is not a probability.

<!--
The product problem: classify the email, then auto / review / (maybe) draft.
Generative models are built to write. We need probabilities we can put in an if-statement.
Don't say "it won't return JSON" -- Terra returned parseable JSON 74/74 in our run. The problem is what the number inside the JSON means.
-->

---

# The bet

**Jev decides. Ordinary code routes. A generative model only handles leftovers.**

<!--
Cascade, not a better prompt. Policy is Python thresholds. LLM is optional, and only for the uncertain band.
-->

---

# This project

74 synthetic labelled emails  
8 typed questions  
One shared answer schema  
Score against gold labels

<!--
Not a live inbox. Generator + answer key. Enough to measure accuracy, cost, latency, and whether we can auto-route.
Three backends ran: Jev (via OpenRouter), gpt-5.6-terra (OpenAI), and GLiNER2.5 (open-weight encoder, self-hosted on the Spark's GPU). Same questions, same report.
-->

---

# What we send Jev

**State** — prepared email  
(from, subject, date, body)

**Questions** — not a prompt

- 1 Choice
- 1 Score
- 6 Noul

One forward pass. No prose back.

<!--
prepare() strips quotes/signature, caps length. Jev never sees the gold labels.
-->

---

# Three output types

**Choice** — one option + distribution + confidence  
**Score** — position on a rubric (can be fractional)  
**Noul** — P(yes) in [0, 1]. No separate confidence.

<!--
Choice: category. Always include "other".
Score: priority 0–3 with observable criteria, not "high/medium/low".
Noul: 0.5 means cannot tell, not "somewhat". Never round it.
-->

---

# Choice — `category`

support · sales · billing · hr  
vendor · internal · spam · **other**

Returns the winner, the full distribution, and a confidence.

<!--
"other" is the escape hatch. Without it, an odd email gets a confident wrong bucket.
Billing needs more confidence downstream (refunds) — that is policy, not the model.
-->

---

# Score — `priority`

| | |
| --- | --- |
| 0 | no action |
| 1 | reply wanted, nothing at risk |
| 2 | act in a few days |
| 3 | act within ~2 days |

Expected index + distribution across levels.

<!--
Rubric is observable facts in the email, not adjectives. A 2.01 is "basically 2".
-->

---

# Noul — six flags

| Flag | Question |
| --- | --- |
| `awaiting_reply` | asks us to do something? |
| `deadline_present` | a real date/time? |
| `dissatisfied` | complaint? |
| `needs_decision` | judgment / money / approval? |
| `opportunity` | revenue for *us*? |
| `injection_suspected` | talking to the classifier? |

<!--
Noul is the probability. ≥ 0.80 act true, ≤ 0.20 act false, middle = cannot tell.
injection_suspected: "ignore previous instructions…" — always review.
opportunity: a vendor pitch is not our opportunity.
-->

---

# One email in

> Trouble connecting our account… I'm frustrated… Could you help as soon as possible?

<!--
support-00. Speaker can read it. Gold labels live on the file; Jev does not see them.
-->

---

# Jev out

| | |
| --- | --- |
| category | support · conf 1.0 |
| priority | 2.01 · conf 0.99 |
| awaiting_reply | 0.98 |
| deadline_present | 0.03 |
| dissatisfied | 0.96 |
| needs_decision | 0.18 |
| opportunity | 0.16 |
| injection_suspected | 0.02 |

Then **code** says: `route = review`  
(reason: dissatisfied)

<!--
This is the whole product output: typed answers + a route. Not a drafted reply.
dissatisfied true → never auto. Policy, not Jev.
A boring "how do I export the report?" with no complaint → auto.
-->

---

# What `auto` / `review` mean

**auto** — stop. No human, no LLM.  
**review** — queue a person, with the evidence attached.  
**llm** — optional draft, still not the decision.

<!--
Auto is not "send a reply." It is "the labels are trusted enough to file and move on."
Thresholds: category 0.60 (0.85 billing), priority 0.60, noul 0.80. Injection always review. needs_decision or dissatisfied → not auto.
-->

---

# Results — 74 emails

| | Jev | gpt-5.6-terra | GLiNER2.5* |
| --- | ---: | ---: | ---: |
| Category | 95.9% | 97.3% | 45.9% |
| Priority exact | 78.4% | 90.5% | 16.2% |
| Flags (mean) | 89.0% | 90.5% | 45.5% |
| Auto | 4 | 26 | 1 |
| Cost | **$0.004** | $0.26 | **$0** |
| Latency p50 | 272 ms | 1.6 s | **79 ms** |

Jev vs Terra: accuracy close, cost and speed not.  
\* first wiring — next slide.

<!--
Same questions, same labels, same scorer. Terra list price ~65× Jev. OpenRouter billed Jev. GLiNER ran on the Spark's GB10, $0 marginal.
Opportunity and injection: 100% for Jev and Terra.
Jev 70/74 review — almost all needs_decision in the "cannot tell" band. Question wording, not "Jev can't classify."
-->

---

# The free one

GLiNER2.5 — open-weight encoder, same class as Jev.  
Typed labels in, scores out, one forward pass.

**79 ms. $0. Scored at chance on yes/no.**

Every flag sat in the "cannot tell" band on 50–59 of 74 emails.

<!--
Same class of model: bidirectional encoder, schema of typed labels, one pass, no generation. Apache 2.0, 287M params, runs on our GPU.
It was never confident about anything -- 45.5% on yes/no is a coin flip. A model that reads "does this email state a deadline?" should do far better than that even at 287M.
Leading suspect is our wiring: GLiNER2 takes label descriptions; our schema builder has a fallback that may be sending bare label names ("internal", "vendor") with no criteria. Category predictions clustered on those two names -- 36 and 16 of 74.
Verdict: our integration, not the model. Fix before judging. It is the only on-prem option on the table, so it matters. Details: docs/cto-brief.md, item 3.
-->

---

# The number that matters

When they were **wrong** on category:

- Jev said 0.39 – 0.65 → **review**
- Terra said 0.93 – 0.96 → would have **auto-filed**

Jev's 4 autos were clean.  
Terra's 26 autos: 8 had a wrong flag.

On the six flags, both were confidently wrong sometimes:  
Jev 21, Terra 29 (of 444) — mostly on labels we'd argue about too.

<!--
Calibration vs a number the model typed in JSON. This is the routing argument.
Be fair: Jev is not never-wrong-with-confidence. billing-00/01/02 "process a refund" → needs_decision 0.85–0.88 against a False label; outage / payroll emails → dissatisfied 0.83–0.96. Our labels are arguable there. The point is fewer confident misses and honest 0.4–0.6s where it counts.
-->

---

# Where Jev was wrong

- "**within 24 hours**" / "**within 48 hours**" → `deadline_present` **0.25–0.35**
- "stand-up moved to **10am**" → `deadline_present` **0.86–0.89**, priority ~1.7 (true: 0)
- 12K-char padded email → category `other` 0.39, deadline 0.29 — both missed

<!--
Time references cut both ways: explicit numeric deadlines under-detected, routine clock times over-detected.
Padding buries the signal: "URGENT, within 24 hours" at the top of 11KB of filler, missed on both questions. Retrieve first, judge second -- measured, not asserted.
None of these is a threshold problem. deadline_present is the next question to reword after needs_decision.
-->

---

# What we would change to auto more

Not a new model.

1. Stop auto-escalating `needs_decision`
2. Tighten that question's wording — then `deadline_present`
3. Then retune thresholds

<!--
Counterfactual on the existing Jev file: drop needs_decision as a route lever entirely → 12 auto, not 4 (1 of the 12 has a wrong label). Not 35.
What still blocks the other 62: dissatisfied true (12), deadline_present in the cannot-tell band (11), category uncertain (8), dissatisfied cannot-tell (6), awaiting_reply cannot-tell (6), priority flat/low (5), injection (3).
Do not lower noul_act first — cheap auto, more confident mistakes.
-->

---

# Takeaway

Jev is a **decider**.  
Python is the **router**.  
Chat earns its cost only on leftovers.

<!--
Hosted US API — compliance is a separate decision. The open-weight encoder path exists and ran (79 ms, $0) but scored at chance; suspected cause is our schema wiring dropping label descriptions. Fix that before the volume run so the free on-prem option gets a fair test.
Ask: (1) fix and re-run GLiNER2.5; (2) volume run after the needs_decision wording fix.
-->
