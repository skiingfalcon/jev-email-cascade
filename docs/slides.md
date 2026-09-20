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
You can strap a classification head on Qwen (openjev). That is still sequential generation or a heavy decoder doing a reader's job.
-->

---

# Why this exists

Inbox volume is **routing**, not drafting.

A chat model can do it.  
Slow, expensive, overconfident, and it might not even return JSON.

<!--
The product problem: classify the email, then auto / review / (maybe) draft.
Generative models are built to write. We need probabilities we can put in an if-statement.
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
Backends: Jev, gpt-5.6-terra, (later) a self-hosted encoder. Same questions, same report.
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

| | Jev | gpt-5.6-terra |
| --- | ---: | ---: |
| Category | 95.9% | 97.3% |
| Priority exact | 78.4% | 90.5% |
| Flags (mean) | 89% | 91% |
| Auto | 4 | 26 |
| Cost | **$0.004** | $0.26 |
| Latency p50 | **272 ms** | 1.6 s |

Accuracy is close. Cost and speed are not.

<!--
Same questions, same labels, same scorer. Terra list price ~65× Jev. OpenRouter billed Jev.
Opportunity and injection: 100% both.
Jev 70/74 review — almost all needs_decision in the "cannot tell" band. Question wording, not "Jev can't classify."
-->

---

# The number that matters

When they were **wrong** on category:

- Jev said 0.39 – 0.65 → **review**
- Terra said 0.93 – 0.96 → would have **auto-filed**

Jev's 4 autos were clean.  
Terra's 26 autos: 8 had a wrong flag.

<!--
Calibration vs a number the model typed in JSON. This is the routing argument.
-->

---

# What we would change to auto more

Not a new model.

1. Stop auto-escalating `needs_decision`
2. Tighten that question's wording
3. Then retune thresholds

<!--
Counterfactual on the existing Jev file: ignore needs_decision as a route lever → ~35 auto, not 4.
Do not lower noul_act first — cheap auto, more confident mistakes.
-->

---

# Takeaway

Jev is a **decider**.  
Python is the **router**.  
Chat earns its cost only on leftovers.

<!--
Hosted US API — compliance is a separate decision. Open-weight encoder path exists; first wiring was not a fair test.
Ask: volume run after the needs_decision wording fix.
-->
