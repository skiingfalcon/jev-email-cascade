Notes for presentation

Text & Semantic Processing
 Generative Autoregressive Models (The Writers): Models like GPT-4, Claude, and Llama (Decoder-only). They predict and generate output one token at a time. They excel at reasoning, drafting, and open-ended conversation, but sequential generation makes them computationally heavy and slow for basic routing.
 Encoder Models (The Readers): Models like BERT and RoBERTa. They process an entire sequence simultaneously to understand bidirectional context. Today, they are predominantly used as Embedding Models to convert text into mathematical vectors for semantic search and retrieval.
 Parallel Classification / System One Models (The Deciders): Models like TypeSafe's Jev. These are transformer-based but completely abandon token-by-token generation. They run a single mathematical forward pass on an unstructured state to simultaneously output probabilistic judgments across multiple structured questions.  


They are all Transformer models, and they all rely on the exact same core scaled dot-product attention equation.
The fundamental mathematical operation shared by all three families is:
Attention(Q, K, V) = softmax\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
(Where Q is Queries, K is Keys, V is Values, and M is the Mask).
If the core equation is identical, the different behaviors (reading, writing, or deciding) come from how they alter the Mask (M) inside that equation, and what they put at the very end of the neural network (the output head).
Here is how the three families alter that same equation to achieve different results:
* Generative Autoregressive Models (GPT, Llama): These use a Causal Mask. In the equation, the mask (M) blocks out all "future" tokens by setting their values to negative infinity before the softmax is applied. A word is only allowed to pay attention to itself and the words that came before it. The final layer of the network is a "Language Modeling Head," which outputs a probability distribution over the vocabulary to guess the next word.
* Encoder Models (BERT, RoBERTa): These use Bidirectional Attention (an open mask). In the equation, M allows every token to pay attention to every other token in the sequence simultaneously—both past and future. Because they don't need to generate text token-by-token, they look at the whole sentence at once to understand deep context. The final layer typically outputs dense mathematical vectors (embeddings) representing the meaning of the text.
* Parallel Classification Models (Jev): These models also use Bidirectional Attention to process the input text (the "state") all at once, allowing for full contextual understanding. However, instead of a Language Modeling Head that generates vocabulary tokens, the final layer consists of Classification Heads tied to specific primitives (Choice, Score, Noul). The network does the exact same attention math, but projects the final numbers directly into decision probabilities rather than words.
In short: the attention equation calculates how words relate to each other. Generative models restrict that relationship to look backward to predict the future. Encoders and parallel classifiers let all words look at each other simultaneously to build a complete picture, differing only in whether they output a representation vector or a final categorical decision.


Jev

Jev (TypeSafe AI; the build we ran is `typesafe/jev-1.13-20260917`, echoed in every response -- 1.13 is a version number, so earlier releases exist) is a "System One" AI model designed specifically for fast, structured decision-making rather than conversational text generation. Instead of generating chains of reasoning or chat responses, Jev acts as a probabilistic decision engine that takes in a "state" (the context) and evaluates it against specific typed questions.
The three core concepts—or primitives—that define how Jev models evaluate data are Choice, Score, and Noul.
1. Choice (Selecting from defined options)
A Choice primitive acts as a classifier. You provide a set of bounded, predefined options (the OpenRouter/TypeSafe docs we read allow up to 255), and Jev selects the single best fit based on the provided state. Always include an escape option ("other").
* How it works: Along with the winning candidate, Jev returns the probability distribution across the entire set of options and a confidence value.
* Use case: Routing a customer support ticket to either billing, technical support, or account access.
2. Score (Evaluating against a descriptive rubric)
A Score primitive evaluates the input against an ordered, descriptive scale. Rather than returning a bare number from 1 to 10 (which is highly subjective for AI), it requires you to define specific levels.
* How it works: It returns an expected rubric index (which can include fractional values between levels) alongside the probability distribution across all levels.
* Use case: Grading incident severity from level 1 ("cosmetic issue") to level 2 ("broken feature with workaround") to level 3 ("completely blocked").
3. Noul (Estimating the probability of "yes")
A Noul primitive evaluates a strict yes/no proposition and returns a continuous value between 0 and 1.
* How it works: Unlike Choice and Score, Noul doesn't output a separate confidence metric; the probability itself is the signal your application uses to act.
* Use case: Asking "Does this message request a refund?" or "Does this passage contain a safeguarding concern?"
How they work together: In a typical Jev implementation, your software provides a single snapshot of data (the "state") and asks multiple Choice, Score, and Noul questions simultaneously. Because Jev processes these in parallel without autoregressive token generation, it returns these structured semantic judgments in a fraction of the time a conventional LLM takes, allowing standard application code to execute the final actions instantly.


It is possible to make a generative model like Qwen behave like a decision model by strapping a classification head onto it. (Projects claiming to do this -- "openjev" was mentioned in an earlier draft -- are unverified; do not name one from the stage unless you have checked it.)
Doing that does not make it identical to a natively built "System One" decision model, but be precise about why:
1. One forward pass either way. A classification head on a decoder is also a single forward pass -- it is not sequential generation. What you carry is a model sized to write code and poetry, and a causal mask: each token only attends to the tokens before it, so the "reading" is one-directional. Asking a chat model to answer 8 questions in JSON, by contrast, *is* autoregressive: it writes the answer token by token (Terra spent 9,733 output tokens plus 1,451 hidden reasoning tokens doing exactly that on our 74 emails).
2. Output guarantees. A generative model can always emit text that breaks the schema. A decision model has no language head: it can only return probabilities over the options you gave it. Say "cannot emit text," not "cannot hallucinate" -- it can still be confidently wrong. In our run Jev was confidently wrong (>=0.80 or <=0.20 against the label) on 21 of 444 yes/no answers; Terra on 29. Most of Jev's are labels we would argue about ourselves (a refund request scored needs_decision 0.85-0.88; an outage scored dissatisfied 0.96).
3. Training objective. TypeSafe describes Jev as trained for calibrated decisions rather than for pleasing a reader. The specific acronym "RLCD" in an earlier draft is unverified -- describe the goal (calibration: an 0.80 should be right about 80% of the time), not a method name we cannot source. Our calibration table is the evidence we actually have: Jev's needs_decision 0.3-0.7 band held 26 emails with a 0% positive rate -- it said "cannot tell" precisely where our own labels were shakiest.
In short: you can bend a generative model into a decider, but you pay for a model built to write, and you get a number it typed rather than a probability it computed.


Architecture: what we know vs. what we infer. Structurally there are two Transformer families for text -- decoders (autoregressive) and encoders (bidirectional). "Parallel classification" / "System One" is a shift in what the last layer emits, not a third base architecture. Traditional encoders (BERT) read a text and emit a vector for search; a decider passes the same encoded context through classification heads and emits a Choice, Score, or Noul. TypeSafe has not published Jev's internals; "single forward pass, typed heads" is their public description and "encoder" is our reading of it. Say so if asked. (An earlier draft named a second model, "Laya" -- unverified; dropped.)


The open-weight comparison (slide "The free one"). GLiNER2.5 (fastino/gliner2.5-multi-v1, Apache 2.0, 287M params, mDeBERTa-v3 encoder) is the same class of model: text plus a schema of typed labels in, labels with scores out, one forward pass, no generation. It ran on the Spark's GB10 in 6.4 s for all 74 emails, 79 ms p50 / 94 ms p95, $0 -- 3.4x faster than Jev, 20x faster than Terra.
Accuracy: 45.9% category, 16.2% priority exact (70.3% within one), 45.5% mean on the six yes/no flags -- a coin flip. Every one of the six flags sat in the 0.2-0.8 "cannot tell" band on 50-59 of 74 emails; the model was never confident about anything. Category predictions piled onto "internal" (36 of 74) and "vendor" (16), against 9 true each.
Why we think it is our integration, not the model: GLiNER2 takes label *descriptions*, and our schema builder (`gliner_backend.py`, `_build_schema`) has a fallback that catches a TypeError and passes bare label names with no criteria. A model guessing from the words "internal" and "vendor" alone would behave exactly like this. It is untested against the live library because torch does not run on the dev box.
Verdict for the room: first wiring, not a verdict. Fix, re-run, then compare -- it is the only on-prem option on the table.


The needs_decision counterfactual (slide "What we would change"). On the existing Jev results, dropping needs_decision as a route lever entirely (both as a "true → escalate" reason and as an uncertain item) yields 12 auto instead of 4; one of the 12 has a wrong label. Not 35. What still blocks the other 62 emails: dissatisfied true (12), deadline_present in the cannot-tell band (11), category uncertain (8), dissatisfied cannot-tell (6), awaiting_reply cannot-tell (6), priority flat or low-confidence (5), injection (3). So the second lever after needs_decision is deadline_present's wording -- Jev under-detects explicit "within 24/48 hours" (0.25-0.35) and over-detects routine clock times like "stand-up moved to 10am" (0.86-0.89). Do not lower noul_act to get more autos; that buys confident mistakes.
