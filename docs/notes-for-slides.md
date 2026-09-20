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

Jev (released by TypeSafe AI in September 2026) is a "System One" AI model designed specifically for fast, structured decision-making rather than conversational text generation. Instead of generating chains of reasoning or chat responses, Jev acts as a probabilistic decision engine that takes in a "state" (the context) and evaluates it against specific typed questions.
The three core concepts—or primitives—that define how Jev models evaluate data are Choice, Score, and Noul.
1. Choice (Selecting from defined options)
A Choice primitive acts as a classifier. You provide a set of bounded, predefined options (from 2 to 50 candidates), and Jev selects the single best fit based on the provided state.
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


It is entirely possible to force a generative model like Qwen to behave like a Jev model—in fact, open-source projects like openjev do exactly this by strapping a classification head onto Qwen to make it evaluate text.
However, forcing a generative model to act as a classifier does not make it identical to a natively built "System One" decision model. They remain separate families because of fundamental differences in architecture, speed, and training objectives.
Here is why a native parallel classification model is in a class of its own:
1. True Parallelism vs. Sequential Generation
If you ask a standard LLM (even a fast one like Qwen) to answer 15 distinct classification questions about a user's prompt, it still has to write out that JSON response autoregressively—token by token, question by question. A native parallel classifier uses a custom parallel sampler. It runs one single forward pass over the text and outputs the probabilities for all 15 questions simultaneously in a fraction of a second.
2. Output Guarantees (Type Safety)
A generative model is fundamentally designed to generate arbitrary text. Even if you aggressively prompt it to only output "Category A" or "Category B," the model's architecture still physically allows it to output "Category C," apologize, or break the JSON schema. A native decision model like Jev lacks a language generation head entirely. It is physically incapable of hallucinating text; it can only return mathematical probabilities mapped to your predefined options.
3. The Training Objective (RLHF vs. RLCD)
Generative models are trained using RLHF (Reinforcement Learning from Human Feedback) to sound helpful, human, and authoritative. As a side effect, this makes them notoriously overconfident—an LLM will often confidently state the wrong answer.
Models in the Jev family are trained using a completely different methodology called RLCD (Reinforcement Learning for Calibrated Decisions). Instead of being trained to please a human reader, they are optimized purely for mathematical calibration. This means if the model claims it is 80% confident in a Choice routing, it will actually be correct 80% of the time. This calibration allows developers to reliably write code that says, "If confidence is below 90%, route to a human."
In short: you can modify a generative model like Qwen to act like Jev, but you are essentially using a sports car to tow a trailer. It carries the massive architectural weight of a language model designed to write code and poetry, making it slower, more expensive, and less reliable than a model structurally stripped down to do nothing but make decisions.


You are absolutely right. Structurally, there are really only two fundamental Transformer architectures for processing text: Decoders (autoregressive generation) and Encoders (bidirectional reading).
The reason "Parallel Classification" or "System One" models are often treated as a distinct third category is because of a major shift in application and output, rather than a brand-new base architecture.
Here is how the encoder family essentially split into two distinct roles:
* Traditional Encoders (The Searchers): For years, encoder models like BERT were primarily used to create embeddings. Their job was to read a text, understand the context, and output a dense mathematical vector so that a database could perform semantic search or retrieval.
* System One Models (The Deciders): Models like Jev and Laya take that exact same bidirectional encoder architecture but swap the final output step. Instead of spitting out an embedding vector to be stored in a database, they pass the encoded context through a classification head to instantly output a usable decision (a Choice, Score, or Noul probability).
So, under the hood, a parallel classifier is an encoder. The distinction simply highlights whether you are using that encoder's deep reading capability to index information for later, or to make an instant, structured decision right now.


You are absolutely right. Structurally, there are really only two fundamental Transformer architectures for processing text: Decoders (autoregressive generation) and Encoders (bidirectional reading).
The reason "Parallel Classification" or "System One" models are often treated as a distinct third category is because of a major shift in application and output, rather than a brand-new base architecture.
Here is how the encoder family essentially split into two distinct roles:
* Traditional Encoders (The Searchers): For years, encoder models like BERT were primarily used to create embeddings. Their job was to read a text, understand the context, and output a dense mathematical vector so that a database could perform semantic search or retrieval.
* System One Models (The Deciders): Models like Jev and Laya take that exact same bidirectional encoder architecture but swap the final output step. Instead of spitting out an embedding vector to be stored in a database, they pass the encoded context through a classification head to instantly output a usable decision (a Choice, Score, or Noul probability).
So, under the hood, a parallel classifier is an encoder. The distinction simply highlights whether you are using that encoder's deep reading capability to index information for later, or to make an instant, structured decision right now.
