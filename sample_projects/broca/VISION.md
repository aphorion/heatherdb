# Broca: Beyond Token-by-Token Generation

## The Problem with Autoregressive Generation

Every modern language model — GPT, Claude, Llama — generates text one token at a time. Encode context, predict next token, append, repeat. This is a computational artifact of the transformer architecture, not a reflection of how language actually works.

Humans don't think token by token. When you recall a quote, the whole phrase activates at once. When you form a sentence, you have the thought first, then you speak it. You don't form one word, say it, form the next word, say it.

## The EAM Insight

An Elastic Associative Memory doesn't predict — it **pattern-completes**. You present a query, and the Hopfield dynamics converge to an **attractor**: a full, rich pattern in memory space. Not a single next-token logit. A complete thought.

This means the architecture doesn't need to be autoregressive at all.

## Two-Level Architecture: Thinking + Speaking

### Level 1 — Thinking (one EAM read)

Encode the current context into semantic space. The EAM converges to an attractor that represents "what comes next" — not as a token, but as a **meaning**. One read from storage. One round trip. The full continuation encoded as a dense vector.

### Level 2 — Speaking (decoder renders the thought)

A small sequence decoder (GRU, tiny transformer) takes the thought-vector and renders it into a token sequence — 10, 20, 50 tokens at once. This is a deterministic rendering step, not a creative/knowledge step. The knowledge was already retrieved in Level 1.

```
Context: "To be or not to be, that is the"
    → Encoder: semantic query vector [128d]
    → EAM read (one disk access): attractor = thought-vector [128d]
    → Decoder: "question. Whether 'tis nobler in the mind to suffer"
```

One retrieval from storage per **thought**, not per **character**.

## Why This Matters

### Speed
Token-by-token generation with a 1M-location EAM means 1M location scans per token (or k=20 lookups per token). For 500 characters, that's 500 round trips to storage. With thought-level generation: **one** round trip, then pure in-memory decoding.

### Storage I/O
The EAM lives in LMDB on disk/flash. LMDB is fast, but disk access is still orders of magnitude slower than RAM. Minimizing disk reads from 500 to 1 per generation is the difference between usable and unusable on edge devices.

### Biological Plausibility
This mirrors how biological memory actually works:
- **Hippocampus** (EAM): single associative retrieval of a memory trace
- **Broca's area** (decoder): renders the retrieved thought into speech
- **Working memory**: holds the current context, not all of language

You don't access long-term memory for every syllable. You retrieve once, then speak from working memory.

### Composability
Thought-vectors are algebraically composable. You could:
- **Add** two thought-vectors: blend ideas from different knowledge bases
- **Subtract**: remove a concept from a thought
- **Interpolate**: smoothly transition between two ideas

This is impossible with token-level autoregressive generation, where each token is a discrete, independent prediction.

## The Stepping Stones

1. **Broca v1** (current): Character-level, token-by-token. Proves the EAM-backed LM works at all.
2. **Broca v2**: Chunk-level decoding. One EAM read → N tokens. The decoder becomes a small sequence model.
3. **Broca v3**: Semantic encoding. Encode meaning, not character sequences. The EAM stores and retrieves at the concept level.
4. **Broca v4**: Multi-scale. Sentence-level EAM for high-level structure, phrase-level for detail. Composed via algebra.

## The Storage Advantage

At GPT-2 scale (768-dim, 500K locations):

| | RAM | Disk |
|---|---|---|
| Controller + decoder | ~550 MB | ~550 MB |
| EAM (knowledge) | ~240 KB per read | ~6 GB |
| **Total at inference** | **~550 MB** | **~6.5 GB** |

A transformer must load ALL parameters into RAM. The EAM model loads a small neural net into RAM and touches a tiny sliver of knowledge per thought. The rest stays on flash.

A phone with 4 GB RAM and 64 GB storage could run this. The knowledge store could be **arbitrarily large** — bounded by disk, not memory.

## The Vision

A language model where:
- The neural network is tiny (~500 MB) and learns HOW to think
- The knowledge is massive (GB-TB) and lives on storage
- Knowledge is swappable: same brain, different expertise
- Knowledge is composable: algebra on thought-spaces
- Knowledge is inspectable: see which memories activated
- Knowledge is editable: add or remove specific information
- Generation happens at the speed of thought, not the speed of tokens
