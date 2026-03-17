# Recall — SDM as Long-Term Associative Memory for LLM Conversations

## The Hypothesis

LLM conversations have fixed context windows. When the window fills up, old messages are truncated and lost. What if SDM could serve as long-term associative memory — writing every exchange into distributed memory and reconstructing relevant past context when a topic returns?

The biological parallel: **context window = working memory, SDM = long-term memory**.

## The Experiment

1. Chat about Topic A (cooking pasta with semolina flour, San Marzano tomato sauce)
2. Chat about Topic B (Python decorators, context managers, asyncio, type hints) for 4 messages
3. Ask about Topic A again: "what was that flour I mentioned for pasta earlier?"
4. Measure: Does the EAM recall the pasta conversation? How confident is it?

## Results

| Metric | Value |
|--------|-------|
| Topic A recalled after 4 intervening messages | **Yes — "semolina flour" surfaced** |
| Fidelity on first message (new topic) | **16%** |
| Fidelity on returning topic | **66%** |
| Recalled memory similarity | **85% (labeled "clear")** |
| Fidelity on repeated same-topic message | **16% → 66%** (jumps on second message) |

All 4 E2E tests passed, verified with Playwright automation.

## How It Works

### Write Path
Every exchange gets embedded (all-MiniLM-L6-v2, 384d → 128d random projection) and written to SDM as 3 vectors:
- User message embedding
- Assistant response embedding
- Combined exchange embedding

Messages + embeddings also stored in SQLite for retrieval.

### Read Path
1. Embed the new user message
2. SDM read (iterative/Hopfield) → reconstruction (not a stored vector — a blend of everything that activates)
3. Compute **fidelity** = cosine similarity between query and reconstruction
4. Find stored messages nearest to the **reconstruction** (not the query — this is key)
5. Adaptive token budget: high fidelity → more recalled context, low fidelity → more recent messages
6. Build Claude prompt: system + recalled context + recent messages + current message

### Fidelity-Adaptive Budgeting
```
sdm_budget = SDM_BASE + int((fidelity - 0.5) * 2000)
sdm_budget = clamp(500, 3000)
recent_budget = TOTAL_BUDGET - SYSTEM_TOKENS - sdm_budget
```

High fidelity = "I recognize this topic" → trust recalled memories, allocate more tokens to them.
Low fidelity = "this is new" → lean on recent messages, don't trust sparse recall.

### Memory Labeling
- **Vivid** (>0.85 similarity) — strong match, highly relevant
- **Clear** (>0.70) — moderate match, likely relevant
- **Vague** (>0.50) — weak match, use cautiously

## Why This Works (and Drift Didn't)

The Drift experiment tried to use SDM for **chained multi-hop reasoning** — feeding reconstructions back as queries, hoping for emergent thought chains. Chains collapsed in 2-3 steps because EAM's iterative read already converges to an attractor; feeding the reconstruction back just confirms it.

Recall works because it uses SDM for **single-shot associative recall**. One write, one read, one reconstruction. The LLM does the reasoning; the EAM provides the memory.

The key difference:
- **Drift**: SDM as the intelligence (failed — needs orders of magnitude more data)
- **Recall**: SDM as the memory, LLM as the intelligence (works — each does what it's good at)

## What SDM Adds Over Vector Search (k-NN)

1. **Associative reconstruction**: k-NN returns the single nearest stored vector. SDM returns a *blend* of all vectors that activate in the neighborhood. If you discussed "Python decorators" in message #5 and "JavaScript closures" in message #30, querying "function wrappers" returns interference from BOTH — k-NN would only return the single nearest.

2. **Superposition compression**: 100 messages about "code architecture" don't need 100 entries retrieved. The EAM reconstruction IS the compressed representation — the interference pattern of all those conversations.

3. **Fidelity as novelty signal**: Cosine similarity between query and reconstruction tells you how familiar the topic is. This is native to EAM's mechanism — vector search gives you distance to nearest neighbor, which is different (you can be close to one vector but the topic is still novel overall).

## Architecture

```
User → [Next.js Frontend] → POST /api/chat → [FastAPI Engine]
                                                → embed message (all-MiniLM-L6-v2 → 128d)
                                                → SDM read (associative recall)
                                                → compute fidelity
                                                → find messages nearest to reconstruction
                                                → adaptive token budget
                                                → build prompt: system + recalled + recent + current
                                                → Claude API (claude-sonnet-4-5)
                                                → SDM write (3 vectors per exchange)
                                                → SQLite store (messages + embeddings)
                                                → return response + recalled memories + fidelity
```

## Parameters
- Dimensions: **128** (good interference for moderate dataset)
- Embedding model: all-MiniLM-L6-v2 (384d) with random projection to 128d
- SDM read strategy: iterative (Hopfield convergence, beta=5.0)
- Token budget: 8000 total (300 system, 2000 base SDM, 4000 base recent, ±2000 adaptive)
- Memory threshold: >0.50 similarity to be recalled
- HeatherDB: default settings (1000 initial hard locations, k=20 activation)

## Key Insight

EAM's sweet spot is **single operations on embedded data**: one-shot write, associative recall, pattern completion, vector arithmetic. It fails at chained reasoning but excels at memory. Pairing it with an LLM — where SDM handles storage/recall and the LLM handles reasoning — produces something neither can do alone: conversations with unbounded associative memory.
