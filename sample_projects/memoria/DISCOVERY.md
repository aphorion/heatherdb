# Discovery: Associative Memory Behaves Like Human Recall

**Date:** 2026-02-13
**Project:** Memoria (HeatherDB sample project)
**Finding:** SDM-based conversational memory produces human-like associative recall — partial cues reconstruct full memories, frequently discussed topics become more "vivid," and dream-mode free association generates coherent thought chains from noise.

## Setup

- Conversation turns embedded as 384-dim vectors (sentence-transformers)
- Each exchange (user message + response summary) written to SDM
- Recall: embed query → SDM reconstructs → find nearest cached texts to reconstruction
- Memories labeled by fidelity: "vivid" (>85%), "clear" (>70%), "vague impression" (>50%)

## Key Findings

### 1. Reconstruction ≠ Retrieval

A vector DB returns the single closest stored text. SDM reconstructs a **blended pattern** influenced by ALL stored memories. The nearest texts to this reconstruction often include memories that are thematically related but wouldn't be the top-1 nearest to the query itself.

This means partial or vague queries ("that thing we talked about earlier") still reconstruct meaningful patterns — the EAM completes the cue.

### 2. Memory Reinforcement Through Superposition

Topics discussed repeatedly form deeper basins in SDM. Their reconstructions have higher fidelity (vivid), while one-off mentions produce lower fidelity (vague). This happens automatically through superposition — no explicit frequency tracking.

| Memory type | Fidelity | Behavior |
|---|---|---|
| Frequently discussed | High (vivid) | Dominates reconstruction, recalled easily |
| Mentioned once | Low (vague) | Fades into background, recalled with effort |
| Never discussed | Very low | SDM returns noise — "I don't remember" |

This mirrors human memory: rehearsed memories strengthen, unrehearsed ones decay.

### 3. Dream Mode — Free Association from Noise

The killer feature impossible with vector DBs:
1. Pick a random cached vector
2. Add 10-30% noise
3. Query SDM — it reconstructs a coherent pattern despite the noise
4. Chain 4 hops, each triggered by the previous reconstruction

This produces **free-association chains** — sequences of related memories connected by EAM's attractor landscape. A vector DB queried with random noise returns garbage. EAM's basins of attraction pull noisy queries toward meaningful patterns.

## Why This Matters

| | Vector DB (RAG) | Memoria (SDM) |
|---|---|---|
| Storage | Each chunk indexed separately | Vectors superimposed in shared memory |
| Query | Nearest-neighbor lookup | Pattern completion via reconstruction |
| Partial cues | Fails — needs semantic overlap | Works — SDM completes partial patterns |
| Repeated topics | All equally retrievable | Become more vivid (deeper basins) |
| Dream/free-association | Impossible (noise → garbage) | Works (noise → nearest attractor) |
| Mental model | Filing cabinet | Human associative memory |

## Parameters
- Dimension: 384 (sentence-transformers default)
- Embedding model: all-MiniLM-L6-v2
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
