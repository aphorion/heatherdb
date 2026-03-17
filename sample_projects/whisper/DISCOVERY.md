# Discovery: Lossy Semantic Compression — Documents Dissolve Into Impressions

**Date:** 2026-02-13
**Project:** Whisper (HeatherDB sample project)
**Finding:** When document sentences are superimposed into EAM, the memory becomes a lossy semantic compressor — frequent themes become vivid, rare details fade, and queries reconstruct impressions rather than exact passages. This mirrors how human memory compresses experience.

## Setup

- Documents split into sentences, each embedded as 384-dim vector
- ALL sentence vectors written to SDM in one batch — they superimpose into shared hard locations
- Query: embed question → SDM reconstructs → find nearest cached sentences to reconstruction
- Fidelity = cosine(query, reconstruction) — measures how strongly memory responds

## Key Findings

### 1. Frequency = Vividness

Themes repeated across many sentences form deeper basins in SDM. Their reconstructions have high fidelity — the memory "responds strongly." One-off details produce weak reconstructions.

| Content type | Reconstruction fidelity | Behavior |
|---|---|---|
| Central themes | High (vivid) | Dominate reconstruction, recalled easily |
| Supporting details | Medium (clear) | Present but not dominant |
| Mentioned once | Low (faded) | Barely present, may not surface |

This is **lossy compression by frequency** — no explicit weighting, just superposition physics. Common patterns reinforce; rare ones don't.

### 2. Reconstruction ≠ Retrieval

Querying SDM doesn't return stored sentences. It returns a reconstructed vector that's been influenced by ALL stored sentences. Finding nearest cached sentences to this reconstruction produces results that are:
- Thematically coherent (not just keyword-matched)
- Cross-referencing (influenced by multiple parts of the document)
- Sometimes surprising (unexpected connections between passages)

### 3. Neighborhood Probing — Exploring Adjacent Meaning

Adding noise to the query vector and reconstructing again reveals **adjacent semantic regions** — related concepts that aren't directly queried but live nearby in EAM's attractor landscape.

This is impossible with vector DBs: nearest-neighbor search with noise just returns worse matches. EAM's attractor structure means noisy queries still converge to meaningful patterns — just different ones.

### 4. Cross-Document Emergence

Load two unrelated documents into the same SDM. Queries reconstruct from the interference of both. This creates unexpected connections:
- A query about topic A might surface fragments from document B that share hidden thematic overlap
- The EAM "discovers" relationships the user didn't explicitly create

## The Compression Analogy

| | JPEG (lossy image) | Whisper (lossy semantic) |
|---|---|---|
| What's preserved | Low-frequency components | Frequent themes |
| What's lost | High-frequency detail | Rare one-off mentions |
| Reconstruction | Approximate image | Approximate meaning |
| Compression ratio | Tunable (quality parameter) | Tunable (dimension, # hard locations) |
| Artifacts | Blocking, ringing | Blending, fading |

## Why This Matters

| | Vector DB (RAG) | Whisper (SDM) |
|---|---|---|
| Storage | 847 chunks indexed separately | 847 sentences superimposed in shared memory |
| Query result | Top-k nearest stored chunks | Reconstruction influenced by ALL sentences |
| Frequent themes | All equally retrievable | More vivid (stronger attractors) |
| Rare details | Equally retrievable | Faded (weaker signal) — true compression |
| Cross-document | Returns chunks from one doc | Interference creates cross-document connections |
| Mental model | Search engine | Human memory of a book read long ago |

The key insight: **forgetting is a feature, not a bug.** A perfect memory retrieves everything equally. A useful memory compresses — it knows what matters (frequent, reinforced) and lets the rest fade. SDM does this naturally through superposition.

## Parameters
- Dimension: 384 (sentence-transformers default)
- Embedding model: all-MiniLM-L6-v2
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
