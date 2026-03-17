# Discovery: Creative Blending Through Interference Patterns

**Date:** 2026-02-13
**Project:** Muse (HeatherDB sample project)
**Finding:** Averaging concept vectors and querying SDM produces genuinely novel blends that don't exist in storage — the reconstruction emerges from interference patterns across all stored concepts. Vector DBs return one stored item; SDM creates something new.

## Setup

- Concepts embedded as 384-dim vectors (sentence-transformers)
- All concepts written to SDM, superimposing into shared memory
- **Blend**: average 2+ concept vectors → query SDM → reconstruction is a NEW pattern
- **Spark**: add heavy noise (30-50%) to one concept → SDM free-associates
- **Explore**: random walk through concept space via repeated noise + reconstruction

## Key Findings

### 1. Reconstruction Creates Novel Patterns

When you average "jazz" and "gothic architecture" and query SDM:
- **Vector DB**: returns whichever stored concept is nearest to the midpoint — either "jazz" or "gothic," not both
- **SDM**: reconstructs a pattern influenced by BOTH concepts simultaneously — the interference of all stored patterns creates something that doesn't exist in storage

The nearest concepts to this reconstruction naturally bridge both input domains — they're the stored ideas that live at the intersection.

### 2. Noise as Creative Tool

Three noise levels produce three distinct creative modes:

| Mode | Noise | Effect |
|---|---|---|
| **Blend** | 5% | Small perturbation — generates variety while staying near the concept intersection |
| **Spark** | 30-50% | Heavy perturbation — pushes far from input, SDM pulls toward unexpected attractors |
| **Explore** | 15-35% | Random walk — each hop discovers adjacent semantic regions |

With a vector DB, adding noise just degrades the query. With SDM, noise becomes a creative force — it pushes the query into new basins of attraction, surfacing unexpected associations.

### 3. Attractor Basins as Creative Spaces

EAM's attractor landscape defines the "creative space":
- Frequently stored concepts form deep attractors — they dominate blends
- Rare concepts form shallow attractors — they surface only with direct cues
- The spaces BETWEEN attractors are where novel ideas emerge
- Noise-driven exploration maps this landscape

## Why This Matters

| | Vector DB | Muse (SDM) |
|---|---|---|
| Blend query | Returns nearest stored item to midpoint | Reconstructs NEW pattern from interference |
| Adding noise | Degrades results | Creative exploration (spark, explore) |
| Multiple concepts | Returns items near the average | Genuine blend influenced by all stored patterns |
| Concept space | Flat (every vector independent) | Landscape of attractors (basins, valleys, ridges) |

The fundamental insight: **interference is creative**. In a vector DB, interference between stored items is a bug (contamination). In SDM, interference is the mechanism that generates novelty — blended reconstructions are genuinely new patterns that emerge from the superposition of everything stored.

## Parameters
- Dimension: 384 (sentence-transformers default)
- Blend noise: 0.05, Spark noise: 0.30-0.50, Explore noise: 0.15-0.35
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
