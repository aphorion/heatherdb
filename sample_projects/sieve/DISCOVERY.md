# Discovery: Reconstruction Fidelity as a Natural Duplicate Detector

**Date:** 2026-02-13
**Project:** Sieve (HeatherDB sample project)
**Finding:** EAM reconstruction fidelity naturally distinguishes duplicates from unique content — high fidelity means "I've seen this before" (deep basin), low fidelity means "this is new" (no basin). No classifier needed.

## Setup

- Text records embedded as 384-dim vectors (sentence-transformers)
- Normal/known records written to SDM
- New record checked: embed → query SDM → measure fidelity (cosine of query vs reconstruction)
- Fidelity > 0.75 = DUPLICATE, 0.50-0.75 = SIMILAR, < 0.50 = UNIQUE

## Key Findings

### 1. Fidelity as Similarity Metric

When you write "I can't log in" to SDM and later query with "Unable to sign into my account":
- **Vector DB**: returns the stored text with a cosine similarity score
- **SDM**: reconstructs a pattern pulled toward the stored basin. Fidelity measures how strongly the query activates existing memory

The key difference: EAM's fidelity reflects how well the query fits into the **overall pattern distribution**, not just distance to one stored item. If many similar complaints exist, the basin is deeper → higher fidelity → stronger duplicate signal.

### 2. Frequency-Aware Detection

Patterns written multiple times form deeper basins. A complaint submitted 50 times has a much deeper attractor than one submitted once. This means:
- Frequent duplicates are caught with higher confidence
- One-off similar content doesn't trigger false positives as easily
- The system naturally adapts to what's "common" vs "rare"

Vector DBs treat every stored record equally — no frequency weighting.

### 3. No Training Required

Unlike ML duplicate classifiers that need labeled training data (duplicate/not-duplicate pairs), Sieve just writes records to SDM. The detection emerges from memory physics:
- Write records → basins form automatically
- Query new record → fidelity reveals whether a basin exists
- No labels, no training loop, no hyperparameter tuning

## Why This Matters

| | Vector DB Cosine | ML Classifier | Sieve (SDM Fidelity) |
|---|---|---|---|
| Detection method | Nearest-neighbor distance | Learned decision boundary | Reconstruction fidelity |
| Frequency aware | No | If trained on frequencies | Yes (deeper basins) |
| Training needed | None | Labeled pairs | None |
| Explains match | Shows nearest stored record | Black box | Shows what SDM "expected" |
| Threshold tuning | Manual similarity cutoff | Learned | Natural (fidelity scale) |

## Parameters
- Dimension: 384 (sentence-transformers default)
- Duplicate threshold: 0.75 fidelity
- Similar threshold: 0.50 fidelity
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
