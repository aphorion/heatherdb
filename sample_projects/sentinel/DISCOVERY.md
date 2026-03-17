# Discovery: SDM Reconstruction Shows What "Normal" Looks Like

**Date:** 2026-02-13
**Project:** Sentinel (HeatherDB sample project)
**Finding:** When SDM is trained on normal log patterns, it doesn't just flag anomalies — it reconstructs what the anomalous input SHOULD look like. The reconstruction IS the explanation: "you sent X, but normal looks like Y."

## Setup

- Normal log entries embedded as 384-dim vectors, written to SDM
- New entry checked: embed → SDM reconstructs → measure fidelity
- Fidelity > 0.80 = NORMAL, 0.65-0.80 = SUSPICIOUS, < 0.65 = ANOMALY
- Find nearest cached text to reconstruction → that's "what normal looks like"

## Key Findings

### 1. Anomaly Detection + Explanation in One Step

Most anomaly detectors give you a score. Sentinel gives you a score AND an explanation:

```
INPUT:    POST /api/admin/users 403 from 185.220.101.42 — 847 requests in 60s
FIDELITY: 0.31 — ANOMALY

EXPECTED: POST /api/users 201 from 10.0.1.15 — 1 requests in 60s
```

The reconstruction reveals exactly what's wrong: admin path (should be users), 403 (should be 201), external IP (should be internal), 847 requests (should be 1). No separate explainability system needed — the EAM's reconstruction IS the explanation.

### 2. Semantic Pattern Matching

Because embeddings capture semantic meaning, Sentinel catches anomalies that rule-based systems miss:
- "DELETE /api/users/1 403" is anomalous not because DELETE is banned, but because the EAM hasn't seen this COMBINATION before
- Path traversal attempts ("../../etc/shadow") are caught because they're semantically unlike any normal pattern
- Brute force (312 login attempts) is caught because the request rate makes the overall pattern dissimilar

### 3. Baseline Fidelity as System Health

The average fidelity during training establishes a baseline. If normal queries suddenly get lower fidelity, the EAM's attractor landscape has shifted — either:
- The system behavior has changed (concept drift)
- Something is polluting the memory
- The deployment environment differs from training

This makes fidelity a system-level health metric, not just per-query.

## Why This Matters

| | Threshold Rules | ML Anomaly Detection | Sentinel (SDM) |
|---|---|---|---|
| Detects anomaly | If rule matches | Score > threshold | Fidelity < threshold |
| Explains anomaly | No | No (black box) | Yes — shows "expected normal" |
| Adapts to patterns | Manual rules | Needs retraining | Write new normals to SDM |
| Setup | Write rules | Train model | Write normal logs |
| Novel attacks | Misses unknown patterns | Depends on training | Catches anything unlike normal |

The core insight: **the EAM's reconstruction is a generative model of "normal."** It doesn't memorize individual log lines — it learns the pattern distribution. The reconstruction shows what the statistical consensus of "normal" looks like for any given query, making every anomaly self-explaining.

## Parameters
- Dimension: 384 (sentence-transformers default)
- Normal threshold: 0.80 fidelity
- Anomaly threshold: 0.65 fidelity
- SDM: k=20 activation, Hopfield iterative read, beta=5.0
