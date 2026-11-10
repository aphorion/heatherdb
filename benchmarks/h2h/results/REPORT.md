# Head-to-head: one HeatherDB vs the specialist stack

Run 2026-11-11T00:20:07 to 2026-11-11T00:31:31 on macOS-26.5.1-arm64-arm-64bit. Engine `heather 1.0.0` at `v1.0.0-6-ge5dfbb2-dirty (feat/selective-attention)`.

## Accuracy, same data, same task

| Capability | Primary metric | HeatherDB | Best incumbent | Ratio | Verdict |
|---|---|---|---|---|---|
| Vector search | recall@10 | 1.000 | 0.997 (Qdrant (v1.12.4)) | 1.00 | parity or better |
| Pattern discovery / segmentation | nmi | 0.523 | 0.491 (k-means (HeatherDB's K)) | 1.06 | parity or better |
| Recommendations | ndcg@10 | 0.053 | 0.062 (ALS + recency (time-decayed confidence)) | 0.85 | gap |
| Fraud / anomaly detection | pr_auc | 0.199 | 0.485 (LOF) | 0.41 | gap |
| Propensity classification | roc_auc | 0.931 | 0.935 (Gradient-boosted trees) | 1.00 | parity or better |

### Vector search

| System | recall@10 | build_s | p50_ms | p95_ms | qps_single_client | server_cpu_s |
|---|---|---|---|---|---|---|
| HeatherDB | 1.000 | 2.600 | 1.894 | 4.616 | 328.213 | 7.900 |
| Qdrant (v1.12.4) | 0.997 | 17.397 | 4.270 | 5.914 | 232.163 | - |

### Pattern discovery / segmentation

| System | k | codebook_locations | fit_s | assign_s | nmi | ari | purity | server_cpu_s |
|---|---|---|---|---|---|---|---|---|
| HeatherDB (codebook + MDL) | 115 | 6397 | 22.476 | 4.687 | 0.523 | 0.307 | 0.648 | 118.369 |
| k-means (HeatherDB's K) | 115 | - | 1.322 | 0.019 | 0.491 | 0.107 | 0.755 | - |
| k-means (true K=10, oracle) | 10 | - | 0.362 | 0.011 | 0.561 | 0.371 | 0.558 | - |

At the true K=10 (an oracle choice), HeatherDB's codebook scores NMI 0.581 / ARI 0.413 / purity 0.569.

### Recommendations

| System | hr@10 | ndcg@10 | load_s | serve_ms_per_user | time_updates | server_cpu_s |
|---|---|---|---|---|---|---|
| HeatherDB (selective attention) | 0.105 | 0.053 | 304.129 | 0.934 | 315453 | 40.792 |
| HeatherDB (no selective attention) | 0.050 | 0.024 | 35.807 | 0.515 | - | 10.455 |
| Raw sum (same vectors, no memory) | 0.048 | 0.023 | - | - | - | - |
| Popularity | 0.029 | 0.016 | - | - | - | - |
| ItemKNN (cosine) | 0.061 | 0.032 | - | - | - | - |
| ALS (implicit, 64 factors) | 0.075 | 0.039 | - | - | - | - |
| ItemKNN + recency | 0.076 | 0.040 | - | - | - | - |
| ALS + recency (time-decayed confidence) | 0.123 | 0.062 | - | - | - | - |

Freshness: a new user's history is servable in 15 ms (p50) / 35 ms (p95) on HeatherDB, against a 2.8 s ALS refit (185x).

### Fraud / anomaly detection

| System | roc_auc | pr_auc | recall@1%fpr | fit_s | score_us_per_txn | server_cpu_s |
|---|---|---|---|---|---|---|
| HeatherDB | 0.953 | 0.199 | 0.711 | 0.210 | 89.239 | 89.615 |
| Isolation Forest | 0.949 | 0.127 | 0.589 | 0.208 | 5.729 | - |
| LOF | 0.951 | 0.485 | 0.785 | 0.928 | 18.822 | - |

### Propensity classification

| System | roc_auc | pr_auc | fit_s | score_us_per_row | server_cpu_s |
|---|---|---|---|---|---|
| HeatherDB (selective attention) | 0.931 | 0.605 | 9.162 | 971.746 | 106.607 |
| HeatherDB (no selective attention) | 0.867 | 0.480 | 0.226 | 229.545 | 6.319 |
| Logistic regression | 0.910 | 0.569 | 0.019 | 0.057 | - |
| Gradient-boosted trees | 0.935 | 0.629 | 0.459 | 0.783 | - |

## Infrastructure: everything above, deployed

Each incumbent is deployed as its own service on the smallest instance holding its measured peak memory x1.5. Models that must be refit to take new data add 30 retrain runs a month. Prices: AWS us-east-1 on-demand list prices (USD), Graviton m7g family, as recalled at time of writing (2026-11). Edit to your provider; the report recomputes.

| Component | Role | Peak memory | Disk | Instance | USD / month |
|---|---|---|---|---|---|
| Qdrant (v1.12.4) | vector database | 526 MiB | 0.45 GiB | m7g.medium | 29.82 |
| k-means (HeatherDB's K) | clustering job | 1615 MiB | 0.00 GiB | m7g.medium | 29.78 |
| ALS + recency (time-decayed confidence) | recommender model | 587 MiB | 0.00 GiB | m7g.medium | 29.78 |
| LOF | outlier model | 620 MiB | 0.00 GiB | m7g.medium | 29.78 |
| Gradient-boosted trees | supervised model | 327 MiB | 0.00 GiB | m7g.medium | 29.78 |
| **Specialist stack** | 5 systems + a retraining scheduler | | | | **148.96** |
| **HeatherDB** | 1 process, 5 databases | 1727 MiB | 2.04 GiB | m7g.medium | **29.95** |

HeatherDB serves all 5 capabilities from one process; its peak memory is measured with every database loaded at once. Ratio: 5.0x lower monthly infrastructure, 6 systems to 1.
