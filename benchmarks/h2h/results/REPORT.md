# Head-to-head: one HeatherDB vs the specialist stack

Run 2026-11-11T10:05:06 to 2026-11-11T10:22:16 on macOS-26.5.1-arm64-arm-64bit. Engine `heather 1.0.0` at `v1.0.0-8-g4a32470-dirty (feat/selective-attention)`.

Results from the 10:05 suite run plus recommendations (learned item symbols) and access control measured separately on feat/learned-symbols.

## Accuracy, same data, same task

| Capability | Primary metric | HeatherDB | Best incumbent | Ratio | Verdict |
|---|---|---|---|---|---|
| Vector search | recall@10 | 1.000 | 0.998 (Qdrant (v1.12.4)) | 1.00 | parity or better |
| Pattern discovery / segmentation | nmi | 0.531 | 0.506 (k-means (HeatherDB's K)) | 1.05 | parity or better |
| Recommendations | ndcg@10 | 0.066 | 0.062 (ALS + recency (time-decayed confidence)) | 1.06 | parity or better |
| Fraud / anomaly detection | pr_auc | 0.582 | 0.582 (LOF (k=50)) | 1.00 | parity or better |
| Propensity classification | roc_auc | 0.931 | 0.935 (Gradient-boosted trees) | 1.00 | parity or better |
| Access control | roc_auc | 0.870 | 0.868 (Logistic regression (one-hot)) | 1.00 | parity or better |

### Vector search

| System | recall@10 | build_s | p50_ms | p95_ms | qps_single_client | server_cpu_s |
|---|---|---|---|---|---|---|
| HeatherDB | 1.000 | 2.440 | 2.093 | 4.831 | 311.246 | 8.031 |
| Qdrant (v1.12.4) | 0.998 | 17.009 | 3.739 | 5.127 | 264.255 | - |

### Pattern discovery / segmentation

| System | k | codebook_locations | fit_s | assign_s | nmi | ari | purity | server_cpu_s |
|---|---|---|---|---|---|---|---|---|
| HeatherDB (codebook + MDL) | 80 | 6378 | 22.269 | 4.380 | 0.531 | 0.315 | 0.657 | 119.351 |
| k-means (HeatherDB's K) | 80 | - | 0.911 | 0.015 | 0.506 | 0.139 | 0.746 | - |
| k-means (true K=10, oracle) | 10 | - | 0.370 | 0.015 | 0.561 | 0.371 | 0.558 | - |

At the true K=10 (an oracle choice), HeatherDB's codebook scores NMI 0.588 / ARI 0.423 / purity 0.584.

### Recommendations

| System | hr@10 | ndcg@10 | load_s | serve_ms_per_user | time_updates | peak_rss_mb_alone |
|---|---|---|---|---|---|---|
| HeatherDB (learned item symbols) | 0.120 | 0.066 | 285.015 | 2.397 | 982089 | 389 |
| HeatherDB (selective attention) | 0.105 | 0.053 | 312.770 | 0.999 | 315453 | - |
| HeatherDB (no selective attention) | 0.051 | 0.025 | 39.201 | 0.510 | - | - |
| Raw sum (same vectors, no memory) | 0.048 | 0.023 | - | - | - | - |
| Popularity | 0.029 | 0.016 | - | - | - | - |
| ItemKNN (cosine) | 0.061 | 0.032 | - | - | - | - |
| ALS (implicit, 64 factors) | 0.075 | 0.039 | - | - | - | - |
| ItemKNN + recency | 0.076 | 0.040 | - | - | - | - |
| ALS + recency (time-decayed confidence) | 0.123 | 0.062 | - | - | - | - |

Freshness: a new user's history is servable in 16 ms (p50) / 39 ms (p95) on HeatherDB, against a 2.8 s ALS refit (176x).

### Fraud / anomaly detection

| System | roc_auc | pr_auc | recall@1%fpr | fit_s | score_us_per_txn | server_cpu_s |
|---|---|---|---|---|---|---|
| HeatherDB (novelty) | 0.953 | 0.582 | 0.805 | 0.574 | 396.408 | 213.169 |
| HeatherDB (nearest distance) | 0.953 | 0.199 | 0.711 | 0.189 | 227.956 | 139.263 |
| Isolation Forest | 0.949 | 0.122 | 0.577 | 0.590 | 19.231 | - |
| LOF | 0.951 | 0.482 | 0.785 | 2.664 | 58.327 | - |
| LOF (k=50) | 0.953 | 0.582 | 0.805 | 3.283 | 80.439 | - |

### Propensity classification

| System | roc_auc | pr_auc | fit_s | score_us_per_row | server_cpu_s |
|---|---|---|---|---|---|
| HeatherDB (selective attention) | 0.931 | 0.605 | 74.487 | 4346.700 | 162.661 |
| HeatherDB (no selective attention) | 0.867 | 0.480 | 0.568 | 910.665 | 10.890 |
| Logistic regression | 0.910 | 0.569 | 0.019 | 0.030 | - |
| Gradient-boosted trees | 0.935 | 0.629 | 0.419 | 0.956 | - |

### Access control

| System | roc_auc | fit_s | score_us_per_row |
|---|---|---|---|
| HeatherDB (learned symbols) | 0.870 | 19.189 | 1829.643 |
| Logistic regression (one-hot) | 0.868 | 0.092 | 2.057 |
| Gradient-boosted trees | 0.831 | 0.396 | 2.904 |

## Infrastructure: everything above, deployed

Each incumbent is deployed as its own service on the smallest instance holding its measured peak memory x1.5. Models that must be refit to take new data add 30 retrain runs a month. Prices: AWS us-east-1 on-demand list prices (USD), Graviton m7g family, as recalled at time of writing (2026-11). Edit to your provider; the report recomputes.

| Component | Role | Peak memory | Disk | Instance | USD / month |
|---|---|---|---|---|---|
| Qdrant (v1.12.4) | vector database | 610 MiB | 0.45 GiB | m7g.medium | 29.82 |
| k-means (HeatherDB's K) | clustering job | 1599 MiB | 0.00 GiB | m7g.medium | 29.78 |
| ALS + recency (time-decayed confidence) | recommender model | 561 MiB | 0.00 GiB | m7g.medium | 29.78 |
| LOF (k=50) | outlier model | 516 MiB | 0.00 GiB | m7g.medium | 29.79 |
| Gradient-boosted trees | supervised model | 325 MiB | 0.00 GiB | m7g.medium | 29.78 |
| Logistic regression (one-hot) | supervised model on ids | 220 MiB | 0.00 GiB | m7g.medium | 29.78 |
| **Specialist stack** | 6 systems + a retraining scheduler | | | | **178.74** |
| **HeatherDB** | 1 process, 6 databases | 2184 MiB | 2.30 GiB | m7g.medium | **29.97** |

HeatherDB serves all 6 capabilities from one process; its peak memory is measured with every database loaded at once. Ratio: 6.0x lower monthly infrastructure, 7 systems to 1.

HeatherDB peak memory, estimated in part: peak = the 5-database run measured together + recommendations-symbols run alone (389 MiB) + access (64 MiB, est.).
