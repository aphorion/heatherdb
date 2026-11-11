# Head-to-head: one HeatherDB vs the specialist stack

A reproducible benchmark of the convergent-intelligence claim: **one**
HeatherDB process serves several enterprise capabilities at accuracy
comparable to the specialist system each would otherwise need, for less
infrastructure.

| Capability | Dataset (public) | Incumbent(s) | Primary metric |
|---|---|---|---|
| Vector search | Fashion-MNIST 60k x 784 | Qdrant v1.12.4 (Docker) | recall@10, latency |
| Pattern discovery | Fashion-MNIST, labels hidden | k-means | NMI |
| Recommendations | MovieLens 1M, leave-last-out | ALS (`implicit`), ItemKNN, popularity | NDCG@10, HR@10 |
| Freshness | MovieLens 1M | ALS refit | time from new data to servable |
| Fraud detection | Credit-card fraud (ULB), 284k txns | Isolation Forest, LOF | PR-AUC, ROC-AUC, recall@1%FPR |
| Propensity classification | UCI bank marketing, 45k | Logistic regression, gradient-boosted trees | ROC-AUC |

Each `task_*.py` docstring states the split, the metric and exactly how each
system is used.

## Fairness rules

- Same dataset bytes (SHA-256 fingerprint recorded), same split, same
  metric, same evaluation code for every system.
- HeatherDB is driven only through its public HTTP API, with authentication
  on (Bearer tokens), the way an application would use it.
- HeatherDB serves **all** capabilities from **one** server process; its
  memory is measured with every database loaded at once. Each incumbent runs
  in its own process or container and is measured alone, as it would be
  deployed.
- Incumbents use their library defaults or standard settings; no per-task
  tuning that HeatherDB doesn't also get.
- Any step done client-side rather than in the engine is labelled in the
  task docstring (currently: MDL selection of K in discovery, finding F8).

## Run

```bash
cargo build --release -p heather_server          # from the repo root
cd benchmarks/h2h
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
docker pull qdrant/qdrant:v1.12.4                # Docker must be running
.venv/bin/python run.py                          # ~20 min on an Apple M4
.venv/bin/python report.py                       # -> results/REPORT.md
```

`run.py --tasks fraud,classify` runs a subset. Datasets download on first
use into `data/` (git-ignored). Cloud prices for the cost model live in
`prices.json`; edit them for your provider and rerun `report.py`.

## Cost model

Each incumbent is deployed as its own service on the smallest instance that
holds its measured peak memory with 1.5x headroom; models that must be refit
to take in new data add the cost of daily retraining. HeatherDB is one
instance sized the same way for its peak memory with everything loaded.
The model excludes engineering and operations staff time. It does report the
number of systems to operate, which is where most of that cost comes from.
