# Sentinel — Anomaly Detection with Reconstruction Diff

Detects anomalous log entries, events, or records by measuring how well HeatherDB's EAM can reconstruct them. The key differentiator: Sentinel doesn't just flag anomalies — it shows you **what normal looks like** for that input by surfacing the EAM's reconstruction.

## Why this is different from other anomaly detection

| Approach | Detects anomaly? | Explains why? |
|----------|------------------|---------------|
| Threshold alerting | Yes (if you pick the right threshold) | No |
| Isolation forest | Yes | No — just an anomaly score |
| Vector DB (cosine distance) | Yes (nearest neighbor distance) | Shows nearest match, not expected pattern |
| **Sentinel (EAM reconstruction)** | **Yes** | **Shows what the EAM expected — the reconstructed "normal" pattern** |

The EAM doesn't just find the nearest stored record. It **reconstructs** what it thinks your input should look like, based on the interference of all stored patterns. The gap between your input and the reconstruction reveals exactly what's unusual.

```
INPUT:    POST /api/admin/users 403 from 185.220.101.42 — 847 requests in 60s
FIDELITY: 0.31 — ANOMALY

EXPECTED: GET /api/users 200 from 10.0.1.15 — 8 requests in 60s
          ^^^             ^^^      ^^^^^^^^^^   ^
          method          status   internal IP  normal rate

DEVIATION: POST→GET, /admin/ path, 403→200, external IP, 100x request rate
```

## Quick start

```bash
# Start HeatherDB
cargo run --release -p heather_server -- \
  --data-dir /tmp/sentinel_db \
  --dimension 384 \
  --port 6380

# Setup
cd sample_projects/sentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Train on normal patterns

```bash
python run.py train sample_data/normal_logs.txt
```

### Monitor a log file

```bash
python run.py monitor sample_data/mixed_logs.txt
```

Output:
```
  ✓ GET /api/users 200 from 10.0.1.15 — 5 requests in 60s (0.78)
  ✓ GET /api/products 200 from 10.0.1.30 — 9 requests in 60s (0.74)
  ☠ ANOMALY (0.29) POST /api/admin/users 403 from 185.220.101.42 — 847 reque
    → expected: POST /api/users 201 from 10.0.1.15 — 1 requests in 60s
  ✓ GET /api/orders/100 200 from 10.0.1.22 — 2 requests in 60s (0.72)
  ☠ ANOMALY (0.33) POST /api/auth/login 401 from 91.240.118.172 — 312 reques
    → expected: POST /api/auth/login 200 from 10.0.1.40 — 3 requests in 60s
  ...

============================================================
  Monitor summary:  20 entries analyzed
    ✓ Normal:     12
    ⚠ Suspicious: 2
    ☠ Anomalies:  6
============================================================
```

### Check individual entries

```bash
python run.py check "GET /etc/passwd 404 from 45.33.32.156 — 23 requests in 60s"
python run.py check "GET /api/users 200 from 10.0.1.15 — 5 requests in 60s"
```

### Interactive mode

```bash
python run.py interactive
```

## Commands

| Command | Description |
|---------|-------------|
| `python run.py train <file>` | Train on normal patterns (one per line) |
| `python run.py check "text"` | Check one entry |
| `python run.py monitor <file>` | Analyze a file of entries |
| `python run.py stats` | Show statistics |
| `python run.py interactive` | Interactive mode |

## How fidelity scoring works

| Fidelity | Level | Meaning |
|----------|-------|---------|
| >= 0.60 | NORMAL | SDM reconstructs it well — matches known patterns |
| 0.45 - 0.60 | SUSPICIOUS | Partial match — similar to known patterns but off |
| < 0.45 | ANOMALY | EAM can't reconstruct — nothing like this in baseline |

## Sample data

- `sample_data/normal_logs.txt` — 32 normal HTTP log entries (internal IPs, 200/201 status, normal request rates)
- `sample_data/mixed_logs.txt` — 20 entries mixing normal traffic with attacks (brute force login, path traversal, admin probing, vulnerability scanning)

## Project structure

```
sentinel/
  .env                          # HEATHER_URL
  requirements.txt              # Dependencies (no Claude API needed)
  sentinel/
    __init__.py
    client.py                   # HeatherDB HTTP client
    embeddings.py               # sentence-transformers + SQLite cache
    detector.py                 # Core: Sentinel with train/check/monitor
    cli.py                      # CLI subcommands
  run.py                        # Entry point
  sample_data/
    normal_logs.txt             # Training data (normal patterns)
    mixed_logs.txt              # Test data (normal + anomalies)
```
