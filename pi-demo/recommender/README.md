# pi-demo / recommender

Spike for Act 2 of the conference Pi demo: MovieLens-scale recommender on
HeatherDB, no GPU. See `SPIKE_RESULTS.md` for the verdict.

## Files

- `encode.py` — deterministic 128d movie encoder (genre + decade + title tokens, SHA-256 hashed). No ML deps.
- `build.py` — builds two collections: `movies_a` (shape A) and `users_b` (shape B). Holds out 1000 random users.
- `bench.py` — Hit@K + latency over held-out users, vs raw cosine baseline.
- `out/` — generated artifacts (movie vectors, indices, held-out set). Gitignored.
- `data/ml-1m/` — MovieLens 1M raw files. Gitignored.

## Run

```bash
# 1. Download MovieLens 1M (already done in data/ml-1m/)

# 2. Build heather_server (release)
cargo build --release -p heather_server

# 3. Start server
HEATHER_DATA_DIR=pi-demo/recommender/heather_data \
HEATHER_DIMENSION=128 \
HEATHER_MAP_SIZE_MB=2048 \
./target/release/heather_server &

# 4. Encode movies + build EAM
cd pi-demo/recommender
pip install -r requirements.txt
python encode.py
python build.py

# 5. Benchmark
python bench.py --n 500 --k 10 --strategy fast
```

## Reset

```bash
rm -rf pi-demo/recommender/heather_data pi-demo/recommender/out
```
