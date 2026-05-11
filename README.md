# HeatherDB

A memory that learns. One primitive — write vectors, read vectors — replaces entire ML pipelines.

HeatherDB is an associative memory database powered by Adaptive Elastic Associative Memory. It doesn't just store vectors. It forms interference patterns across writes, reconstructs signals from noise, and surfaces relationships that were never explicitly programmed. No training. No GPUs. No model weights. Just memory physics.

## What makes it different

**Vector databases** store embeddings and return the nearest match. HeatherDB writes patterns into distributed memory and reconstructs *new* knowledge from the interference. Two movies written separately produce a taste fingerprint that neither contained alone. Word co-occurrences written as pairs produce Word2Vec-style analogies — king:queen :: man:woman — without a neural network ever touching the data.

**ML pipelines** require training data, GPUs, retraining schedules, model versioning, and teams of engineers. HeatherDB requires an HTTP POST. Write your data. Read it back. The memory does the rest.

| Capability | Traditional approach | HeatherDB |
|---|---|---|
| Recommendations | Collaborative filtering, matrix factorization, retraining | Write user preferences, read back taste fingerprint |
| Anomaly detection | Autoencoders, isolation forests, threshold tuning | Write normal patterns, measure read-back fidelity |
| Semantic similarity | Pretrained embeddings, fine-tuning | Deterministic random projections + memory interference |
| Signal denoising | Kalman filters, neural denoisers | Write clean signals, query with noisy input |
| One-shot learning | Meta-learning, few-shot fine-tuning | Single write, immediate recall |
| Pattern completion | Sequence models, transformers | Partial query → full reconstruction |
| Explainability | SHAP, LIME, attention visualization | Analyze read → activated locations + weights = "why this?" |

## Quick start

### Prerequisites

- Rust 1.80+

### Build and run

```bash
git clone <repo-url>
cd heather_db
cargo build --release
```

Start the server:

```bash
cargo run --release -p heather_server -- \
  --data-dir ./mydb \
  --dimension 128
```

The server starts on port 6380 by default. Data persists to disk automatically.

### Write vectors

```bash
curl -X POST http://localhost:6380/collections/my_collection/write \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": [
      [0.1, 0.3, 0.5, 0.2, ...],
      [0.8, 0.1, 0.4, 0.7, ...]
    ]
  }'
```

```json
{ "count": 2 }
```

### Read (recall) a pattern

```bash
curl -X POST http://localhost:6380/collections/my_collection/read \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [0.1, 0.29, 0.48, 0.19, ...],
    "strategy": "iterative"
  }'
```

```json
{ "result": [0.1, 0.3, 0.5, 0.2, ...] }
```

The query doesn't need to be exact. Noisy, partial, or approximate queries reconstruct the closest stored pattern. That's the point — HeatherDB is content-addressable memory with error correction built in.

## Proven across 15 domains

Each project uses the same two operations — write and read — against the same server. No domain-specific models, no retraining between use cases.

| Project | What it does | Dims | Key result |
|---|---|---|---|
| **Lexicon** | Distributional semantics from co-occurrence | 128 | king→queen :: man→woman analogy without neural networks |
| **Cinema** | Movie recommendations from taste fingerprints | 128 | 98.7% fingerprint fidelity, built-in confidence scores |
| **Sentinel** | Anomaly detection via fidelity drop | 384 | Novel patterns detected by low read-back similarity |
| **Navigator** | Robot danger memory, one-shot reinforcement learning | 128 | Single-write hazard avoidance, no replay buffer |
| **Chorus** | Signal denoising via superposition averaging | 128 | 76% of theoretical denoising limit |
| **Finance** | Cross-asset pattern completion | 128 | 80% directional accuracy, fidelity dips predict regime changes |
| **Cipher** | Code pattern ecosystem discovery | 384 | Surfaces co-occurring patterns, not just nearest matches |
| **Memoria** | Conversational AI with associative recall | 384 | Context retrieval from partial cues |
| **Muse** | Creative idea blending | 384 | Novel concepts from interference of stored ideas |
| **Genesis** | Artificial life simulation | 32 | Emergent behavior from shared memory pressure |
| **Sieve** | Near-duplicate detection | 384 | Fuzzy matching without exact hashing |
| **Swarm** | Multi-agent emergent intelligence | 384 | Shared EAM as collective memory |
| **Whisper** | Lossy semantic compression | 384 | Meaning-preserving dimensionality reduction |
| **Oracle** | Diagnostic pattern completion | 64 | Partial symptom → full diagnosis reconstruction |
| **Cortex** | Thinking notebook (Next.js + FastAPI) | 384 | Web app comparing EAM vs vector search |
| **Emergence** | Zero-shot capability via memory composition | 384 | Coder+Writer→Documentation without training |

Source code + per-project READMEs live in the standalone
[`heatherdb-samples`](https://github.com/aphorion/heatherdb-samples) repo
(formerly `sample_projects/` here).

## The fidelity signal

When you write a vector and read it back, the cosine similarity between input and output is the **fidelity** — a built-in confidence score that no other system gives you for free.

- **High fidelity** (>0.9): The memory has a strong attractor for this pattern. The data is well-represented.
- **Low fidelity** (<0.7): This pattern is novel, underrepresented, or conflicts with existing memory. It's an outlier.

This single metric enables anomaly detection (Sentinel), cold-start detection (Cinema), regime change detection (Finance), and capacity monitoring — all without any additional model.

## How it works

HeatherDB is not a key-value store or a vector search engine. It is an *associative memory* — a system that stores patterns distributed across a network of hard locations and reconstructs them from approximate queries.

**Writing (three phases):** When you write a vector, the write cycle runs in three phases. *Select:* activate the k=20 nearest hard locations with conscience-based winner selection. *Update:* accumulate weighted counters and migrate addresses via competitive learning in a single pass. *Regulate:* novelty splits spawn new locations for unseen regions (tau_split=0.3), overload splits distribute saturated locations (tau_overload=100), and local dedup cleans up. The memory self-organizes its own topology — location count is a diagnostic of data complexity, not a tunable parameter.

**Navigable graph search:** Every write produces neighborhood knowledge as a free byproduct. The k activated locations learn who they co-activated with, building a navigable graph that mirrors the data manifold. At query time, instead of comparing against all L locations (O(LD)), the system enters the graph at the nearest landmark and follows neighbor edges via greedy descent — one batched matrix-vector multiply per hop. Query cost: ~17-22 µs at d=128 regardless of collection size, vs linear growth for brute force. Neighbor capacity adapts to the memory: nb_max = max((k-1)·⌈ln L⌉, 2k) — scaling with collection size for O(log L) global navigability. The graph isn't bolted on — it's what competitive learning was building all along.

**Reading:** Given a query vector, graph search finds the k most relevant locations on the learned manifold. An energy-based Hopfield network (beta=5.0 softmax temperature) iteratively reconstructs the best-matching stored pattern from these locations. This creates basins of attraction around stored memories — noisy queries are pulled toward the correct pattern.

**Persistence:** All writes are immediately persisted to disk via LMDB in atomic transactions. The server can crash and restart without data loss.

**Self-regulating capacity:** Starts with 1,000 hard locations. Splits are self-limiting (novelty splits consume the void that caused them, overload splits distribute the load that triggered them). Merges consolidate redundancy. No maximum location cap — the system finds its own equilibrium.

## Dimension guidelines

| Dimensions | Character | Best for |
|---|---|---|
| 32 | Very tight, high interference | Compact simulations (Genesis) — needs careful tuning |
| 64 | Moderate compression | Diagnostic systems (Oracle) |
| 128 | Balanced interference | Recommendations, semantics, finance (Cinema, Lexicon, Navigator) |
| 384 | Spacious, high fidelity | Rich content domains (Memoria, Sentinel, Cipher) |

Lower dimensions create more interference between stored patterns — useful when you *want* patterns to blend (recommendations, denoising). Higher dimensions preserve individual patterns more faithfully — useful for precise recall (anomaly detection, duplicate detection).

## Fingerprints — self-summarizing collections

Every collection knows what it contains. `GET /collections/{name}/fingerprint` returns a single vector that summarizes everything ever written — the collection's emergent identity.

Under the hood: compute the write-count-weighted centroid of all hard location patterns, then refine it through Hopfield iterative read to snap to the nearest attractor in memory space. The result is not a simple average — it's a memory-sharpened consensus.

```bash
curl http://localhost:6380/collections/user_alice/fingerprint
```

```json
{ "fingerprint": [0.13, -0.07, 0.26, ...] }
```

### What this enables

| Capability | How |
|---|---|
| **Stateless clients** | No need to track history or compute centroids client-side. One GET and you have the full taste/identity. |
| **Cross-application identity** | Write movie, book, and music vectors to one collection. The fingerprint fuses them — one portable profile across apps. |
| **Drift detection** | Compare `fingerprint(t0)` vs `fingerprint(t1)`. A drop in cosine similarity = taste shift or regime change. |
| **Collection arithmetic** | `fp(alice) + fp(bob)` = blended taste. `fp(scifi) - fp(comedy)` = what separates them. Cluster users by fingerprint for automatic cohorts. |
| **Cold-start transfer** | New user, 2 data points? Find the nearest existing fingerprint and bootstrap from that neighbor. |
| **Multi-modal fusion** | Mix image embeddings, text embeddings, audio features in one collection. The fingerprint summarizes all modalities in superposition. |

### Benchmarked (Cinema project, 128d, 5 movies)

| Metric | Client-side centroid + read | Server-side fingerprint |
|---|---|---|
| Latency | 2.3 ms | **0.7 ms** (3.3x faster) |
| Recommendation ranking | baseline | Spearman ρ = 1.0 (identical) |
| Leave-one-out mean rank | 82.2 | **49.2** (lower = better) |
| Hit@20 rate | 20% | **40%** |

Same or better quality, zero client state, one API call.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/collections/{name}/write` | POST | Store one or more vectors in a collection |
| `/collections/{name}/read` | POST | Recall a pattern from a query vector |
| `/collections/{name}/fingerprint` | GET | Get the collection's emergent identity vector |
| `/collections/{name}/analyze` | POST | Trace a read — returns activated locations, weights, iterations, convergence |
| `/collections/{name}/batch_analyze` | POST | Batch trace — analyze multiple queries in a single call |
| `/collections/{name}/documents` | GET | List all documents with metadata |
| `/collections/{name}/documents/{id}` | GET | Get a single document by ID |
| `/collections/{name}/documents/query` | POST | Similarity search over documents |
| `/collections` | GET | List all collections |
| `/health` | GET | Server liveness check |

### POST /collections/{name}/write

**Request body:**
```json
{
  "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "metadata": [{"title": "Doc A"}, {"title": "Doc B"}]
}
```

`metadata` is optional. When provided, each entry is a JSON value paired with the corresponding vector. The response includes document IDs:

**Response (without metadata):** `{ "count": 2 }`

**Response (with metadata):** `{ "count": 2, "ids": [0, 1] }`

Documents persist alongside the EAM — the vector gets distributed across hard locations as usual, while the metadata is stored as a sidecar keyed by document ID. This lets you attach identity (names, labels, structured data) to the intelligence (vector patterns) without affecting EAM math.

Collections are created automatically on first write.

### POST /collections/{name}/read

**Request body:**
```json
{
  "query": [0.1, 0.2, ...],
  "strategy": "iterative"
}
```

`strategy` is optional. Values: `"iterative"` (default, best quality) or `"fast"` (single-step, lower latency).

**Response:** `{ "result": [0.1, 0.2, ...] }`

### POST /collections/{name}/analyze

**Request body:**
```json
{
  "query": [0.1, 0.2, ...],
  "strategy": "iterative"
}
```

**Response:**
```json
{
  "iterations": 3,
  "converged": true,
  "activated_locations": [
    { "id": 42, "similarity": 0.87, "weight": 0.31 },
    { "id": 78, "similarity": 0.82, "weight": 0.24 }
  ],
  "result": [0.1, 0.2, ...]
}
```

Like `/read`, but returns the full activation trace — which hard locations fired, their similarity to the query, their Hopfield softmax weights, how many iterations it took, and whether it converged. This enables explainability: trace which stored patterns contributed to a recall, build "why this?" explanations, or monitor convergence behavior.

### POST /collections/{name}/batch_analyze

**Request body:**
```json
{
  "queries": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "strategy": "iterative"
}
```

**Response:**
```json
{
  "results": [
    { "iterations": 3, "converged": true, "activated_locations": [...], "result": [...] },
    { "iterations": 2, "converged": true, "activated_locations": [...], "result": [...] }
  ]
}
```

Analyzes multiple queries in a single call, avoiding per-query network round trips. Each result has the same structure as `/analyze`. All queries run sequentially on the same thread with the collection lock held once — significantly faster than N individual calls.

### GET /collections/{name}/fingerprint

**Response:** `{ "fingerprint": [0.13, -0.07, ...] }` or `{ "fingerprint": null }` if the collection is empty.

Returns the collection's self-summary — a single vector representing the weighted consensus of everything written. Computed server-side via weighted centroid + Hopfield refinement.

### GET /collections/{name}/documents

**Response:**
```json
{
  "documents": [
    { "id": 0, "metadata": { "title": "Doc A" } },
    { "id": 1, "metadata": { "title": "Doc B" } }
  ]
}
```

Returns all documents stored in a collection with their metadata.

### GET /collections/{name}/documents/{id}

**Response:** `{ "id": 0, "metadata": { "title": "Doc A" } }`

### POST /collections/{name}/documents/query

**Request body:**
```json
{
  "query": [0.1, 0.2, ...],
  "n": 10
}
```

**Response:**
```json
{
  "results": [
    { "id": 0, "similarity": 0.92, "metadata": { "title": "Doc A" } },
    { "id": 1, "similarity": 0.78, "metadata": { "title": "Doc B" } }
  ]
}
```

Similarity search over documents. Returns the top-n documents ranked by cosine similarity to the query vector, along with their metadata. Uses the EAM hard-location posting list index internally — documents are indexed by which hard locations they activated during write, so query time scales with the number of candidates rather than total documents.

Combine with `/fingerprint` to get "recommend from this collection" in two calls:
1. `GET /collections/{name}/fingerprint` → query vector
2. `POST /collections/{name}/documents/query` with the fingerprint → ranked results with metadata

### GET /collections

**Response:** `{ "collections": ["users", "movies", ...] }`

### GET /health

**Response:** `{ "status": "ok" }`

All errors return `{ "error": "<message>" }` with an appropriate HTTP status code.

## Configuration

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--data-dir` | `HEATHER_DATA_DIR` | *(required)* | Directory for database storage |
| `--dimension` | `HEATHER_DIMENSION` | *(required)* | Vector dimension |
| `--port` | `HEATHER_PORT` | `6380` | Listen port |
| `--host` | `HEATHER_HOST` | `0.0.0.0` | Bind address |
| `--map-size-mb` | `HEATHER_MAP_SIZE_MB` | `4096` | LMDB map size — set to ~2× expected on-disk dataset |
| `--request-timeout` | `HEATHER_REQUEST_TIMEOUT` | `600` | Seconds, for long algebra ops |
| — | `RUST_LOG` | `info` | `debug` for verbose tracing |

## Deployment

HeatherDB is a single statically-linked Rust binary. It needs no system
dependencies, no GPU, and no accelerator. Three production paths, simplest
first.

### Path A — Docker (recommended)

A multi-stage `Dockerfile` ships in this repo. Final image is ~25 MB on
`debian:bookworm-slim`, runs as a non-root user, exposes `6380`, has a
healthcheck, and persists data in a named volume.

```bash
# Build
docker build -t heatherdb:latest .

# Run (named volume keeps your data across restarts)
docker run -d --name heatherdb \
  -p 6380:6380 \
  -v heatherdb_data:/var/lib/heatherdb \
  -e HEATHER_DIMENSION=128 \
  heatherdb:latest

# Smoke test
curl http://127.0.0.1:6380/health
```

Or with the bundled Compose file — brings up **engine + Fovea** (the
operator GUI) in one shot:

```bash
docker compose up -d        # build + run both
docker compose logs -f      # tail
docker compose down         # stop (data persists in the volume)
```

After it's up:

| URL                    | What's there |
|------------------------|--------------|
| http://localhost:6380  | HeatherDB engine HTTP API |
| http://localhost:8080  | **Fovea** — open this; it talks to the engine over a Caddy reverse-proxy so there's no CORS to configure. |

Override config via env in `docker-compose.yml` or with `-e` on `docker run`.
Common overrides:

```bash
-e HEATHER_DIMENSION=384 \
-e HEATHER_MAP_SIZE_MB=8192 \
-e RUST_LOG=debug
```

### Path B — Bare metal / $5 VPS (Hetzner / DO / Linode)

SSH into a fresh VPS and run **one** command. The script ([`deploy/deploy`](./deploy/deploy))
installs system deps, Rust if missing, the `heatherdb` user, the binary, the
systemd unit, and smoke-tests `/health`. Idempotent — re-run any time to update.

```bash
# Option 1 — clone first (if you want to inspect or pin a ref)
git clone https://github.com/aphorion/heather-db
cd heather-db
sudo ./deploy/deploy

# Option 2 — one-liner from the internet
curl -fsSL https://raw.githubusercontent.com/aphorion/heather-db/main/deploy/deploy | sudo bash
```

Common env overrides:

```bash
HEATHER_VERSION=v0.1.0 HEATHER_DIM=384 \
  sudo -E ./deploy/deploy
```

See [`deploy/README.md`](./deploy/README.md) for the full env-var table and
post-install operational notes.

### Path C — Raspberry Pi 5 (cross-compiled)

Production-grade scripts (cross, snapshot bake, rsync deploy, smoke test) live
in the [`heatherdb-pi-demo`](https://github.com/aphorion/heatherdb-pi-demo) repo
under `deploy/`. Quick version:

```bash
# In heatherdb-pi-demo/
./deploy/cross-compile.sh                                   # builds aarch64 binary
./deploy/bake_snapshot.sh                                   # warm LMDB snapshot
./deploy/deploy.sh pi@heatherdb-pi.local --with-snapshot    # rsync + restart
./deploy/smoke-test.sh pi@heatherdb-pi.local
```

### Reverse proxy (HTTPS)

HeatherDB speaks plain HTTP. Front it with Caddy for free auto-HTTPS:

```caddy
api.heather.example.com {
  reverse_proxy 127.0.0.1:6380
  encode zstd gzip
}
```

### Operational notes

| Concern | What to do |
|---------|------------|
| **Backups**     | The data dir is a single LMDB env. `tar -czf` it while the service is stopped, or use `mdb_copy` for a live copy. |
| **Memory**      | ~1.5× the on-disk LMDB size at steady state. |
| **Sizing**      | A Pi 5 (8 GB) comfortably runs the 25M MovieLens demo (≈ 187k attractors). For larger workloads, scale `HEATHER_MAP_SIZE_MB` up before first write. |
| **Updates**     | Replace the binary, `systemctl restart heatherdb`. Restart is < 1 s — LMDB hydrates lazily. |
| **Monitoring**  | Scrape `/health` and `/stats`. The `/analyze` endpoint is your built-in explainability — no extra tooling required. |

## Performance

Measured on Apple Silicon (M-series). Config: l_0=1000, k=20, neighbor_cap=max((k-1)·⌈ln L⌉, 2k), 32 landmarks.

### Throughput

| Operation | d=64 | d=128 | d=384 |
|---|---|---|---|
| Single write | 3.3 ms | 3.7 ms | 4.1 ms |
| Warmed write (post-200) | — | 4.6 ms | 4.8 ms |
| Single read (iterative) | 18 µs | 43 µs | 153 µs |
| Single read (single-step) | — | 20 µs | — |
| Reads/sec (sustained) | 19,418 | 9,336 | 3,473 |

### Batch writes

Writes use graph-activated search (same graph as reads), dot-product-only similarity (addresses are unit-normalized), and struct-of-arrays address layout for cache-friendly brute-force fallback. Batch writes amortize lock acquisition and LMDB persistence into a single transaction.

| Batch size | Individual writes | Batch write | Speedup | Per-write cost |
|---|---|---|---|---|
| 10 | 39.4 ms | **6.8 ms** | **5.8x** | 0.68 ms |
| 50 | 192.9 ms | **11.4 ms** | **16.9x** | 0.23 ms |
| 100 | 390.3 ms | **11.6 ms** | **33.7x** | 0.12 ms |
| 500 | 2,460 ms | **31.7 ms** | **77.6x** | 0.06 ms |

At batch=500, per-write cost drops to 63 microseconds. The LMDB transaction overhead (~3ms per individual write) is amortized across the entire batch.

### Graph search vs. brute force

Activation latency (d=128, the step that finds relevant locations):

| Collection size (L) | Brute force (sequential) | Brute force (rayon) | Graph search | Graph vs. rayon |
|---|---|---|---|---|
| 1,384 | 143 µs | 131 µs | **18 µs** | **7.4x faster** |
| 1,696 | 178 µs | 64 µs | **19 µs** | **3.4x faster** |
| 2,251 | 237 µs | 76 µs | **19 µs** | **3.9x faster** |
| 3,607 | 402 µs | 102 µs | **19 µs** | **5.3x faster** |
| 5,097 | 538 µs | 124 µs | **22 µs** | **5.7x faster** |

Graph search is near-constant (~17-22 µs) regardless of L. It beats multi-threaded brute force at every data point because it follows the learned manifold structure instead of scanning all locations.

Across dimensions (5000 writes):

| Dimension | Brute force (rayon) | Graph search | Speedup |
|---|---|---|---|
| 64 | 36 µs | **8 µs** | **4.3x** |
| 128 | 98 µs | **18 µs** | **5.4x** |
| 256 | 312 µs | **49 µs** | **6.3x** |
| 384 | 475 µs | **90 µs** | **5.3x** |

### Scaling

| Metric | Result |
|---|---|
| Read latency vs. collection count | Constant (LMDB prefix isolation) |
| Write latency vs. fill level | Flat from 1000→1800 locations |
| Read latency vs. fill level | Near-constant (graph search) |
| Flush to disk (1000 writes, d=128) | 7.6 ms |
| Memory footprint (2000 writes, d=128) | 4.5 MB |

### Capacity stress test

```
d=64:  1002 locs, 1.0 MB,  ~3.3 ms/write, 19,418 reads/sec
d=128: 2284 locs, 4.5 MB,  ~3.7 ms/write,  9,336 reads/sec
d=384: 3000 locs, 17.6 MB, ~4.1 ms/write,  3,473 reads/sec
```

Sub-millisecond reads. Single-digit millisecond writes. Megabytes, not gigabytes. Runs on a Raspberry Pi.

## Project structure

```
heather_db/
  Cargo.toml              # Workspace root
  heather_db/             # Core library
    src/
      lib.rs              # Public API
      config.rs           # EAM configuration
      collection.rs       # Main engine (thread-safe, persistent, graph-accelerated)
      read.rs             # Hopfield read + navigable graph search
      write.rs            # Three-phase adaptive write pipeline (select → update → regulate)
      merge.rs            # KNN location merging
      store.rs            # LMDB persistence layer (5 databases: registry, locations, metadata, documents, doc_index)
      location.rs         # Hard location data model
      vec_ops.rs          # Vector math utilities
      error.rs            # Error types
  heather_server/         # HTTP server
    src/
      main.rs             # CLI, startup, configuration
      routes.rs           # Request handlers
      models.rs           # JSON request/response types
```

## Inside this repo

- **`heather_db/`** · **`heather_server/`** · **`heather_algebra/`** · **`heather_fornix/`** — the engine (Rust workspace).
- **`fovea/`** — the operator GUI (Tauri 2 + React 19). Run `npm run tauri:dev` for the desktop app, or `docker compose up` to get the web build alongside the engine. See [fovea/README.md](./fovea/README.md).
- **`deploy/`** — `deploy/deploy` script for in-VPS deploys + the systemd unit.
- **`docs/`** — RFCs and architecture notes.

## Related repos

- [`heatherdb-samples`](https://github.com/aphorion/heatherdb-samples) — 26
  example apps (Memoria, Cinema2, Lexicon, Sentinel, Cortex, Emergence, …).
- [`heatherdb-pi-demo`](https://github.com/aphorion/heatherdb-pi-demo) —
  conference / demo-day rig: recommender catalog, FastAPI proxy, Pi cross-compile.
- [`heatherdb-landing`](https://github.com/aphorion/heatherdb-landing) — the
  product website at heather.aphorion.co.

## Running tests

```bash
cargo test --workspace
```

## Running benchmarks

```bash
cargo bench -p heather_db                              # all benchmarks
cargo bench -p heather_db --bench write_throughput     # single category
cargo bench -p heather_db --bench capacity             # stress test
cargo bench -p heather_db --bench graph_search         # graph vs flat activation
```

## Based on

*"Adaptive Elastic Associative Memory: Self-Organizing Indexing with Energy-Based Retrieval"*

## Releases

Pre-built binaries for every tagged release land on
[GitHub Releases](https://github.com/aphorion/heather-db/releases).
Eight targets per release:

```
linux-x86_64-gnu       linux-x86_64-musl       (Linux glibc / Alpine-static)
linux-aarch64-gnu      linux-aarch64-musl      (Linux arm64, glibc / static)
macos-x86_64           macos-aarch64           (Intel / Apple Silicon)
windows-x86_64         freebsd-x86_64          (MSVC / FreeBSD 14)
```

Each `.tar.gz`/`.zip` ships next to a combined `SHA256SUMS.txt`:

```bash
VERSION=v0.1.0
TARGET=linux-x86_64-musl
curl -fsSLO "https://github.com/aphorion/heather-db/releases/download/${VERSION}/heather_server-${VERSION}-${TARGET}.tar.gz"
curl -fsSLO "https://github.com/aphorion/heather-db/releases/download/${VERSION}/SHA256SUMS.txt"
sha256sum --check SHA256SUMS.txt --ignore-missing
tar -xzf "heather_server-${VERSION}-${TARGET}.tar.gz"
./heather_server-${VERSION}-${TARGET}/heather_server --help
```

Container images are published to GHCR on every tag and on `main`:

```bash
# Latest stable
docker pull ghcr.io/aphorion/heather-db:latest

# A specific tag
docker pull ghcr.io/aphorion/heather-db:v0.1.0

# Bleeding edge (auto-built from main)
docker pull ghcr.io/aphorion/heather-db:edge
```

All published images are signed with [cosign](https://github.com/sigstore/cosign)
keyless. Verify:

```bash
cosign verify ghcr.io/aphorion/heather-db:v0.1.0 \
  --certificate-identity-regexp "^https://github.com/aphorion/heather-db/" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

## License

TBD
