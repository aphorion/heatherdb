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

## Proven across 14 domains

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

See `sample_projects/` for full source code and READMEs.

## The fidelity signal

When you write a vector and read it back, the cosine similarity between input and output is the **fidelity** — a built-in confidence score that no other system gives you for free.

- **High fidelity** (>0.9): The memory has a strong attractor for this pattern. The data is well-represented.
- **Low fidelity** (<0.7): This pattern is novel, underrepresented, or conflicts with existing memory. It's an outlier.

This single metric enables anomaly detection (Sentinel), cold-start detection (Cinema), regime change detection (Finance), and capacity monitoring — all without any additional model.

## How it works

HeatherDB is not a key-value store or a vector search engine. It is an *associative memory* — a system that stores patterns distributed across a network of hard locations and reconstructs them from approximate queries.

**Writing:** When you write a vector, it activates the k=20 nearest hard locations, weighted by similarity. Each location accumulates a weighted sum of all vectors written to it. The memory self-organizes: locations migrate toward frequently written patterns via competitive learning, overloaded locations split (tau_overload=100), novel patterns trigger new locations (tau_split=0.3), and similar locations merge automatically.

**Reading:** Given a query vector, the system activates nearby locations and uses an energy-based Hopfield network (beta=5.0 softmax temperature) to iteratively reconstruct the best-matching stored pattern. This creates basins of attraction around stored memories — noisy queries are pulled toward the correct pattern.

**Persistence:** All writes are immediately persisted to disk via LMDB in atomic transactions. The server can crash and restart without data loss.

**Adaptive capacity:** Starts with 1,000 hard locations, grows to 2,000 max. The memory manages its own topology — no manual tuning required.

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

## Performance

Measured on Apple Silicon. Config: l_0=1000, l_max=2000, k=20.

### Throughput

| Operation | d=64 | d=128 | d=384 |
|---|---|---|---|
| Single write | 3.45 ms | 3.68 ms | 5.24 ms |
| Sustained write (per op) | — | 4.1 ms | — |
| Single read (iterative) | 55 µs | 153 µs | 644 µs |
| Reads/sec (sustained) | 14,604 | 3,479 | 945 |

### Scaling

| Metric | Result |
|---|---|
| Read latency vs. collection count | Constant (LMDB prefix isolation) |
| Write latency vs. fill level | Flat from 1000→1800 locations |
| Read latency vs. fill level | Linear with location count |
| Flush to disk (1000 writes, d=128) | 17.7 ms |
| Cold reload (1000 writes) | 2.6 ms |
| Memory footprint (2000 writes, d=128) | 4.0 MB |

### Capacity stress test

```
d=64:  1000 locs, 1.0 MB,  ~4.1 ms/write, 14,604 reads/sec
d=128: 2000 locs, 4.0 MB,  ~4.4 ms/write,  3,479 reads/sec
d=384: 2000 locs, 11.8 MB, ~5.4 ms/write,    945 reads/sec
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
      memory.rs           # Main engine (thread-safe, persistent)
      read.rs             # Hopfield iterative + single-step read
      write.rs            # 6-step adaptive write pipeline
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
  sample_projects/        # 14 demo applications
```

## Running tests

```bash
cargo test --workspace
```

## Running benchmarks

```bash
cargo bench -p heather_db                              # all benchmarks
cargo bench -p heather_db --bench write_throughput     # single category
cargo bench -p heather_db --bench capacity             # stress test
```

## Based on

*"Adaptive Elastic Associative Memory: Self-Organizing Indexing with Energy-Based Retrieval"*

## License

TBD
