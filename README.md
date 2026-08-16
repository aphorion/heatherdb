# HeatherDB

**Storing is learning. Reading is reasoning.**

HeatherDB is a content-addressable associative memory, exposed as an
ordinary HTTP database. It has one write primitive and one read primitive —
and both do more than their names suggest.

- **Write** doesn't insert a row. It's an act of *online, self-organizing
  learning*: competitive learning grows, splits, and merges a codebook of
  "hard locations" to accommodate what it's shown. There is no separate
  training step — the memory *is* the model, updated in place, on every
  write.
- **Read** doesn't fetch a row. It *reconstructs* a vector as a weighted
  superposition of everything relevant ever written — via
  softmax-over-dot-products, the same operation that sits at the center of
  three different fields at once:

  | As a... | It does this |
  |---|---|
  | **Attention head** | `softmax(β · Q·Kᵢ)` — structurally identical to transformer attention |
  | **Hopfield network** | The softmax carves out basins of attraction, so a noisy or partial query gets pulled toward the nearest genuine attractor — that's the whole "fidelity" signal below |
  | **Posterior under a mixture model** | Soft-clustering over prototype locations, weighted by how much evidence (`write_count`) each has accumulated |

A compositional vector algebra (`bind` / `unbind` / `permute` / `bundle`) —
Vector Symbolic Architecture in the Holographic Reduced Representation
tradition — sits on top, so you can bind roles to fillers, superpose
multiple bindings, and pull compositional structure back out. That's what
makes `king − man + woman ≈ queen`-style analogical retrieval possible with
no neural network in the loop.

None of this requires training data, GPUs, retraining schedules, or model
versioning. It requires an HTTP POST.

## What makes it different

**Vector databases** (Pinecone, Weaviate, Milvus) store embeddings and
return the *k*-nearest stored match — always something that was explicitly
put in. **HeatherDB** writes patterns into distributed memory and
*reconstructs* a vector from the interference between everything relevant —
which can represent things that were never individually stored. Two movies
written separately produce a taste fingerprint that neither contained
alone. Word co-occurrences written as pairs produce Word2Vec-style
analogies, without a neural network ever touching the data.

| Capability | Traditional approach | HeatherDB |
|---|---|---|
| Recommendations | Collaborative filtering, matrix factorization, retraining | Write user preferences, read back a taste fingerprint |
| Anomaly detection | Autoencoders, isolation forests, threshold tuning | Write normal patterns, measure read-back fidelity |
| Semantic similarity | Pretrained embeddings, fine-tuning | Deterministic projections + memory interference |
| Signal denoising | Kalman filters, neural denoisers | Write clean signals, query with noisy input |
| One-shot learning | Meta-learning, few-shot fine-tuning | Single write, immediate recall |
| Pattern completion | Sequence models, transformers | Partial query → full reconstruction |
| Explainability | SHAP, LIME, attention visualization | `/analyze` → activated locations + weights = "why this?" |

## Quick start

```bash
git clone https://github.com/aphorion/heather-db
cd heather-db
cargo build --release
```

```bash
HEATHER_DATA_DIR=./mydb HEATHER_DIMENSION=128 \
HEATHER_ADMIN_USER=admin HEATHER_ADMIN_PASSWORD='change-me-1234' \
  cargo run --release -p heather_server
```

The server listens on `:6380`. **Auth is HTTP Basic and on by default** —
if you don't set `HEATHER_ADMIN_USER`/`HEATHER_ADMIN_PASSWORD`, the engine
mints an `admin` user with a random password and prints it once on stderr
(`--auth-disabled` / `HEATHER_AUTH_DISABLED=1` skips auth entirely, dev
only). Data persists to disk automatically via LMDB.

### Write

```bash
curl -u admin:change-me-1234 -X POST http://localhost:6380/collections/my_collection/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[0.1, 0.3, 0.5, 0.2], [0.8, 0.1, 0.4, 0.7]]}'
```
```json
{ "count": 2 }
```

### Read (recall)

```bash
curl -u admin:change-me-1234 -X POST http://localhost:6380/collections/my_collection/read \
  -H 'Content-Type: application/json' \
  -d '{"query": [0.1, 0.29, 0.48, 0.19], "strategy": "iterative"}'
```
```json
{ "result": [0.1, 0.3, 0.5, 0.2] }
```

The query doesn't need to be exact. Noisy, partial, or approximate queries
reconstruct the closest stored pattern — content-addressable memory with
error correction built in.

## The fidelity signal

Cosine similarity between what you write and what a read reconstructs is
the **fidelity** — a confidence score no other system gives you for free.

- **High fidelity** (>0.9): a strong attractor exists for this pattern. The data is well-represented.
- **Low fidelity** (<0.7): novel, underrepresented, or conflicting with existing memory — an outlier.

One metric, several jobs: anomaly detection, cold-start detection, regime-change detection, capacity monitoring — no extra model required.

## How it works

HeatherDB is not a key-value store or a nearest-neighbor index. It stores
patterns distributed across a self-organizing network of hard locations
and reconstructs them from approximate queries.

**Writing (select → update → regulate):** *Select* activates the *k*
nearest hard locations with conscience-based winner selection. *Update*
accumulates weighted counters and migrates addresses via competitive
learning in a single pass. *Regulate* spawns new locations for unseen
regions (novelty splits), redistributes saturated ones (overload splits),
and dedups locally. Location count is a diagnostic of data complexity, not
a tunable parameter — the memory finds its own topology.

**Navigable graph search:** every write produces neighborhood knowledge as
a free byproduct — the *k* activated locations learn who they co-activated
with, building a graph that mirrors the data manifold. Reads enter at the
nearest landmark and follow neighbor edges via greedy descent instead of
scanning every location: near-constant query cost regardless of collection
size (see [Performance](#performance)). The graph isn't bolted on — it's
what competitive learning was building all along.

**Reading:** graph search finds the *k* most relevant locations; an
energy-based Hopfield network iteratively reconstructs the best-matching
pattern from them, creating basins of attraction that pull noisy queries
toward the correct memory.

**Dreaming (idle-time consolidation):** when the engine has been idle for a
configurable interval, a background pass re-presents every stored
attractor to the write rule — hippocampal replay. This sharpens attractors,
lets novelty/overload splits reorganize, and merges redundancy, all without
external input. Optional coarser passes grow cluster-level "super-attractors"
alongside the fine ones.

**Persistence:** every write is committed to LMDB in an atomic transaction. The server can crash and restart without data loss.

## Multi-tenancy and auth

One server hosts **N databases**, each with its own LMDB environment and
its own fixed vector dimension — set at creation, immutable after. To
change it, create a new database and re-write the data.

```bash
curl -u admin:pw -X POST http://localhost:6380/db -d '{"name": "movies", "dimension": 384}'
curl -u admin:pw http://localhost:6380/db
```

Every collection route exists both **unscoped** (`/collections/...`, targets
the `default` database — kept for backwards compatibility) and **scoped**
(`/db/{db}/collections/...`).

Auth is **HTTP Basic**, backed by an LMDB-resident user store — CLI
mutations (`heather user create ...`) are visible to the running engine
immediately, no restart. Scope is `Root` (everything) or `Database(name)`
(one DB only). Short-lived bearer tokens are available via `POST
/auth/token` for clients that want to skip re-hashing a password on every
request.

## Vector algebra

Energy-correct composition operators on top of the memory, exposed at
`/vec/*` (and legacy `/algebra/*`):

| Op | What it does |
|---|---|
| `bind` / `unbind` | HRR-style role↔filler binding and its inverse — the mechanism behind analogical retrieval |
| `permute` | Non-commutative binding (order matters — sequences, not just sets) |
| `bundle` | Weighted superposition of multiple vectors into one |
| `pow` | Fractional powers of a bind operator (phase-unwrapped) |
| `add` / `sub` / `scale` / `intersect` | Direct vector arithmetic |

## API reference

The full route table (see `heather_server/src/routes.rs` for exact request/response shapes — the in-repo `docs/` tree is retired pending a rewrite, this is the source of truth today):

| Category | Routes |
|---|---|
| **Collections** | `POST /collections/{name}/write` · `POST .../bulk_load` · `DELETE /collections/{name}` · `GET /collections` · `GET .../stats` · `GET .../config` · `GET .../locations` · `POST .../compress` |
| **Reads** | `POST .../read` · `POST .../attention` · `POST .../attention/mdl` (self-selects its own temperature) · `POST .../attention/calibrate` · `POST .../analyze` (full activation trace) · `POST .../batch_analyze` |
| **Documents** | `GET .../documents` · `GET .../documents/{id}` · `DELETE .../documents/{id}` · `POST .../documents/query` (similarity search over metadata-tagged docs) |
| **Identity** | `GET .../fingerprint` — a collection's emergent self-summary, see below |
| **Vector algebra** | `POST /vec/{bind,unbind,bundle,pow}`, `POST /algebra/{add,sub,scale,intersect,bind,unbind,permute}`, `POST /compose/read` |
| **Multi-tenancy** | `GET|POST /db` · `GET|DELETE /db/{db}` · every collection/algebra route above, mirrored under `/db/{db}/...` |
| **Consolidation** | `POST /db/{db}/dream` — trigger idle-time reorganization on demand |
| **Audit** | `GET /db/{db}/audit` — paginated access log (who read what, when — Root-scoped) |
| **Auth** | `POST /auth/token` · `POST /auth/token/revoke` |
| **Ops** | `GET /health` · `GET /healthz` · `GET /ready` (real LMDB read) · `GET /readyz` |

### Fingerprints — self-summarizing collections

`GET /collections/{name}/fingerprint` returns a single vector summarizing
everything ever written to a collection — its emergent identity. Computed
server-side as the write-count-weighted centroid of all hard locations,
refined through a Hopfield read to snap onto the nearest attractor. Not a
simple average — a memory-sharpened consensus.

```bash
curl -u admin:pw http://localhost:6380/collections/user_alice/fingerprint
```
```json
{ "fingerprint": [0.13, -0.07, 0.26] }
```

Enables stateless clients (one `GET` instead of client-side history
tracking), cross-application identity (fuse movie + book + music vectors
into one portable profile), drift detection (compare fingerprints over
time), collection arithmetic (`fp(alice) + fp(bob)` = blended taste), and
cold-start transfer (bootstrap a new user from their nearest existing
fingerprint).

## Configuration

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--data-dir` | `HEATHER_DATA_DIR` | *(required)* | Storage root |
| `--dimension` | `HEATHER_DIMENSION` | `128` | Default vector dimension for new DBs |
| `--port` | `HEATHER_PORT` | `6380` | Listen port |
| `--host` | `HEATHER_HOST` | `0.0.0.0` | Bind address |
| `--map-size-mb` | `HEATHER_MAP_SIZE_MB` | `256` | LMDB map size — set to ~2× expected on-disk dataset |
| `--request-timeout` | `HEATHER_REQUEST_TIMEOUT` | `30` | Seconds, for long algebra ops |
| `--max-body-size` | `HEATHER_MAX_BODY_SIZE` | `2097152` | Max request body, bytes |
| `--admin-user` / `--admin-password` | `HEATHER_ADMIN_USER` / `HEATHER_ADMIN_PASSWORD` | `admin` / *(generated)* | First-boot admin credentials |
| `--auth-disabled` | `HEATHER_AUTH_DISABLED` | `false` | Skip auth entirely — **dev only** |
| — | `RUST_LOG` | `info` | `debug` for verbose tracing |

## Operator CLI

The `heather` binary is also the admin tool — same process, `--data-dir` targets a running or stopped store directly:

```bash
heather --data-dir ./mydb user create alice --password '…' --scope root
heather --data-dir ./mydb backup create --output backup.tar.gz    # cold, engine stopped
heather --data-dir ./mydb snapshot create --db movies             # live, safe with engine running
heather --data-dir ./mydb restore restore --input backup.tar.gz
```

## Deployment

A single, statically-linked-enough Rust binary. No GPU, no accelerator, no
system dependencies beyond libc. Every install path below lands on the
same layout — `/usr/bin/heather`, `/var/lib/heatherdb`,
`/etc/heatherdb/env` — so they're interchangeable.

### Docker

```bash
docker build -t heatherdb:latest .
docker run -d --name heatherdb \
  -p 6380:6380 \
  -v heatherdb_data:/var/lib/heatherdb \
  -e HEATHER_ADMIN_USER=admin -e HEATHER_ADMIN_PASSWORD='change-me' \
  heatherdb:latest
```

Or `docker compose up -d` with the bundled `docker-compose.yml` — also the
canonical [Coolify](https://coolify.io) deploy path.

Published images (multi-arch, `linux/amd64` + `linux/arm64`, cross-compiled
natively — no QEMU-emulated builds):

```bash
docker pull ghcr.io/aphorion/heather-db:latest      # GHCR
docker pull aphorion/heatherdb:latest               # Docker Hub
```

All published images are signed with [cosign](https://github.com/sigstore/cosign) keyless:

```bash
cosign verify ghcr.io/aphorion/heather-db:latest \
  --certificate-identity-regexp "^https://github.com/aphorion/heather-db/" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### `.deb` package (Ubuntu / Debian)

```bash
curl -fsSLO https://github.com/aphorion/heather-db/releases/latest/download/heatherdb_<version>-1_amd64.deb
sudo apt install ./heatherdb_<version>-1_amd64.deb
```

Installs a `heatherdb` systemd service, a dedicated system user, and
generates admin credentials into `/etc/heatherdb/env` on first install —
read that file, then `sudo systemctl start heatherdb`. See
[`packaging/deb/`](./packaging/deb/).

### Bare-metal VPS (build from source)

```bash
curl -fsSL https://raw.githubusercontent.com/aphorion/heather-db/main/deploy/deploy | sudo bash
```

One idempotent script: system deps, Rust if missing, the `heatherdb` user,
the binary, the systemd unit (the same one the `.deb` ships), and a
`/health` smoke test. See [`deploy/README.md`](./deploy/README.md) for env
overrides.

### Kubernetes

A Helm chart ships at [`charts/heatherdb/`](./charts/heatherdb/) — a
single-replica `StatefulSet` (LMDB is single-writer, like Postgres — no
`replicaCount` knob), `volumeClaimTemplate`, headless + client `Service`s,
optional `Ingress`.

```bash
helm install heatherdb ./charts/heatherdb
```

### Reverse proxy (HTTPS)

HeatherDB speaks plain HTTP. Front it with Caddy for free auto-HTTPS:

```caddy
api.heather.example.com {
  reverse_proxy 127.0.0.1:6380
  encode zstd gzip
}
```

## Performance

Measured on Apple Silicon (M-series). Config: `l_0=1000, k=20, neighbor_cap=max((k-1)·⌈ln L⌉, 2k)`, 32 landmarks. Reproduce with `cargo bench -p heather_db`.

| Operation | d=64 | d=128 | d=384 |
|---|---|---|---|
| Single write | 3.3 ms | 3.7 ms | 4.1 ms |
| Single read (iterative) | 18 µs | 43 µs | 153 µs |
| Reads/sec (sustained) | 19,418 | 9,336 | 3,473 |

**Batch writes** amortize lock acquisition and LMDB persistence into one transaction:

| Batch size | Individual writes | Batch write | Speedup |
|---|---|---|---|
| 10 | 39.4 ms | 6.8 ms | 5.8× |
| 100 | 390.3 ms | 11.6 ms | 33.7× |
| 500 | 2,460 ms | 31.7 ms | 77.6× |

**Graph search vs. brute force** (activation latency, d=128) — near-constant regardless of collection size, because it follows the learned manifold instead of scanning every location:

| Collection size | Brute force (rayon) | Graph search | Speedup |
|---|---|---|---|
| 1,384 | 131 µs | 18 µs | 7.4× |
| 5,097 | 124 µs | 22 µs | 5.7× |

## Project structure

```
heather_db/
  heather_db/            # Core engine library
    src/
      collection.rs       # Main engine — reads, writes, attention, tracing
      write.rs             # select → update → regulate write pipeline
      read.rs               # Hopfield read + navigable graph search
      merge.rs              # Location merging / dedup
      config.rs             # EAM configuration
      db_config.rs          # Per-database persisted config (dimension, EAM knobs)
      store.rs               # LMDB persistence (registry, locations, metadata, documents, doc_index, audit)
      audit.rs                # Buffered, batched access-log storage
      hive.rs                 # One LMDB env, many collections
      server.rs                # N hives, one per database
      vec_ops.rs                # Vector math, softmax, HRR unbind
      location.rs, filter.rs, error.rs
  heather_server/         # HTTP server + CLI
    src/
      main.rs              # CLI entry, route assembly, startup
      routes.rs             # Legacy (default-DB) handlers
      routes_db.rs           # Scoped (/db/{db}/...) wrappers
      auth.rs, users.rs, tokens.rs   # HTTP Basic + Bearer auth
      audit.rs                # Audit HTTP plumbing + flush loop
      dream.rs                  # Idle-time consolidation
      backup.rs, cli.rs           # Backup/restore/snapshot + user CLI
      models.rs                    # JSON request/response types
  heather_algebra/        # Energy-correct vector algebra (bind/unbind/permute/bundle/compose)
  deploy/                 # VPS install script (deploy/deploy)
  packaging/deb/          # .deb maintainer scripts + systemd unit
  charts/heatherdb/       # Kubernetes Helm chart
```

## Related repos

- [`heatherdb-samples`](https://github.com/aphorion/heatherdb-samples) — 26 example apps (recommendation, anomaly detection, semantic analogies, denoising, and more) built against this engine.
- [`heatherdb-pi-demo`](https://github.com/aphorion/heatherdb-pi-demo) — conference/demo-day rig: recommender catalog, FastAPI proxy, Raspberry Pi cross-compile.
- [`heatherdb-landing`](https://github.com/aphorion/heatherdb-landing) — the product site.

## Development

```bash
cargo test --workspace
cargo bench -p heather_db                        # all benchmarks
cargo bench -p heather_db --bench graph_search    # one category
```

## Releases

Tagged releases publish 4 Linux targets (`x86_64`/`aarch64` × `gnu`/`musl`)
plus a `.deb` for the two glibc targets, all on
[GitHub Releases](https://github.com/aphorion/heather-db/releases) with a
combined `SHA256SUMS.txt`:

```bash
VERSION=v0.2.1
TARGET=linux-x86_64-musl
curl -fsSLO "https://github.com/aphorion/heather-db/releases/download/${VERSION}/heather-${VERSION}-${TARGET}.tar.gz"
curl -fsSLO "https://github.com/aphorion/heather-db/releases/download/${VERSION}/SHA256SUMS.txt"
sha256sum --check SHA256SUMS.txt --ignore-missing
```

## Based on

*"Adaptive Elastic Associative Memory: Self-Organizing Indexing with Energy-Based Retrieval"*

## License

Proprietary — see [`LICENSE`](./LICENSE). All rights reserved, Aphorion Labs.
