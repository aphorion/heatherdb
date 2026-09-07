# HeatherDB

**Storing is learning. Reading is reasoning.**

HeatherDB is a content-addressable associative memory, exposed as an ordinary
HTTP database. It has one write primitive and one read primitive — and both do
more than their names suggest.

- **Write** doesn't insert a row. It's an act of *online, self-organizing
  learning*: competitive learning grows, splits, and merges a codebook of
  "hard locations" to accommodate what it's shown. There is no separate
  training step — the memory *is* the model, updated in place, on every write.
- **Read** doesn't fetch a row. It *reconstructs* a vector as a weighted
  superposition of everything relevant ever written — via
  softmax-over-dot-products, the same operation that sits at the centre of
  three different fields at once:

  | As a… | It does this |
  |---|---|
  | **Attention head** | `softmax(β · Q·Kᵢ)` — structurally identical to transformer attention |
  | **Hopfield network** | The softmax carves basins of attraction, so a noisy or partial query is pulled toward the nearest genuine attractor |
  | **Posterior under a mixture model** | Soft-clustering over prototype locations, weighted by the evidence (`write_count`) each has accumulated |

A compositional vector algebra (`bind` / `unbind` / `permute` / `bundle`) —
Vector Symbolic Architecture in the Holographic Reduced Representation
tradition — sits on top, so you can bind roles to fillers, superpose multiple
bindings, and pull compositional structure back out. That is what makes
`king − man + woman ≈ queen`-style analogical retrieval possible with no
neural network in the loop.

None of this requires training data, GPUs, retraining schedules, or model
versioning. It requires an HTTP POST.

Apache-2.0 licensed, and it runs on a laptop.

## Documentation

The full tree is in [`docs/`](./docs/README.md), and it separates two things
deliberately:

**Heather — the paradigm.** Intelligence as a property of how data is
arranged. [Storage native intelligence](./docs/explanation/storage-native-intelligence.md)
is the argument; [what similarity means](./docs/explanation/what-similarity-means.md)
is the design surface; [boundary conditions](./docs/explanation/boundary-conditions.md)
is where it stops. These stay true of any system built this way.

**HeatherDB — this engine.** [Getting started](./docs/getting-started.md) ·
[deploy](./docs/how-to/deploy.md) · [run it in production](./docs/how-to/production.md) ·
[HTTP API](./docs/api/overview.md) · [configuration](./docs/reference/configuration.md) ·
[parameters](./docs/reference/parameters.md) ·
[behaviour and silent failures](./docs/reference/behaviour.md)

Nine [tutorials](./docs/README.md#tutorials) run in order, each turning on one
result you produce yourself. Every technical term has a
[glossary entry](./docs/terms/vector.md), explained once plainly and once
precisely.

## Quick start

```bash
git clone https://github.com/aphorion/heatherdb
cd heatherdb
cargo build --release -p heather_server

HEATHER_DATA_DIR=/tmp/heatherdb \
HEATHER_DIMENSION=8 \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='quickstart-password' \
  ./target/release/heather
```

The admin password must be at least 8 characters; a shorter one aborts the
boot. Leave it unset in production and the engine generates one, prints it
once, and writes it to `$HEATHER_DATA_DIR/initial-admin-password`.

Write a vector, then read back a corrupted version of it:

```bash
curl -s -u admin:quickstart-password -H 'Content-Type: application/json' \
  -d '{"vectors": [[1, 1, 1, 1, 0, 0, 0, 0]]}' \
  http://localhost:6380/collections/quickstart/write

curl -s -u admin:quickstart-password -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, -1, 0, 0, 0, 0]}' \
  http://localhost:6380/collections/quickstart/read
```

The collection is created by the write; you never declared it. The read returns
the pattern the query belongs to rather than the query or a stored row — see
[Getting started](./docs/getting-started.md) for the rest, including running it
embedded as a Rust library instead of as a server.

## Performance

Measured on Apple Silicon (M-series). Config: `l_0=1000, k=20,
neighbor_cap=max((k-1)·⌈ln L⌉, 2k)`, 32 landmarks. Reproduce with
`cargo bench -p heather_db`.

| Operation | d=64 | d=128 | d=384 |
|---|---|---|---|
| Single write | 3.3 ms | 3.7 ms | 4.1 ms |
| Single read (iterative) | 18 µs | 43 µs | 153 µs |
| Reads/sec (sustained) | 19,418 | 9,336 | 3,473 |

Batch writes amortise lock acquisition and LMDB persistence into one
transaction — 100 vectors take 11.6 ms as a batch against 390 ms individually,
a 33.7× speedup, rising to 77.6× at 500.

Graph search follows the learned manifold instead of scanning every location,
so activation latency is near-constant in collection size: 18 µs at 1,384
locations and 22 µs at 5,097, against 131 µs and 124 µs for brute force.

## Project structure

```
heather_db/               # Engine library
  src/
    collection.rs         # Collection API: write, read, attention, documents
    write.rs              # Competitive learning, splits, merges, the MDL gate
    read.rs               # Activation, the Hopfield loop, graph search
    merge.rs              # KNN merge with posting-list migration
    config.rs             # EAMConfig
    db_config.rs          # Per-database persisted config (dimension, EAM knobs)
    store.rs              # LMDB persistence
    audit.rs              # Buffered, batched access-log storage
    hive.rs               # One LMDB env, many collections
    server.rs             # N hives, one per database
    vec_ops.rs            # Vector math, softmax, HRR unbind
heather_server/           # HTTP server + CLI
  src/
    main.rs               # CLI entry, route assembly, startup
    routes.rs             # Legacy (default-DB) handlers
    routes_db.rs          # Scoped (/db/{db}/…) wrappers
    auth.rs, users.rs, tokens.rs
    dream.rs              # Idle-time consolidation
    backup.rs, cli.rs     # Backup/restore/snapshot + user CLI
heather_algebra/          # Energy-correct vector algebra
docs/                     # The documentation tree
deploy/                   # VPS install script
packaging/deb/            # .deb maintainer scripts + systemd unit
charts/heatherdb/         # Kubernetes Helm chart
```

The dual-router pattern is the thing most likely to confuse a newcomer: every
collection and algebra route exists twice, as `/collections/…` against the
default database and as `/db/{db}/collections/…` scoped. The scoped handlers in
`routes_db.rs` are three-line wrappers that resolve the hive and delegate to
`routes.rs`, so logic belongs in the legacy handler and the wrapper picks it up.
See [architecture](./docs/explanation/architecture.md).

## Development

```bash
cargo test --workspace
cargo test -p heather_server --bin heather        # binary crate — no --lib target
cargo bench -p heather_db                         # all benchmarks
cargo bench -p heather_db --bench graph_search    # one category
```

## Releases

Tagging `v*.*.*` builds seven targets — Linux x86_64 and aarch64 in gnu and
musl, macOS on Apple Silicon and Intel, and Windows x86_64 — and publishes them
to [GitHub Releases](https://github.com/aphorion/heatherdb/releases) with a
combined checksum manifest. Multi-arch container images go to
`aphorion/heatherdb` on Docker Hub and `ghcr.io/aphorion/heatherdb`. macOS
binaries are unsigned, so a downloaded
archive carries `com.apple.quarantine` until it is cleared.

## Related repos

- [`heatherdb-samples`](https://github.com/aphorion/heatherdb-samples) — 26
  example apps built against this engine.
- [`heatherdb-pi-demo`](https://github.com/aphorion/heatherdb-pi-demo) —
  demo rig: recommender catalogue, FastAPI proxy, Raspberry Pi cross-compile.
- [`heatherdb-landing`](https://github.com/aphorion/heatherdb-landing) — the
  product site, which also serves this documentation.

## Based on

*"Adaptive Elastic Associative Memory: Self-Organizing Indexing with
Energy-Based Retrieval"*, and the lineage described in
[Pentti Kanerva](./docs/explanation/kanerva.md).

## Sponsor

HeatherDB is Apache-2.0 and developed in the open by
[Aphorion Labs](https://aphorion.co) in Nairobi. Sponsorship pays for the
unglamorous half of that — measured benchmarks, documentation, release
engineering across seven targets, and the time to answer questions from people
building on it.

[**Sponsor on GitHub**](https://github.com/sponsors/aphorion)

If you are deploying HeatherDB commercially and want support, ingestion help or
a say in the roadmap, [get in touch](https://aphorion.co/contact) — that is a
conversation rather than a checkout.

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

Permissive, with an explicit patent grant: you can use, modify, distribute
and build products on HeatherDB, commercially or otherwise, and the patent
licence means adopting it does not leave you exposed to a later claim over
the algorithm.
