# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository orientation

HeatherDB is a Rust HTTP database that stores vectors in an Adaptive Elastic
Associative Memory. The repo holds **the engine + the operator GUI**:

- **Rust workspace** at the root — engine library, HTTP server, vector
  algebra, model-ingestion CLI (`Cargo.toml` lists 4 members).
- **`fovea/`** — a Tauri 2 + React 19 desktop app. **Deliberately excluded
  from the workspace** via `exclude = ["fovea/src-tauri"]` because it has
  its own resolver, deps, and build cadence. Touching the engine workspace
  doesn't touch Fovea and vice versa.
- **`docs/`** — operator-grade documentation; `docs/README.md` is the
  index. Always cross-link these instead of restating their content.
- **`deploy/`** — `deploy/deploy` is a self-contained in-VPS install
  script + the systemd unit it drops.

The deeper "why does this look like this" answers live in
`docs/0001-multi-tenancy.md`. Read that before refactoring anything in
`heather_db::server` or `heather_server::routes_db`.

`docs/0002-binding-operator.md` is the second RFC — it documents the
**HRR binding operator** added to `heather_algebra`, the 17 substrate
demos in `heather_algebra/examples/`, and the substrate-level identities
(EAM read = SDM = HRR cleanup = transformer attention = free-energy
descent). Read that before touching `heather_algebra/src/bind.rs` or
any of the `examples/*.rs` demos.

## Common commands

### Engine workspace

```bash
cargo build --release -p heather_server
cargo check                                          # whole workspace
cargo check -p heather_db                            # one crate
cargo test -p heather_db                             # engine lib tests (50+)

# heather_server is a *binary* crate, not a lib — there's no `--lib` target.
cargo test -p heather_server --bin heather                     # all
cargo test -p heather_server --bin heather users::             # one mod
cargo test -p heather_server --bin heather users::tests::create_and_verify

cargo bench -p heather_db                                  # all benches
cargo bench -p heather_db --bench write_throughput         # one
# benches: capacity, graph_search, multi_collection, persistence,
#          read_latency, store_ops, write_throughput
```

### Running the server

```bash
HEATHER_DATA_DIR=/tmp/heatherdb HEATHER_DIMENSION=128 \
HEATHER_ADMIN_USER=admin HEATHER_ADMIN_PASSWORD='change-me-1234' \
  cargo run --release -p heather_server
```

`HEATHER_AUTH_DISABLED=1` skips auth (dev only — emits a loud warning).
First boot mints the admin user from the env vars; if both are unset, a
random 24-char password is printed once on stderr.

### Engine subcommands (`heather <subcommand>`)

```bash
# Run mode is the default. Subcommands run synchronously and exit:
heather --data-dir DIR user create NAME --password PW [--scope root|<db>]
heather --data-dir DIR user {list,delete NAME,passwd NAME --password PW}

heather --data-dir DIR backup create [--db NAME] --output X.tar.gz
heather --data-dir DIR backup list                      # $DATA/backups/
heather --data-dir DIR restore restore --input X.tar.gz [--db NAME]

heather --data-dir DIR snapshot create --db NAME [--output DIR] [--compact]
heather --data-dir DIR snapshot {list,delete NAME}      # $DATA/snapshots/
```

`snapshot create` is **safe with the engine running** (live LMDB
copy via `Env::copy_to_file`). `backup` is cold; `restore` requires
the engine stopped (LMDB lock).

### Fovea (separate sub-project)

```bash
cd fovea
npm install
npx tauri icon src-tauri/icons/icon.svg     # one-time, generates platform icons
npm run tauri:dev                            # desktop dev shell (recommended)
npm run dev                                  # browser-only iteration (CORS may bite)
npm run build                                # web bundle to dist/
npm run tauri:build                          # signed installable
```

Type-check before claiming a Fovea change works: `cd fovea && npx tsc -b`.

### Docker

```bash
docker compose up -d            # builds engine + (optional) fovea container
docker compose logs -f
```

The `fovea` service may be commented out in `docker-compose.yml` — leave
it however the user has it.

**Production target is Coolify** (self-hosted PaaS, Docker-Compose
runner). The compose file is already Coolify-friendly: every tunable
is `${VAR:-default}` so Coolify's UI can override; named volume
`heatherdb_data` maps to Coolify's persistent storage; healthcheck
drives Coolify's app-up status. Don't introduce host-port hardcoding
or hostname assumptions that would break behind Coolify's Traefik.
Full guide: `docs/coolify.md`.

## Architecture (the bits that span files)

### Engine crate split

- **`heather_db`** — the engine library. `Hive` (one LMDB env, many
  collections) wraps `Store` (LMDB layer with 5 named sub-DBs:
  `_registry`, `_locations`, `_metadata`, `_documents`, `_doc_index`).
  `Server` (added in v0.2) sits *above* `Hive` and owns N hives, one
  per database. **Vector dimension is per-DB**, not per-server: it
  lives in `db_config::DbConfig::eam.d` (persisted as `db.toml` per DB).
- **`heather_server`** — Axum HTTP. Imports `heather_db::Server` and
  exposes it.
- **`heather_algebra`** — energy-correct vector algebra ops (`add`,
  `sub`, `scale`, `intersect`) plus **HRR binding** (`bind`, `unbind`,
  `circular_convolve`, `involve`) in `bind.rs`. Owns `EAMSnapshot` for
  ferrying state between collections. **17 runnable demos** under
  `heather_algebra/examples/` substantiate the substrate framing — see
  RFC 0002. The `MiniEAM` shim in those demos is a ~50-line in-memory
  mirror of `heather_db::Collection`'s read/write shape; the
  `engine_integration` example proves the substrate algebra composes
  with the production engine without API change.
- **`heather_fornix`** — model-ingestion (Pinecone/Weaviate/etc. → Heather).

### The route-table dual-router pattern (the most likely thing to confuse you)

Every collection / algebra route exists in **two shapes**:

1. **Legacy** `/collections/{name}/...` and `/algebra/...` — kept for
   backwards-compat. These take `State(hive): State<Arc<Hive>>` (the
   *default* DB's hive, resolved once at boot in `main.rs`).
2. **Scoped** `/db/{db}/collections/{name}/...` and `/db/{db}/algebra/...`
   — multi-tenancy. These take `Extension(server): Extension<Arc<Server>>`
   plus a `Path<(String, String)>` (db + collection name) — they resolve
   the hive from the path.

Algebra routes currently exposed (both shapes): `add`, `sub`, `scale`,
`intersect`, `bind`, `unbind`.

The handlers live in two files:

- **`routes.rs`** — the legacy handlers, full implementations.
- **`routes_db.rs`** — the scoped handlers. **Each is a 3-line wrapper**
  that resolves the `Arc<Hive>` from the DB name and synthesises a
  fresh `State(...)`/`Path(...)` extractor to delegate into the
  matching legacy handler. **Don't duplicate logic** — extend the
  legacy handler and the wrapper picks it up automatically.

**Quirk on `/algebra/unbind`**: its request takes `key_vector: Vec<f64>`
(a raw vector), not a `key: String` collection name like the other
two-argument ops. Reason: `heather_algebra::unbind` requires a
single-location key snapshot, but the engine's adaptive memory creates
~1000 locations per collection by default. Users hold binding keys as
plain vectors (random unit vectors generated at agent birth, persisted
as JSON), not as separate collections. The handler builds a one-location
snapshot from the supplied vector internally.

The router is assembled in `heather_server/src/main.rs` by `.merge()`-ing
three sub-routers: `legacy_router` (with `Arc<Hive>` state), `db_admin_router`,
and `db_scoped_router`. Auth middleware is layered last with
`from_fn_with_state(auth_state, auth::middleware)`.

### Auth model (`auth.rs` + `users.rs`)

- HTTP Basic Auth, **on by default**. `--auth-disabled` flag for dev.
- User store is **LMDB at `$ROOT/system/data/`** (was `users.json` in
  v0.2.x — JSON is gone, no migration). `verify()` opens a fresh read
  txn per request, so CLI mutations are visible to the running engine
  immediately, no restart.
- **Scope** is `Root` or `Database(name)`. `is_authorized(scope, path)`
  classifies routes:
  - `/health` always passes (whitelist before middleware fires).
  - `/db` (mgmt) requires `Root`.
  - `/db/{name}/...` requires `Root` or `Database(name)`.
  - Legacy `/collections/...` etc. require `Root` or `Database("default")`.
- **Important quirk**: `Scope` enum uses bincode's default discriminant
  encoding — do **not** add `#[serde(tag, content)]` attributes to it
  (bincode doesn't support tagged enums).

### Storage layout under `$HEATHER_DATA_DIR/`

```
server.toml              # tag file (engine layout marker)
system/data/             # LMDB env: HTTP Basic Auth user store
db/<name>/db.toml        # per-DB persisted config (dimension, EAM knobs)
db/<name>/data/          # LMDB env per database
snapshots/               # default landing zone for `snapshot create`
_trash/                  # dropped DBs land here (recoverable by mv-back)
```

### CI / release

`.github/workflows/`:
- **ci.yml** — `fmt → clippy → test` across Linux/macOS/Windows × stable+beta,
  plus `doc`, `audit`, `msrv` (1.84 — required by `resolver = "3"`).
- **release.yml** — fires on `v*.*.*` tags. Builds 8 targets including
  **freebsd-x86_64** (via `vmactions/freebsd-vm@v1`). Cosign-signs every
  pushed tag.
- **docker.yml** — multi-arch (amd64+arm64) image to `ghcr.io`.

## Things that will trip you up

- **`heather_server` has no library target.** `cargo test -p heather_server --lib`
  fails. Use `--bin heather` to run its tests.
- **The workspace excludes `fovea/src-tauri`.** Adding it back will
  break `cargo check` (different Rust resolver, different deps).
- **Dimension is immutable per-DB.** Engine refuses to load a
  collection at a dimension that doesn't match `db.toml`. To change,
  create a new DB and re-write data.
- **Default DB is named `default`** and cannot be dropped via the API
  (`DELETE /db/default` returns 403). The legacy non-`/db/` routes all
  target it.
- **Pre-1.0**: no migration code anywhere. Per the user's standing
  policy, fresh-start changes are fine; don't write migration shims
  unless asked.
- **`heather_fornix`** existed before multi-tenancy — when adding new
  APIs, mirror the `routes.rs` / `routes_db.rs` dual-router pattern.
- **`heather_algebra::bind` is both a module and a re-exported function.**
  `pub mod bind` exposes the module; `pub use bind::{bind, unbind, ...}`
  re-exports the functions at crate root. Inside `heather_server` we
  call `heather_algebra::bind::bind_with_limit(...)` and
  `heather_algebra::bind::unbind(...)` with the explicit module path
  to disambiguate.
- **Substrate demos use `MiniEAM`, not `Collection`.** The 17 demos
  under `heather_algebra/examples/` use a 50-line in-memory shim
  (`MiniEAM`) that has the same read/write/softmax shape as
  `Collection`. This is deliberate — it isolates the substrate-algebra
  proof from the storage-engine proof. The single exception is
  `engine_integration.rs`, which uses both side-by-side and is the
  validation that the algebra composes with the production engine
  (within ~5pt accuracy, with ~16× LMDB write overhead, 30% faster
  reads via graph-search activation).

## Sibling repos (referenced from this code, not vendored)

- `aphorion/heatherdb-pi-demo` — FastAPI proxy + recommender bakers.
  Sets `HEATHER_AUTH_USER` + `HEATHER_AUTH_PASSWORD` to talk to this
  engine. Splits its data into `movies` + `anomaly` databases.
- `aphorion/heatherdb-landing` — the Next.js landing site that talks
  to the proxy (never directly to the engine).
- `aphorion/heatherdb-samples` — 26 example apps.
