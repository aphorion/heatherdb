# Architecture

HeatherDB is a Rust workspace of three crates behind one HTTP binary. This page
covers how they fit together, the one routing pattern that surprises people,
and the concurrency and storage models underneath.

## The three crates

**`heather_db`** is the engine library, and holds everything that is actually
a database. It is layered:

```
Server        N databases, one per LMDB environment
  └ Hive      one LMDB environment, many collections
      └ Collection   one adaptive memory: locations, graph, documents
          └ Store    the LMDB layer, six named sub-databases
```

`Server` was added in v0.2 and sits *above* `Hive`. The distinction matters
because it is where the dimension lives: **vector dimension is per database,
not per server**, persisted as `eam.d` in each database's `db.toml`.

**`heather_algebra`** holds the energy-correct vector algebra —
`bind`/`unbind`/`permute`/`bundle`/`compose` — plus `EAMSnapshot`, the vehicle
for ferrying memory state between collections without going through LMDB.

**`heather_server`** is the Axum HTTP layer and the operator CLI. It imports
`heather_db::Server` and exposes it. It is a **binary crate with no library
target**, which matters when you run its tests:

```bash
cargo test -p heather_server --bin heather        # not --lib; there isn't one
```

## The dual-router pattern

This is the thing most likely to confuse you when reading the code.

Every collection and algebra route exists in two shapes:

1. **Legacy** — `/collections/{name}/...`, `/algebra/...`. These take
   `State(hive): State<Arc<Hive>>`: the *default* database's hive, resolved
   once at boot in `main.rs`. Kept for backwards compatibility.
2. **Scoped** — `/db/{db}/collections/{name}/...`, `/db/{db}/algebra/...`.
   These take `Extension(server): Extension<Arc<Server>>` plus a `Path` that
   includes the database name, and resolve the hive per request.

The handlers live in two files, and the split is not symmetric:

- **`routes.rs`** holds the full implementations.
- **`routes_db.rs`** holds the scoped handlers, and **each is a three-line
  wrapper**: resolve the `Arc<Hive>` from the database name, synthesise fresh
  `State(...)` / `Path(...)` extractors, delegate to the matching legacy
  handler.

So the collection logic is single-sourced. **Extend the legacy handler and the
scoped shape picks it up automatically** — never duplicate logic across the
two. The one maintenance cost is that when a legacy handler grows a new
argument, its wrapper needs the matching update; the call sites are kept
line-aligned to make that obvious.

The router is assembled in `main.rs` by merging four sub-routers — `legacy`
(with `Arc<Hive>` state), `db_admin`, `db_scoped`, `auth` — onto a base router
carrying the health checks. Auth middleware is layered last, so it wraps
everything.

Two consequences worth knowing:

- The audited read routes are instrumented **once each, in `routes.rs`**; the
  scoped wrappers inherit the instrumentation and simply overwrite the audit
  context's database field with the one from the path.
- The `/vec/*` routes have no scoped twin, because they touch no database.

## Request lifecycle

```
TCP → body-size limit (2 MB)
    → tower stack: trace, permissive CORS, request timeout (30s → 408)
    → Extension(Server), Extension(Tokens)
    → activity stamp (feeds idle detection for dreaming)
    → auth middleware  ── 401 / 403 ──▶
    → router → handler
        └ spawn_blocking for anything that touches the engine
```

Health routes (`/health`, `/healthz`, `/ready`, `/readyz`) short-circuit auth
by path whitelist, before any credential parsing.

Every handler that touches LMDB or runs engine compute does so inside
`tokio::task::spawn_blocking`, so the async reactor never blocks on a
transaction or a `d`-dimensional matrix walk.

## Storage and concurrency

Each database is an independent LMDB environment in its own directory.
Cross-database isolation is provided by the operating system — separate
environments, separate file handles — not by key-prefix multiplexing. Inside an
environment, six named sub-databases: `_registry`, `_locations`, `_metadata`,
`_documents`, `_doc_index`, `_audit`. Keys are composite and big-endian, so
LMDB's own ordering does useful work: `[collection_id | location_id]` makes a
collection scan a prefix scan, and `[timestamp_ms | seq]` makes the audit log's
key order *be* time order — newest-first paging is a reverse cursor walk, and a
time window is a bounded scan rather than a table read.

The concurrency model follows LMDB's: many concurrent readers, exactly one
writer. `Hive` and `Collection` are `Send + Sync` and every method takes
`&self`; reads take a shared lock, writes an exclusive one. Readers never block
on the writer.

This is also why the engine takes an exclusive advisory lock (`engine.lock`) on
the data directory for its whole lifetime and refuses to double-mount. Two
engine processes on one data directory corrupt LMDB. The cold `backup` and
`restore` subcommands take the same lock, which is exactly how they refuse to
run against a live engine.

Single-writer is why the Helm chart has no `replicaCount`. Adding pods against
one volume does not scale reads; it corrupts data. Scale vertically, or shard
across databases.

Two places in the engine trade transactional purity for throughput, both
deliberately:

- **The audit log** buffers in memory and flushes in batches, because a write
  transaction per read would funnel every concurrent reader through the single
  writer lock. The cost is a bounded loss window on an unclean shutdown. See
  [The audit log and privacy](audit-and-privacy.md).
- **Algebra and `bulk_load`** write their results version-checked rather than
  under a held lock: they read a target version up front and refuse with `409`
  if a concurrent write landed first. Retryable, and it keeps long operations
  from holding the writer.

## The data directory as the interface

There is no configuration server and no control plane. A deployment is one
directory:

```
server.toml   engine.lock   system/data/   db/<name>/{db.toml,data/}
snapshots/    backups/      _trash/
```

`server.toml` is a layout tag; the engine refuses to mount a layout it does not
recognise rather than risk silent corruption. `db.toml` is human-readable on
purpose — `cat db.toml` should explain a database without grepping the engine
source — and is written atomically so a crash mid-write cannot leave a config
that fails to parse.

Everything an operator needs to do lives at that level. Recovering a dropped
database is `mv` out of `_trash/`. Raising a map ceiling is an edit and a
restart. Moving a database between servers is a snapshot and a copy.

## Pre-1.0: no migrations

There is no migration code anywhere in the repository, and that is the policy,
not an oversight. Layout and format changes are fresh-start: back up, recreate,
re-ingest. The layout tag exists so that an incompatible engine refuses loudly
instead of mounting and corrupting.

The user store moved from a `users.json` file to LMDB in v0.3 with no migration
path — an upgraded engine simply ignores an old `users.json` and bootstraps a
fresh admin.

## Where to look in the source

| Question | File |
|---|---|
| What routes exist? | `heather_server/src/main.rs` |
| What does route X do? | `heather_server/src/routes.rs` |
| What is the JSON shape? | `heather_server/src/models.rs` |
| Who is allowed to call it? | `heather_server/src/{auth,users}.rs` |
| How does a write work? | `heather_db/src/write.rs` |
| How does a read work? | `heather_db/src/read.rs` |
| What is persisted, and how? | `heather_db/src/store.rs` |
| What knobs exist? | `heather_db/src/{config,db_config}.rs` |
