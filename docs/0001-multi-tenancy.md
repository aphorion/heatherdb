# RFC 0001 — Multi-tenancy (the `database` layer)

**Status:** Draft · v0.2 target
**Owner:** —
**Last updated:** 2026-05-11

## Summary

Today, one `heather_server` process owns one data dir, one global vector
dimension, and one flat collection namespace. Hosting two unrelated apps
on the same engine means running two engines on two ports. This RFC adds
a **database** layer between server and collection — the missing middle
that Postgres, MongoDB, and friends all have — so a single engine can
multiplex many isolated, differently-shaped projects.

## Goals

1. **One server hosts many databases.** Each with its own vector
   dimension, EAM config, LMDB env, and snapshot stream.
2. **No regression for single-tenant deploys.** Existing
   `/collections/...` routes keep working; existing data dirs migrate
   transparently.
3. **Per-database backup / restore.** A database is a directory; tar it.
4. **Per-database resource caps.** Map size, soon: rate limit, soon:
   auth.
5. **Operationally boring.** No replication, no clustering, no shared
   global state across DBs that needs locking.

## Non-goals (this RFC)

- Per-database auth (API keys, roles). Separate RFC, v0.3.
- Replication / multi-writer.
- Cross-database algebra. Different dimensions make this not
  geometrically meaningful — refuse loudly.
- Hot dimension change. Dimension is set at DB creation, immutable.
- Hot resize of an LMDB map. v0.3 maybe.

---

## Domain model

```
Server
├── ServerConfig         (port, host, root data dir, request timeout)
├── DatabaseRegistry     (LMDB env at $ROOT/registry/)
└── databases: HashMap<DbName, Arc<Database>>

Database
├── name: String
├── DbConfig             (dimension, eam knobs, map_size_mb)
├── env: heed::Env       (LMDB env at $ROOT/db/<name>/data/)
├── CollectionRegistry   (in-env, just like today)
└── collections: HashMap<ColName, Arc<Collection>>

Collection
├── name
├── (existing internals — locations, neighbour graph, write/read paths,
│    merge, persistence) — unchanged
```

The `Collection` type is essentially what exists today. The new types
sit above it and own the LMDB `Env` (today owned implicitly by
`Collection`).

## Storage layout

```
$HEATHER_DATA_DIR/                  # set globally (server.toml or env)
├── server.toml                     # global config
├── .migrated                       # marker — single→multi migration ran
├── registry/
│   └── data.mdb                    # LMDB env: db_name → DbConfig snapshot
├── db/
│   ├── default/                    # implicit DB if none exist
│   │   ├── db.toml
│   │   └── data/
│   │       └── data.mdb            # the existing 5-sub-DB layout
│   ├── memoria/
│   │   ├── db.toml
│   │   └── data/
│   └── navigator/
│       ├── db.toml
│       └── data/
└── _trash/                         # soft-delete staging (TTL janitor)
    └── memoria-1730000000/
```

Two LMDB envs at minimum: `registry/` plus at least one DB. Multiple DBs
= multiple envs. LMDB's single-writer-per-env limit then applies
**per database**, not globally — concurrent writes across DBs are fine
and the OS handles isolation for free.

Per-DB backup is `tar -czf db/memoria.tar.gz db/memoria/`. Restore is
the inverse. No cross-DB consistency to worry about.

## Per-database config (`db.toml`)

Created by `POST /db { … }`. Never mutated after first write — changes
require a new DB + migration.

```toml
name = "memoria"
created_at = 1730000000
dimension = 384

[eam]
l_0 = 1000
k = 20
neighbour_cap = 96      # auto if omitted: max((k-1)*ceil(ln L), 2k)
beta = 5.0
tau_split = 0.3
tau_overload = 100

[storage]
map_size_mb = 4096
sync = "always"         # "lazy" trades durability for write tps
```

## Public HTTP API

### Database-level

| Method | Path                       | Body / params                                               | Returns                          |
|--------|----------------------------|-------------------------------------------------------------|----------------------------------|
| `POST` | `/db`                      | `{name, dimension, map_size_mb?, eam?}`                     | `DbInfo`                         |
| `GET`  | `/db`                      |                                                             | `{databases: DbInfo[]}`          |
| `GET`  | `/db/{db}`                 |                                                             | `DbInfo` (config + counts)       |
| `DELETE` | `/db/{db}?confirm={name}`|                                                             | `{deleted: true}`                |
| `POST` | `/db/{db}/snapshot`        |                                                             | streamed tar.gz                  |
| `POST` | `/db/{db}/restore`         | streamed tar.gz                                             | `{restored: true}`               |

### Collection-level (now scoped)

| Method | Path                                              | Notes                              |
|--------|---------------------------------------------------|------------------------------------|
| `GET`  | `/db/{db}/collections`                            |                                    |
| `POST` | `/db/{db}/collections/{c}/write`                  | dimension must match DB's          |
| `POST` | `/db/{db}/collections/{c}/read`                   |                                    |
| `POST` | `/db/{db}/collections/{c}/analyze`                |                                    |
| `GET`  | `/db/{db}/collections/{c}/fingerprint`            |                                    |
| `GET`  | `/db/{db}/collections/{c}/documents`              |                                    |
| `DELETE`| `/db/{db}/collections/{c}`                       |                                    |

### Algebra (per-DB, refuses cross-DB)

| Method | Path                          | Notes                                   |
|--------|-------------------------------|-----------------------------------------|
| `POST` | `/db/{db}/algebra/add`        | `{source_a, source_b, target?}` — both inputs must be in `{db}` |
| `POST` | `/db/{db}/algebra/sub`        |                                         |
| `POST` | `/db/{db}/algebra/scale`      | `{source, alpha, target?}`              |

### Backwards-compat shim

The legacy routes are aliases handled by a thin redirect in the router:

```
/collections/{c}/...  →  /db/default/collections/{c}/...
/algebra/{op}         →  /db/default/algebra/{op}
```

No behaviour change for existing clients (heatherdb-pi-demo proxy,
sample projects, Fovea v0.1, deployment scripts).

## Rust module structure

```
heather_db/
  src/
    server/
      mod.rs                Server (top), holds DBs map
      registry.rs           DatabaseRegistry — persisted db list
      config.rs             ServerConfig
    database/
      mod.rs                Database — owns env, holds collections
      config.rs             DbConfig (dim, map_size, eam knobs)
      registry.rs           per-DB CollectionRegistry
    collection/             (existing — promoted into a sub-mod)
      mod.rs
      read.rs · write.rs · merge.rs · location.rs · ...

heather_server/
  src/
    main.rs                 CLI + bootstrap + migration
    state.rs                AppState { server: Arc<Server> }
    routes/
      mod.rs                Axum router assembly
      databases.rs          POST /db, GET /db, etc.
      collections.rs        scoped /db/{db}/collections/...
      algebra.rs            scoped /db/{db}/algebra/...
      legacy.rs             /collections/* + /algebra/* redirect handlers
    models.rs               request/response DTOs
```

## Lifecycle

### Server boot

1. Open `$ROOT/registry/` LMDB env (create if missing).
2. List `(name, _config_pointer)` entries.
3. For each, read `db/<name>/db.toml`, open `db/<name>/data/` LMDB env,
   instantiate `Database`. Lazy: discover collections, don't load
   attractors yet.
4. If registry was empty AND legacy single-tenant layout detected
   (`$ROOT/data.mdb` exists), run **migration** (see below) before
   continuing.
5. If `default` DB still doesn't exist, create it lazily using
   `HEATHER_DIMENSION` env or a default (128).
6. Start Axum.

### Database create (`POST /db`)

```rust
async fn create_db(name, cfg) -> Result<Arc<Database>> {
    validate_name(&name)?;                    // [a-z][a-z0-9_-]{0,62}, reserved list
    let mut map = self.databases.write();
    if map.contains_key(&name) { Err(AlreadyExists) }
    let dir = self.root.join("db").join(&name);
    fs::create_dir_all(dir.join("data"))?;
    write_atomic(dir.join("db.toml"), &cfg)?;  // anchor first
    let env = open_env(dir.join("data"), cfg.map_size_mb)?;
    let db  = Arc::new(Database::open(name.clone(), cfg.clone(), env)?);
    self.registry.insert(&name, &cfg)?;        // persist after env opens cleanly
    map.insert(name, db.clone());
    Ok(db)
}
```

Failures partway through: the `db.toml` anchor lets a re-run pick up a
half-built DB (registry insert is idempotent).

### Database drop (`DELETE /db/{db}?confirm={name}`)

```rust
async fn drop_db(name) -> Result<()> {
    if name == "default" { Err(Refused("cannot drop default")) }   // optional safety
    let arc = self.databases.write().remove(&name).ok_or(NotFound)?;
    // Wait for in-flight handles to drain.
    drop_with_quiescence(arc, Duration::from_secs(30)).await?;
    self.registry.remove(&name)?;
    move_to_trash(self.root.join("db").join(&name))?;
    Ok(())
}
```

`move_to_trash` renames into `_trash/<name>-<ts>/`. A janitor job
(separate concern) GCs old trash entries.

### Default DB

Created lazily on first boot. `dimension` from `HEATHER_DIMENSION` env
if set, else 128. After creation, the env var is no longer consulted —
the value lives in `db/default/db.toml`.

This makes single-tenant deploys zero-config: the user sees the same
HTTP shape they used before, the new layer is invisible.

## Concurrency

- `Server::databases: parking_lot::RwLock<HashMap<DbName, Arc<Database>>>`
  — read-heavy. Acquire read lock per request, look up `Arc`, drop lock,
  use the Arc. Drop takes the write lock briefly to remove the entry,
  then releases before `Arc` ref count drains.
- Each `Database` is wrapped in `Arc` — concurrent requests against the
  same DB don't contend on the server-level lock.
- Within a `Database`, the existing per-collection locking semantics
  carry over.
- LMDB allows multi-reader, single-writer per env. Multiple DBs =
  multiple envs = concurrent writes across DBs are fine.

## Error semantics

New variants on `heather_db::Error`:

```rust
pub enum Error {
    DatabaseNotFound(String),
    DatabaseAlreadyExists(String),
    DatabaseInUse(String),                              // can't drop while active
    InvalidDatabaseName { name: String, reason: &'static str },
    DimensionMismatch { db: String, expected: usize, got: usize },
    CrossDatabaseAlgebra { a: (String, String), b: (String, String) },
    // existing variants…
}
```

HTTP mapping:

| Variant                         | Status |
|----------------------------------|-------|
| `DatabaseNotFound`               | 404 |
| `DatabaseAlreadyExists`          | 409 |
| `DatabaseInUse`                  | 409 |
| `InvalidDatabaseName`            | 400 |
| `DimensionMismatch`              | 400 |
| `CrossDatabaseAlgebra`           | 422 |

## Migration

For an existing instance with the single-tenant layout
(`$ROOT/data.mdb` + `$ROOT/lock.mdb`, no `db/` dir):

1. On boot, detect: `$ROOT/data.mdb exists && !$ROOT/db/.exists`.
2. Create `$ROOT/db/default/data/`.
3. **Move** `$ROOT/data.mdb` → `$ROOT/db/default/data/data.mdb`.
4. Move `$ROOT/lock.mdb`  → `$ROOT/db/default/data/lock.mdb`.
5. Write `$ROOT/db/default/db.toml` with `dimension = HEATHER_DIMENSION`
   (or 128 if unset; emit a warning).
6. Insert `default` into the new registry.
7. Touch `$ROOT/.migrated` (idempotency marker).

Migration is atomic enough — a half-migrated layout is detectable on
the next boot (env in new place, marker missing) and re-runs are no-ops.

A `heather_server migrate --dry-run` CLI subcommand prints what would
move without doing it.

## Configuration precedence

| Setting                  | Scope      | Source                           |
|--------------------------|-----------|-----------------------------------|
| `data_dir`               | global    | `HEATHER_DATA_DIR` / `--data-dir` |
| `host`, `port`           | global    | env / CLI                         |
| `request_timeout`        | global    | env / CLI                         |
| `dimension`              | per-DB    | `db.toml` (set at create)         |
| `map_size_mb`            | per-DB    | `db.toml`                         |
| `eam.*`                  | per-DB    | `db.toml`                         |

`HEATHER_DIMENSION` and `HEATHER_MAP_SIZE_MB` env vars apply **only to
the default DB on first creation**, for back-compat with single-tenant
deploys. Other DBs ignore them — that would conflate scopes.

## Resource accounting

A new `GET /server/stats` endpoint:

```json
{
  "uptime_s": 1234,
  "databases": [
    {
      "name": "default",
      "dimension": 128,
      "collections": 5,
      "total_attractors": 187432,
      "total_writes": 25000000,
      "map_size_mb": 4096,
      "used_mb": 312
    },
    {
      "name": "memoria",
      "dimension": 384,
      "collections": 12,
      ...
    }
  ]
}
```

Fovea's status bar / Inspector use this. Per-DB Prometheus metrics:

```
heather_database_collections{db="memoria"}     12
heather_database_attractors{db="memoria"}      245031
heather_database_writes_total{db="memoria"}    1234567
heather_database_used_bytes{db="memoria"}      327155712
heather_request_duration_seconds_bucket{db="memoria",op="read",le="0.001"} 9821
```

Logs gain a `db=<name>` field on every request span.

## Testing

| Layer | What to verify |
|-------|----------------|
| Unit  | DB create/drop round-trip; default-DB lazy creation; cross-DB collection isolation; cross-DB algebra refused with `422`; name validation. |
| Integration | 100 DBs × 50 collections, no cross-pollution; drop one DB while another is being written to; LMDB env-handle limits. |
| Migration | Single-tenant fixture upgrades to multi-tenant on boot; idempotent re-run; partial-migration recovery. |
| Backup | `POST /db/{db}/snapshot` → `POST /db/{db}/restore` round-trip preserves collections + attractor counts. |
| Fuzz  | DB name validation; concurrent create + drop of same name. |

## Observability

- Per-DB tracing spans (`db = <name>`).
- Per-DB Prometheus metrics (above).
- `journalctl -u heatherdb -f` shows `db=…` on every line.
- Fovea consumes `/server/stats` for the bottom status bar; the
  Connections screen sprouts a database picker after URL.

## Backwards-compat checklist

- [x] `/collections/{c}/...` → `/db/default/collections/{c}/...` shim.
- [x] `/algebra/...` → `/db/default/algebra/...` shim.
- [x] `/health`, `/stats` (when added) — unchanged shape, additive
  fields only.
- [x] `HEATHER_DIMENSION` env — used only when first creating the
  default DB.
- [x] Existing data dirs — migrated on first boot of v0.2.
- [x] CLI flags — unchanged.
- [x] Existing client libs (heatherdb-pi-demo proxy, sample projects,
  Fovea v0.1) — keep working without code changes.

## Out of scope

- Per-DB auth (API keys, scoped roles). Belongs in v0.3 once the
  multi-tenant boundary exists.
- Replication or clustering.
- Cross-DB algebra (likely never — different dimensions, not meaningful).
- Hot dimension change.
- Hot resize of LMDB map. (LMDB supports `mdb_env_set_mapsize` after
  re-open — schedule for v0.3.)

## Phasing

| Phase | Scope                                                      | Effort |
|-------|------------------------------------------------------------|--------|
| 0.2.0 | Domain model + storage layout + scoped routes + legacy shim| 1 wk   |
| 0.2.1 | Migration tooling + `--dry-run` + integration tests        | 2 d    |
| 0.2.2 | `/db/{db}/snapshot` / `restore` endpoints + tar streaming  | 3 d    |
| 0.2.3 | `/server/stats` + Prometheus metrics + per-DB log fields   | 2 d    |
| 0.2.4 | Fovea database picker + DB-scoped Inspector / Algebra      | 2 d    |
| 0.2.5 | Docs (this RFC stays as the spec of record)                | rolling|

Total ≈ 2–3 calendar weeks for one engineer.

## Open questions

1. **Soft-delete vs hard-delete on `DELETE /db/{db}`.** Default to
   trash-with-TTL or hard delete? Postgres-ish: `DROP DATABASE` is
   immediate. Mongo: ditto. Erring toward hard delete with a confirm
   token, no trash. Power users with `_trash/` always have manual
   restore via `restore` endpoint.
2. **Quoting in path segments.** Are `:`, `.`, `/` allowed in DB or
   collection names? Recommendation: no — `[a-z][a-z0-9_-]{0,62}` for
   both, period.
3. **Cap on number of DBs per server.** None enforced; document a soft
   target of 64 (LMDB env handle cost) and let users override with a
   server config setting.
4. **Default DB name override.** Some users may want to call it
   something other than `default`. Recommendation: don't allow — the
   shim semantics depend on a fixed name. They can create
   `production` and never use `default` if that bothers them.
5. **`/server/stats` cost.** Walking every env to compute `used_mb`
   takes a few ms per env. Cache for 1 s server-side.

## Migration risk

Lowest-risk path of any breaking change we can ship. The shim layer
means existing clients literally don't notice. The migration step is a
file rename inside one atomic boot. Worst case rollback: stop the new
server, move `db/default/data/data.mdb` back to `data.mdb`, run the
old binary. Trivial.

The only behavioural change a careful client could observe: a DB-name
collision with a collection literally named `db` in the legacy
namespace — reserved, document the conflict, refuse to migrate if
detected (one error message + manual rename).
