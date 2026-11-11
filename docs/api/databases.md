# Database endpoints

Create, list, inspect and drop databases. A database owns one LMDB
environment, its own vector dimension, and its own collections.

`/db` (the collection of databases) is **root only**. `/db/{db}` is reachable
by root or by a user scoped to that database. See [Overview](overview.md) and
[Run several databases](../how-to/multi-database.md).

### GET /db

List every database.

Auth: Basic or Bearer, **root scope**.

Request body: no body.

```bash
curl -u admin:pw http://localhost:6380/db
```

```json
{ "databases": [
  { "name": "default", "created_at": 1767225600, "dimension": 128,
    "map_size_mb": 4096, "collections": 3 }
] }
```

Half-initialised directories (no readable `db.toml`) are skipped rather than
failing the call.

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a database |
| `500 Internal Server Error` | the database directory could not be listed |

### POST /db

Create a database.

Auth: Basic or Bearer, **root scope**.

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | string | *required* | 1–63 chars, `[a-z][a-z0-9_-]*`. Must not start with `_` |
| `dimension` | integer | *required* | Immutable for the life of the database |
| `map_size_mb` | integer | `4096` | LMDB ceiling for this database's env |
| `mdl_gate` | boolean | `false` | Allocate by description length instead of `tau_split`/`tau_overload` |
| `eam` | object | — | Partial `EAMConfig` override: `k`, `t_max`, `beta`, `neighbor_cap`, `num_landmarks`, `l_0`, `competitive`, `precision`, `residency`. Unset fields keep the engine default; the merged config is validated. `"competitive": false` makes every collection a plain store: writes append each vector as its own location with no activation, merge or migration, use it to retain individual observations, including similarity-based associative recall through attention. Competitive mode learns shared patterns and can adapt through graceful forgetting. See [Choose a memory mode](../how-to/choose-memory-mode.md). `"precision": "f32"` stores vectors in half the space (rounded to f32 when written; all math stays f64). Both are fixed for the database's lifetime, and `GET /db` / `GET /db/{db}` report them. `"residency": "shadow"` (plain databases only) keeps an int8 shadow of each vector in memory instead of the vector, and reads exact vectors from storage when an answer needs them; answers are identical, memory is several times smaller. See [Configuration](../reference/configuration.md#eam-knobs). `"selective_attention"` turns on attention learned from prediction error (plain databases only); see [GET /db/{db}/selective-attention](#get-dbdbselective-attention) and [Selective attention](../explanation/selective-attention.md) |

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"movies","dimension":384,"map_size_mb":32768,
       "eam":{"k":64,"t_max":8}}'
```

Responds with the same shape as [`GET /db/{db}`](#get-dbdb):

```json
{ "name": "movies", "created_at": 1767225600, "dimension": 384,
  "map_size_mb": 32768, "collections": 0, "competitive": true, "precision": "f64", "residency": "memory",
  "format_version": 2, "skipped_records": 0 }
```

| Status | Cause |
|---|---|
| `400 Bad Request` | invalid name, or the merged EAM config failed validation |
| `403 Forbidden` | the caller is scoped to a database |
| `409 Conflict` | a database of that name already exists |
| `500 Internal Server Error` | the database was created but its config could not be read back |

### GET /db/{db}

One database's metadata.

Auth: Basic or Bearer, root or `Database(db)`.

Request body: no body.

```bash
curl -u admin:pw http://localhost:6380/db/movies
```

```json
{ "name": "movies", "created_at": 1767225600, "dimension": 384,
  "map_size_mb": 32768, "collections": 0, "competitive": true, "precision": "f64", "residency": "memory",
  "format_version": 2, "skipped_records": 0 }
```

`collections` is `0` when the collection list cannot be read, not an error.

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a different database |
| `404 Not Found` | no `db.toml` for that name |
| `409 Conflict` | (any `/db/{db}/...` route) the database is on disk in a format newer than this server reads; the message says which. See [Storage format](../reference/storage-format.md) |

`format_version` is the database's on-disk format (`2` since selective
attention; a format-`1` database is raised to `2` the first time a newer
server mounts it, without rewriting any record). Databases created with
`selective_attention` also report it. `skipped_records` counts
records reads have skipped since it was opened because they did not decode
(non-zero means damage).

### DELETE /db/{db}

Drop a database. Moves `$DATA/db/{db}/` to
`$DATA/_trash/{db}-<unix-seconds>/` rather than deleting it — recover by
moving the directory back and restarting.

Auth: Basic or Bearer, root or `Database(db)`. The scope classifier is
method-blind on `/db/{name}/...`, so a user scoped to `movies` can drop
`movies`. Give a database-scoped user a database you are willing to let them
delete.

Request body: no body.

```bash
curl -u admin:pw -X DELETE http://localhost:6380/db/movies
```

```json
{ "dropped": true, "purged": false }
```

`{"dropped": false}` when the database was not mounted.

With `?purge=true` the database is deleted for good instead of kept in
`_trash/` (it is still moved there first, so it is unmounted atomically):
`{"dropped": true, "purged": true}`. Use it for data you will never
recover, such as scratch databases or ones rebuilt from a log; trash is
never emptied automatically, so repeated drops of large databases fill the
disk. Without it, `purged` is `false`.

| Status | Cause |
|---|---|
| `400 Bad Request` | the move to `_trash/` failed |
| `403 Forbidden` | the caller is scoped to a different database, or the target is `default`, which cannot be dropped (the legacy routes target it) |

See also [Backup and restore](../how-to/backup-and-restore.md).

### GET /db/{db}/selective-attention

What a database created with `selective_attention` has learned about where
to look. Legacy: `GET /selective-attention` (the `default` database).

Auth: Basic or Bearer, root or `Database(db)`.

Request body: no body.

```bash
curl -u admin:pw http://localhost:6380/db/crm/selective-attention
```

```json
{ "k": 200, "warmup": 64,
  "heads": { "outcome": { "dims": [51, 52],
                          "weights": [0.31, 0.02, …, 0.0, 0.0],
                          "updates": 31583 } },
  "time": { "bucket_starts": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
            "weights": [1.0, 0.77, 0.61, …],
            "updates": 327412 } }
```

A head's `weights` are its attention per dimension, the head's own fields
reading `0`; `null` until the head has seen `warmup` rows. Their scale is
also the read's temperature. `time.weights` are the consolidated per-age
weights reads use; bucket `b` covers lags from `bucket_starts[b]` up to the
next start (lag `1` is a collection's newest row). `time` is absent when the
database was created without it.

| Status | Cause |
|---|---|
| `404 Not Found` | no such database, or it was created without selective attention |

### GET /db/{db}/durability

Whether the database syncs every commit to disk.

Auth: Basic or Bearer, root or `Database(db)`.

Request body: no body.

```bash
curl -u admin:pw http://localhost:6380/db/movies/durability
```

```json
{ "database": "movies", "sync": true }
```

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a different database |
| `404 Not Found` | no such database |

### PUT /db/{db}/durability

Stops or restarts syncing every commit to disk. By default each commit is
synced (LMDB's default), which is what makes a write durable, and on
macOS it is a full device flush (`F_FULLFSYNC`) that can dominate write
time. `{"sync": false}` lets commits reach the operating system without
waiting for the disk (LMDB `NO_SYNC`): much faster bulk loads, at the cost
that a **system** crash (not a server crash) can lose recent commits, or
corrupt the database on file systems that reorder writes. Use it for data
you can rebuild, such as a memory replayed from a log, and turn it back on
when the load is done: `{"sync": true}` flushes everything written
meanwhile before it answers.

The setting is not persisted: a restarted server syncs every commit again.

Auth: Basic or Bearer, root or `Database(db)`.

| Field | Type | Notes |
|---|---|---|
| `sync` | boolean | *required* |

```bash
curl -u admin:pw -X PUT http://localhost:6380/db/movies/durability \
  -H 'content-type: application/json' -d '{"sync": false}'
```

```json
{ "database": "movies", "sync": false }
```

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a different database |
| `404 Not Found` | no such database |
| `500 Internal Server Error` | the flag could not be changed or the flush failed |
