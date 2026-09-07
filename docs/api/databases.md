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
| `eam` | object | — | Partial `EAMConfig` override: `k`, `t_max`, `beta`, `neighbor_cap`, `num_landmarks`, `l_0`. Unset fields keep the engine default; the merged config is validated |

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"movies","dimension":384,"map_size_mb":32768,
       "eam":{"k":64,"t_max":8}}'
```

Responds with the same shape as [`GET /db/{db}`](#get-dbdb):

```json
{ "name": "movies", "created_at": 1767225600, "dimension": 384,
  "map_size_mb": 32768, "collections": 0 }
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
  "map_size_mb": 32768, "collections": 0 }
```

`collections` is `0` when the collection list cannot be read, not an error.

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a different database |
| `404 Not Found` | no `db.toml` for that name |

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
{ "dropped": true }
```

`{"dropped": false}` when the database was not mounted.

| Status | Cause |
|---|---|
| `400 Bad Request` | the move to `_trash/` failed |
| `403 Forbidden` | the caller is scoped to a different database, or the target is `default`, which cannot be dropped (the legacy routes target it) |

See also [Backup and restore](../how-to/backup-and-restore.md).
