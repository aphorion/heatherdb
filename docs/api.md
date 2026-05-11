# HTTP API reference

Every route. Every body. Every status code that means something
specific. Routes are grouped by surface area; the **scope column** in
each table tells you which user scopes can call it.

| Symbol | Meaning |
|--------|---------|
| ✓     | Allowed |
| —     | Forbidden (403) |
| ✓¹     | Allowed only when `Database("default")` |
| open  | No auth check (always allowed) |

All bodies are JSON unless noted. All errors are
`{"error": "<message>"}` with an HTTP status.

---

## Health

| Method | Path     | Scope | Notes |
|--------|----------|-------|-------|
| `GET`  | `/health`| open  | Liveness probe. |

```bash
curl http://127.0.0.1:6380/health
# {"status":"ok"}
```

Always 200 if the process is up. Doesn't indicate that the LMDB env
is mountable or the disk has space — for that, use `/db/{name}` on a
known DB.

---

## Database management

All require `Root`.

| Method | Path           | Scope | Body / params                                                     | Returns                                |
|--------|----------------|-------|-------------------------------------------------------------------|----------------------------------------|
| `GET`  | `/db`          | Root  |                                                                   | `{ databases: DatabaseInfo[] }`        |
| `POST` | `/db`          | Root  | `{ name, dimension, map_size_mb? }`                               | `DatabaseInfo`                         |
| `GET`  | `/db/{name}`   | Root  |                                                                   | `DatabaseInfo`                         |
| `DELETE`| `/db/{name}?confirm={name}` | Root | (refuses `default`)                                       | `{ dropped: boolean }`                 |

```ts
type DatabaseInfo = {
  name: string;
  created_at: number;       // unix epoch seconds
  dimension: number;
  map_size_mb: number;
  collections: number;
};
```

### `POST /db`

```bash
curl -u admin:'…' -X POST http://127.0.0.1:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"memoria","dimension":384,"map_size_mb":4096}'
```

| Status | When |
|--------|------|
| 200    | Created (or already existed and config matches). |
| 400    | Invalid name / invalid dimension. |
| 409    | Same name already exists. |

### `DELETE /db/{name}`

Moves `$ROOT/db/{name}` to `$ROOT/_trash/{name}-<unix-ts>`. Recoverable
by stopping the engine and renaming back. `default` is refused with
403 — wipe by stopping the engine and removing the directory.

---

## Collections (database-scoped)

Every collection lives inside a database. The new shape:
`/db/{db}/collections/{name}/...`. The engine also keeps **legacy
aliases** that target `default` (the auto-created DB) to avoid breaking
existing clients.

| Method | New path                                              | Legacy alias                                | Scope |
|--------|-------------------------------------------------------|---------------------------------------------|-------|
| `POST` | `/db/{db}/collections`                                | `/collections`                              | Root or scope==`db` |
| `GET`  | `/db/{db}/collections`                                | `/collections`                              | Root or scope==`db` |
| `DELETE`| `/db/{db}/collections/{name}`                        | `/collections/{name}`                       | Root or scope==`db` |
| `POST` | `/db/{db}/collections/{name}/write`                   | `/collections/{name}/write`                 | Root or scope==`db` |
| `POST` | `/db/{db}/collections/{name}/read`                    | `/collections/{name}/read`                  | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/stats`                   | `/collections/{name}/stats`                 | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/config`                  | `/collections/{name}/config`                | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/locations`               | `/collections/{name}/locations`             | Root or scope==`db` |
| `POST` | `/db/{db}/collections/{name}/analyze`                 | `/collections/{name}/analyze`               | Root or scope==`db` |
| `POST` | `/db/{db}/collections/{name}/batch_analyze`           | `/collections/{name}/batch_analyze`         | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/fingerprint`             | `/collections/{name}/fingerprint`           | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/documents`               | `/collections/{name}/documents`             | Root or scope==`db` |
| `POST` | `/db/{db}/collections/{name}/documents/query`         | `/collections/{name}/documents/query`       | Root or scope==`db` |
| `GET`  | `/db/{db}/collections/{name}/documents/{doc_id}`      | `/collections/{name}/documents/{doc_id}`    | Root or scope==`db` |

The legacy alias is allowed only for `Root` and `Database("default")`
scopes. New code should use the explicit form.

### `POST /db/{db}/collections/{name}/write`

```ts
type WriteRequest = {
  vectors: number[][];               // each vector must match the DB's dimension
  metadata?: unknown[];              // optional, must align with `vectors` if present
};
type WriteResponse = {
  count: number;                     // how many were stored
  ids?: number[];                    // present only when metadata was supplied
};
```

```bash
curl -u admin:… -X POST http://127.0.0.1:6380/db/default/collections/notes/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[0.1, 0.2, …], [0.3, 0.4, …]]}'
```

| Status | When |
|--------|------|
| 200    | Stored. |
| 400    | Empty `vectors` array, dimension mismatch, or invalid metadata. |
| 404    | Database doesn't exist. |
| 408    | Write timed out (raise `--request-timeout` for big batches). |

### `POST /db/{db}/collections/{name}/read`

```ts
type ReadRequest = {
  query: number[];                   // dimension must match the DB's
  strategy?: "iterative" | "fast";   // default "iterative"
};
type ReadResponse = { result: number[] };
```

`iterative` runs the Hopfield read until convergence (up to
`config.t_max` iterations). `fast` is one shot — use for latency-bounded
paths where good-enough recall is fine.

### `POST /db/{db}/collections/{name}/analyze`

Same input as `read`, but returns the full activation trace — what
fired, with what weight, how long it took to settle:

```ts
type AnalyzeResponse = {
  iterations: number;
  converged: boolean;
  total_activations: number;
  activated_locations: { id: number; similarity: number; weight: number }[];
  result: number[];
};
```

This is the "why this?" feature. Use it whenever you need to explain
or debug a recall.

### `GET /db/{db}/collections/{name}/fingerprint`

```ts
type FingerprintResponse = { fingerprint: number[] };
```

The collection's centroid — a single vector that summarises the
attractor population. Re-querying the collection with its own
fingerprint via `/analyze` lights up the dominant attractors (Fovea
calls this the "probe fingerprint" button).

---

## Algebra (database-scoped)

| Method | New path                            | Legacy alias              | Body                                          |
|--------|-------------------------------------|---------------------------|-----------------------------------------------|
| `POST` | `/db/{db}/algebra/add`              | `/algebra/add`            | `{ source_a, source_b, target?, max_cross_k?, k?, min_ratings? }` |
| `POST` | `/db/{db}/algebra/sub`              | `/algebra/sub`            | (same)                                        |
| `POST` | `/db/{db}/algebra/scale`            | `/algebra/scale`          | `{ source, alpha, target?, k?, min_ratings? }`|
| `POST` | `/db/{db}/algebra/intersect`        | `/algebra/intersect`      | `{ source_a, source_b, target?, threshold? }` |
| `POST` | `/db/{db}/compose/read`             | `/compose/read`           | `{ ops, target? }`                            |

Returns:

```ts
type AlgebraResponse = {
  collection: string;
  num_locations: number;
};
```

**Inputs must live in the same database.** Cross-database algebra is
rejected (different dimensions, different attractor spaces). See
[databases](./databases.md#algebra-and-database-boundaries).

### Energy-correct addition

Mathematically: `E(A+B) = E(A) + E(B)`. The result is a new collection
whose attractors are the pairwise sums of A's and B's. n×m new
locations; cap with `max_cross_k` for very large pairs.

### Scale

`scale(source, alpha)` rewrites the source's counters in-place
(produces a new collection with `count_i' = alpha * count_i`).
Equivalent to changing the read temperature — used as a building block
for "scaled subtraction" (`A - alpha*B`).

---

## Failure modes (across all routes)

| Status | Meaning |
|--------|---------|
| 400    | Bad input — invalid JSON, dimension mismatch, name validation failed, etc. |
| 401    | Missing or malformed `Authorization` header. Response carries `WWW-Authenticate: Basic realm="heatherdb"`. |
| 403    | Authenticated, but the user's scope doesn't permit the path. Body says which scope is missing. |
| 404    | Database, collection, or document doesn't exist. |
| 408    | Request timed out — raise `--request-timeout` or break the work into smaller chunks. |
| 409    | Conflict — `POST /db` for an existing name. |
| 500    | Internal error. Body has the underlying message. Check the server logs. |

## Versioning policy

The engine is pre-1.0 — minor versions can break wire shapes. The
**legacy aliases** (`/collections/...`, `/algebra/...`, `/compose/...`)
are **kept indefinitely** so existing clients don't break across
multi-tenancy upgrades. New routes will only be added under
`/db/{db}/...` or `/server/...`.
