# HTTP API reference

Every route the engine serves, as of `heather_server` 0.3.0. Bodies are JSON
(`Content-Type: application/json`); responses are JSON.

Source of truth: `heather_server/src/main.rs` (route table),
`heather_server/src/routes.rs` (handlers), `heather_server/src/models.rs`
(wire types).

## Conventions

### The two route shapes

Every collection and algebra route exists twice:

| Shape | Prefix | Targets |
|---|---|---|
| **Legacy** | `/collections/...`, `/algebra/...`, `/compose/...` | the `default` database |
| **Scoped** | `/db/{db}/collections/...`, `/db/{db}/algebra/...`, `/db/{db}/compose/...` | the named database |

The two are the same handler. The scoped shape resolves a database from the
path and delegates; behaviour, request bodies and responses are identical.
Below, a route written `…/collections/{name}/read` means both
`/collections/{name}/read` and `/db/{db}/collections/{name}/read`.

The `/vec/*` routes are **not** mirrored under `/db/{db}/` — they are stateless
vector math and never touch a database.

### Authentication

HTTP Basic on every route except `/health`, `/healthz`, `/ready`, `/readyz`.
A bearer token from `POST /auth/token` may be sent instead. See
[Manage users and authentication](../how-to/manage-users.md).

```
Authorization: Basic base64(user:password)
Authorization: Bearer <token>
```

### Scope

A user is `root` or scoped to one database.

| Path | Root | `Database(name)` | `Database("default")` |
|---|---|---|---|
| `/health`, `/healthz`, `/ready`, `/readyz` | ✅ (no auth) | ✅ | ✅ |
| `/auth/token`, `/auth/token/revoke` | ✅ | ✅ | ✅ |
| `/db` | ✅ | ❌ 403 | ❌ 403 |
| `/db/{name}/...` | ✅ | ✅ only when `name` matches | ✅ only for `default` |
| `/db/{name}/audit` | ✅ | only if that database sets `visibility = "DbUsers"` | same |
| `/collections/...`, `/algebra/...`, `/compose/...` | ✅ | ❌ 403 | ✅ |
| `/vec/...` | ✅ | ❌ 403 | ❌ 403 |

`/vec/*` is root-only. The classifier admits a `default`-scoped user to
`/collections`, `/algebra` and `/compose` prefixes only; `/vec` is not on that
list (`heather_server/src/users.rs::is_authorized`).

### Errors

Failures return `{"error": "<message>"}` with one of:

| Status | Meaning |
|---|---|
| `400 Bad Request` | malformed body, dimension mismatch, unknown strategy, a cap exceeded, an engine validation error |
| `401 Unauthorized` | missing, malformed, or invalid credentials; expired bearer token. Carries `WWW-Authenticate: Basic realm="heatherdb"` |
| `403 Forbidden` | authenticated but out of scope; dropping the `default` database; minting a token with a bearer token |
| `404 Not Found` | no such database, collection, document, or source collection |
| `408 Request Timeout` | exceeded `--request-timeout` (default 30s) |
| `409 Conflict` | database name already exists; a concurrent write invalidated an algebra or `bulk_load` target (retryable); audit read on a database with logging disabled |
| `413 Payload Too Large` | body over `--max-body-size` (default 2 MiB) |
| `500 Internal Server Error` | storage failure or a panicked worker task |
| `503 Service Unavailable` | `/ready` could not reach the storage layer |

Note the asymmetry on reads and writes: `write`, `read`, `attention`,
`analyze`, `batch_analyze`, `documents/query` and `bulk_load` call
`get_or_create_collection`, so an unknown collection is **created**, not a 404.
`stats`, `config`, `locations`, `fingerprint`, `compress`, `documents` and the
algebra source collections require the collection to exist and return 404.

---

## Ops

### `GET /health` · `GET /healthz`

Liveness. Unauthenticated, never touches storage.

```json
{ "status": "ok" }
```

### `GET /ready` · `GET /readyz`

Readiness. Unauthenticated. Lists databases and performs a real LMDB read
against the `default` hive. `200` with `{"status":"ok"}`, or `503` with an
error when the storage layer is broken — so an orchestrator stops routing
traffic instead of restart-looping an engine that still answers `/health`.

---

## Auth

### `POST /auth/token`

Mint a session token for the caller. **Requires Basic auth** — a
bearer-authenticated caller gets `403`, so a token cannot extend its own
lifetime without paying the Argon2 cost. No request body.

The token inherits the user's scope *at mint time*; a later scope change does
not affect tokens already issued. TTL is 1 hour, not configurable.

```json
{ "token": "…43 chars…", "token_type": "Bearer", "expires_at": 1767225600 }
```

### `POST /auth/token/revoke`

Revoke the bearer token that authenticated this request. No body.

```json
{ "revoked": true }
```

`400` when the request used Basic auth (there is no token to revoke).

---

## Databases

### `GET /db`

Root only. Lists every database.

```json
{ "databases": [
  { "name": "default", "created_at": 1767225600, "dimension": 128,
    "map_size_mb": 4096, "collections": 3 }
] }
```

Half-initialised directories (no readable `db.toml`) are skipped.

### `POST /db`

Root only. Creates a database.

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
  -d '{"name":"movies","dimension":384,"map_size_mb":32768}'
```

Responds with the same `DatabaseInfo` shape as `GET /db/{db}`. `409` if the
name exists, `400` if the name or the merged EAM config is invalid.

### `GET /db/{db}`

```json
{ "name": "movies", "created_at": 1767225600, "dimension": 384,
  "map_size_mb": 32768, "collections": 0 }
```

### `DELETE /db/{db}`

Root only. Moves `$DATA/db/{db}/` to `$DATA/_trash/{db}-<unix-seconds>/`
rather than deleting it — recover by moving the directory back and restarting.

```json
{ "dropped": true }
```

`{"dropped": false}` when the database was not mounted. `403` for
`default`, which cannot be dropped (the legacy routes target it).

### `POST /db/{db}/dream`

Trigger idle-time consolidation on demand, ignoring the idle gate and the
`dream.enabled` opt-in. Query parameters override the persisted `[dream]`
config for this call only:

| Param | Type | Meaning |
|---|---|---|
| `mode` | `replay` \| `ladder` | Reorganise in place, or build `<name>__L1`, `__L2`… derived collections |
| `levels` | integer | Replay: granularity passes. Ladder: levels to climb |
| `tau_cohere` | float | Ladder: minimum counter coherence to join a family |
| `tau_split` | float | Ladder: minimum address similarity for a candidate family |
| `collections` | comma-separated string | Restrict to these collections. Default: all |

```json
{ "database": "movies",
  "collections": [
    { "collection": "taste", "passes": 1, "locations_before": 812,
      "locations_after": 806, "merged": 6, "tau_split": 0.0, "tau_cohere": 0.0 }
  ] }
```

See [Dreaming](../explanation/dreaming.md).

### `GET /db/{db}/audit`

The access log, newest first. Root only by default; a database can opt its own
users in with `visibility = "DbUsers"` in `db.toml`.

| Param | Type | Default | Meaning |
|---|---|---|---|
| `user` | string | — | Exact username match |
| `collection` | string | — | Exact collection match |
| `route` | string | — | Logical route: `read`, `attention`, `analyze`, `documents/query` |
| `since` | integer | — | Inclusive lower bound, unix **milliseconds** |
| `until` | integer | — | Inclusive upper bound, unix milliseconds |
| `limit` | integer | `100` | Clamped to `1..=1000` |

```json
{ "database": "movies", "count": 2, "limit": 100,
  "entries": [
    { "seq": 41, "timestamp_ms": 1767225600123, "user": "alice", "scope": "db:movies",
      "database": "movies", "collection": "taste", "route": "read", "status": 200,
      "result_count": 1, "location_ids": [], "document_ids": [],
      "query_hash": "9f2c1ab30de4f781", "query_raw": null }
  ] }
```

`409` when audit logging is disabled for that database. `query_raw` is `null`
unless the database sets `store_raw_query = true`. Only reads are recorded —
`read`, `attention`, `analyze` and `documents/query`. See
[The audit log and privacy](../explanation/audit-and-privacy.md).

---

## Collections

### `POST …/collections`

```json
{ "name": "taste" }
```
→
```json
{ "name": "taste", "created": true }
```

`created` is `false` when the collection already existed; the call is
idempotent either way.

### `GET …/collections`

```json
{ "collections": ["taste", "history"] }
```

### `DELETE …/collections/{name}`

```json
{ "dropped": true }
```

### `GET …/collections/{name}/stats`

```json
{ "num_locations": 812, "total_writes": 10450.0, "current_eta": 0.0031,
  "avg_write_count": 12.87, "max_write_count": 96.0 }
```

`num_locations` is a diagnostic of data complexity, not a tunable.
`current_eta` is the decayed learning rate. 404 if the collection does not exist.

### `GET …/collections/{name}/config`

The effective `EAMConfig` for this collection. See
[Configuration](configuration.md#eam-knobs) for every field.

```json
{ "config": { "d": 128, "l_0": 0, "k": 20, "eta_0": 0.01, "lambda": 0.9999,
  "eta_min": 0.001, "tau_split": 0.3, "tau_merge": 0.95, "gamma": 1.0,
  "tau_damp": 10.0, "tau_overload": 8.0, "beta": 5.0, "t_max": 10,
  "epsilon": 1e-6, "neighbor_cap": 32, "num_landmarks": 32,
  "mdl_gate": false, "competitive": true } }
```

### `GET …/collections/{name}/locations`

| Param | Type | Default |
|---|---|---|
| `full` | boolean | `false` |

Summary form (`full=false`):

```json
{ "locations": [ { "id": 0, "write_count": 12.0, "avg_counter_magnitude": 0.041 } ] }
```

Full form (`full=true`) replaces `avg_counter_magnitude` with the raw vectors —
`L × 2D` floats, so it is large:

```json
{ "locations": [ { "id": 0, "write_count": 12.0,
                   "address": [ … ], "counter": [ … ] } ] }
```

### `POST …/collections/{name}/compress`

Merge locations while it lowers the collection's description length, stopping
at the MDL minimum. Both fields optional.

| Field | Type | Default |
|---|---|---|
| `kappa` | float | the collection's dimension — model cost per location, in bits |
| `lambda` | float | `1.0` — weight on the data-fit (variance) cost of a merge |

```json
{ "locations_before": 812, "locations_after": 640,
  "description_length_before": 91234.5, "description_length_after": 88012.1,
  "merges": 172 }
```

### `GET …/collections/{name}/fingerprint`

The collection's emergent self-summary: the write-count-weighted centroid of
every hard location's normalised pattern, refined by an iterative Hopfield read
so it lands on a real attractor rather than a bare average.

```json
{ "fingerprint": [0.13, -0.07, 0.26] }
```

`{"fingerprint": null}` when nothing has been written.

---

## Writes

### `POST …/collections/{name}/write`

Creates the collection if it does not exist.

| Field | Type | Notes |
|---|---|---|
| `vectors` | array of arrays | Required, non-empty. Each must match the database's dimension |
| `metadata` | array of JSON values | Optional. Length must equal `vectors` |

Without metadata the vectors are written as one batch (single LMDB
transaction) and the response is just a count:

```json
{ "count": 2 }
```

With metadata each vector becomes a retrievable **document** and the response
also carries the assigned ids:

```json
{ "count": 2, "ids": [17, 18] }
```

`400` on an empty `vectors` array, a dimension mismatch, or a
`metadata`/`vectors` length mismatch.

### `POST …/collections/{name}/bulk_load`

Replace a collection's hard locations atomically with a caller-supplied set.
Bypasses competitive learning entirely: no activation, no novelty splits, no
address migration. This is what makes snapshot algebra operate on known
operands.

| Field | Type | Notes |
|---|---|---|
| `addresses` | array of arrays | Required, non-empty. Each of length `d` |
| `counters` | array of arrays | Required. Same length as `addresses`, each of length `d` |
| `write_counts` | array of floats | Optional. Same length as `addresses`. Defaults to `1.0` each |

```json
{ "n_loaded": 4096, "dim": 128 }
```

The collection's existing config (dimension, EAM knobs) is preserved. The
replace is version-checked: if a concurrent write lands first the call returns
`409` and nothing is changed — retry. `400` when a length or dimension does not
line up, or when `addresses` exceeds `--max-bulk-items` (default 100 000).

---

## Reads

All four record an audit entry when logging is enabled for the database.

### `POST …/collections/{name}/read`

| Field | Type | Default |
|---|---|---|
| `query` | array | *required* |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |

`iterative` runs the Hopfield loop to convergence (up to `t_max`); `fast` is a
single step.

```json
{ "result": [0.1, 0.3, 0.5, 0.2] }
```

`400` on an unknown strategy or a dimension mismatch.

### `POST …/collections/{name}/attention`

Raw dot-product attention at a temperature you supply.

| Field | Type | Notes |
|---|---|---|
| `query` | array | *required* |
| `scale` | float | *required* — inverse temperature β |
| `exclude_id` | integer | Optional. Drops one stored location (leave-one-out) |

```json
{ "result": [ … ], "beta": 12.5,
  "contributors": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ] }
```

`contributors` is ordered by descending weight. `similarity` here is the raw
dot product `Q·Kᵢ`, **not** a cosine — unlike the identically-named field in
`analyze`.

### `POST …/collections/{name}/attention/mdl`

The same read, except β is not supplied: the engine picks it per query by
minimising the description length of the activated key set (leave-one-out KDE
bandwidth selection over a log grid in `[0.1, 500]`). Only `query` is read from
the body; a stray `scale` is ignored.

```json
{ "result": [ … ], "beta": 37.2, "entropy": 0.41,
  "contributors": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ] }
```

`entropy` is the normalised Shannon entropy of the weights, `-Σ w·ln w / ln n`,
in `0..1` — `0` for fewer than two contributors. It is **reported, never acted
on**: attention entropy is a weak abstention signal on its own (measured AUC
0.61–0.76, against 0.997 for the top-1 match margin), so the threshold, if any,
is the caller's.

### `POST …/collections/{name}/analyze`

A `read` with the full activation trace attached — the "why this?" endpoint.

| Field | Type | Default |
|---|---|---|
| `query` | array | *required* |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |

```json
{ "iterations": 4, "converged": true,
  "activated_locations": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ],
  "result": [ … ] }
```

Here `similarity` **is** a cosine.

### `POST …/collections/{name}/batch_analyze`

Many `analyze` calls in one request, run in parallel.

| Field | Type | Default |
|---|---|---|
| `queries` | array of arrays | *required*, non-empty |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |

```json
{ "results": [ { "iterations": 4, "converged": true,
                 "activated_locations": [ … ], "result": [ … ] } ] }
```

`400` when `queries` exceeds `--max-batch-queries` (default 1 000). Not audited —
use `analyze` when you need the log.

### `POST …/collections/{name}/attention/calibrate`

Pick an attention temperature for the whole collection by leave-one-out
value-reconstruction description length.

| Field | Type | Default |
|---|---|---|
| `betas` | array of floats | 81 log-spaced points across `[0.5, 200]` |

```json
{ "beta": 42.6, "dl": 1183.4 }
```

---

## Documents

A document is a vector written with metadata. It keeps its own id and is
retrievable and searchable independently of the hard locations that absorbed it.

### `GET …/collections/{name}/documents`

```json
{ "documents": [ { "id": 17, "metadata": { "title": "Solaris" } } ] }
```

Documents whose metadata does not parse as JSON are skipped.

### `GET …/collections/{name}/documents/{doc_id}`

```json
{ "id": 17, "metadata": { "title": "Solaris" } }
```

`404` if the document does not exist. Unparseable metadata comes back as `null`.

### `DELETE …/collections/{name}/documents/{doc_id}`

```json
{ "deleted": true }
```

Deletion is idempotent: a missing document is `200 {"deleted": false}`, not a
404, so a tombstone replayed after a crash does not fail.

This removes the document and every posting-list reference to it, so it can no
longer be retrieved or cited. It does **not** subtract its contribution from
merged engrams — superposition cannot do that.

### `POST …/collections/{name}/documents/query`

Similarity search over documents, using the hard-location posting list as the
index: the query activates locations, candidate document ids are gathered from
their posting lists, and exact cosine is computed on the candidates.

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | array | *required* | Also the recall vector in every mode |
| `n` | integer | `10` | Results returned |
| `unbind_role` | array | — | Single-role scoring: unbind each candidate by this role and compare the recovered filler to `query` |
| `role_pairs` | array of `{role, filler}` | `[]` | Multi-role scoring. Mutually exclusive with `unbind_role` |
| `cleanup` | `"mdl"` \| `"off"` \| `{"beta": N}` | `"mdl"` | Cleanup of the recovered filler. `role_pairs` only |

```json
{ "results": [ { "id": 17, "similarity": 0.88, "metadata": { "title": "Solaris" } } ] }
```

With `role_pairs`, each hit also carries `role_scores` — one similarity per
requested pair, in request order, unweighted and unaggregated — and the
response carries the temperature cleanup actually ran at:

```json
{ "results": [ { "id": 17, "similarity": 0.88, "metadata": { … },
                 "role_scores": [0.71, 0.34] } ],
  "cleanup_beta": 37.2 }
```

Both extra fields are omitted when not applicable, so a single-role response is
byte-identical to what pre-multi-role clients parse.

**Recall is not re-ranked by the pairs.** Candidate retrieval always uses
`query` and plain full-bundle cosine; `role_pairs` only score the set the index
already produced. A document that would rank well under your weighting can fail
to be recalled at all — ask for an `n` far larger than you intend to display.

Betas below the engine floor (30.0) are raised to it. See
[Store and query structured documents](../how-to/structured-documents.md) for
the sharpness cliff this guards against. `400` when `unbind_role` and
`role_pairs` are both present.

---

## Vector algebra on raw vectors

Stateless: no collection, no database, no persistence. **Root scope only.**

### `POST /vec/bind`

Circular-convolution bind of two vectors.

| Field | Type | Default |
|---|---|---|
| `a`, `b` | arrays of equal length | *required* |
| `normalize` | boolean | `true` |

`normalize: false` returns the raw convolution, for spiral-plane ops where
spectrum magnitude carries growth or decay.

```json
{ "result": [ … ] }
```

### `POST /vec/unbind`

Unbind `a` from `b`.

| Field | Type | Default |
|---|---|---|
| `a`, `b` | arrays of equal length | *required* |
| `exact` | boolean | `false` |
| `eps` | float | `1e-9` |

Default is circular correlation — the approximate inverse, exact on
unit-spectrum keys. `exact: true` does spectral division, the true inverse for
keys with non-unit spectrum magnitudes; `eps` zeroes null bins.

### `POST /vec/bundle`

Weighted superposition, `Σ weightᵢ · vectorᵢ`.

| Field | Type | Default |
|---|---|---|
| `terms` | array of `{vector, weight}` | *required*. `weight` defaults to `1.0` |
| `normalize` | boolean | `true` |

Weights may be negative — that is subtraction, not an error.

### `POST /vec/pow`

Spectral power `a^⊗t` for real `t`. Integer `t` equals `t` successive binds;
fractional `t` is fractional power encoding. Never normalised.

| Field | Type |
|---|---|
| `a` | array, non-empty |
| `t` | float |

### `POST /vec/rotate`

Continuous permutation-cycle rotation `ρ^t` — the "dimmer dial" generalisation
of `/algebra/permute`'s integer `power`. `t=0` is the identity, `t=1` exactly
reproduces the discrete permutation, integer `t` reproduces `permute_pow`.

| Field | Type | Notes |
|---|---|---|
| `a` | array, non-empty | |
| `seed` | integer | Mutually exclusive with `name` |
| `name` | string | Hashed (SHA-256) to a seed. Mutually exclusive with `seed` |
| `t` | float | |

Supplying neither or both of `seed`/`name` is a `400`. Exact isometry on
odd-length permutation cycles.

---

## Vector algebra on collections

These read one or two source collections, compute, and write the result into a
target collection, creating it if needed. Sources must exist (`404` otherwise).
All respond with:

```json
{ "collection": "target_name", "num_locations": 4096 }
```

The target write is version-checked (except `permute`): a concurrent write to
the target returns `409`, retryable.

### `POST …/algebra/add` · `POST …/algebra/sub` · `POST …/algebra/bind`

| Field | Type | Default |
|---|---|---|
| `source_a`, `source_b`, `target` | string | *required* |
| `max_cross_k` | integer | `0` = full cartesian product |

These form an n×m cross product. The server refuses before allocating when the
estimated result (`n_a × n_b`, or `n_a × max_cross_k` when limited) exceeds
`--max-algebra-locations` (default 250 000), with a `400` naming both numbers.

### `POST …/algebra/intersect`

| Field | Type | Default |
|---|---|---|
| `source_a`, `source_b`, `target` | string | *required* |
| `threshold` | float | `0.95` |

A non-positive `threshold` admits every cross pair, so the cross-product cap
applies in that case.

### `POST …/algebra/scale`

| Field | Type |
|---|---|
| `source`, `target` | string |
| `alpha` | float |

### `POST …/algebra/permute`

Apply `ρ^k` to every location of `source`.

| Field | Type | Default |
|---|---|---|
| `source`, `target` | string | *required* |
| `seed` | integer | mutually exclusive with `name` |
| `name` | string | hashed (SHA-256) to a seed |
| `power` | integer | `1`. Negative applies `ρ⁻¹` |

Exactly one of `seed`/`name` is required.

### `POST …/algebra/unbind`

| Field | Type | Notes |
|---|---|---|
| `source` | string | The bundle to unbind |
| `key_vector` | array | The binding key, as a raw vector of length `d` |
| `target` | string | |

The key is a raw vector rather than a collection because the adaptive memory
means collections rarely hold exactly one location — clients hold binding keys
as plain vectors.

### `POST …/compose/read`

Route one query across several collections at once and blend the answers.

| Field | Type | Default |
|---|---|---|
| `collections` | array of names | *required*, at least 2 |
| `query` | array | *required* |
| `routing_sharpness` | float | `20.0` |

```json
{ "result": [ … ],
  "weights":     { "movies": 0.81, "books": 0.19 },
  "confidences": { "movies": 0.93, "books": 0.44 } }
```
