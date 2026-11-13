# Read endpoints

Recall from a collection. Six routes: the plain Hopfield `read`, three
attention variants, `analyze` (a read with its activation trace), and
`batch_analyze`.

Every route here has a legacy form without the `/db/{db}` prefix that targets
the `default` database. Auth for all of them: Basic or Bearer, root or
`Database(db)` (legacy form: root or `Database("default")`). All **create the
collection if it does not exist** — a read against an unknown name returns an
empty result, not a 404. See [Overview](overview.md).

`read`, `attention` and `analyze` record an audit entry when logging is
enabled for the database (as does
[`documents/query`](documents.md#post-dbdbcollectionsnamedocumentsquery)).
`attention/mdl`, `attention/calibrate` and `batch_analyze` do not.

### POST /db/{db}/collections/{name}/read

Reconstruct a pattern associated with the query. Legacy:
`POST /collections/{name}/read`.

| Field | Type | Default |
|---|---|---|
| `query` | array | *required* |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |
| `diagnostics` | bool | `false` |

`iterative` runs the Hopfield loop to convergence (up to `t_max`); `fast` is a
single step.

The result is a reconstructed vector. In competitive mode it can complete an
individual pattern or express shared structure learned from related writes.
Novel writes receive separate locations; joining writes can combine observations.
Plain mode keeps each vector as a separate location, but attention can still
blend them. Read strategy, temperature and data overlap affect reconstruction.

For document identity, use metadata-bearing writes and the document API.
`documents/{id}` returns the ID and metadata; [`documents/query`](documents.md#post-dbdbcollectionsnamedocumentsquery)
returns similarity-ranked document results. Neither requires a different write
mode. See [Choose a memory mode](../how-to/choose-memory-mode.md).

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/read \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"strategy":"iterative"}'
```

```json
{ "result": [0.1, 0.3, 0.5, 0.2] }
```

#### Diagnostics

With `"diagnostics": true` the response gains a `diagnostics` object saying
how far the result can be trusted. Without it, or with `false`, the response is
exactly `{ "result": [...] }` as above.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/read \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"diagnostics":true}'
```

```json
{ "result": [0.1, 0.3, 0.5, 0.2],
  "diagnostics": { "top_weight": 0.84, "margin": 0.82, "entropy": 0.29,
                   "support": 3.0, "fidelity": 0.998 } }
```

| Field | Meaning |
|---|---|
| `top_weight` | Largest activation weight. |
| `margin` | Largest weight minus the second largest. |
| `entropy` | Normalised entropy of the weights: 0 when one location takes everything, 1 when they are uniform. Same measure as [`attention/mdl`](#post-dbdbcollectionsnameattentionmdl), over these weights. |
| `support` | `write_count` of the location with the largest weight: how many writes it holds. Always `1` in plain mode (`"competitive": false`), which never merges. Compare it with `capacity` from [`stats`](collections.md#get-dbdbcollectionsnamestats). |
| `fidelity` | Cosine between the query and the `fast` result. |

All five come from **one `fast` step on the query, whatever `strategy` the
read itself uses**, the same numbers as
[`analyze`](#post-dbdbcollectionsnameanalyze) with `"strategy": "fast"`. An
iterative read always converges onto *some* stored pattern, so its final
weights look the same for queries the collection knows and queries it has
never seen. The first step still reflects the query. In a test with 60 stored
random vectors (`d = 256`), stored queries scored `top_weight` 0.83–0.86 and
`fidelity` 0.999. Never-stored queries scored 0.06–0.12 and 0.21–0.42. Overfilled
competitive collections (5× capacity per cluster) scored `top_weight` ≤ 0.12,
`entropy` ≥ 0.75 and `fidelity` ≤ 0.80.

The endpoint reports numbers and makes no decision. Where they separate
depends on the data and on how full the collection is, so derive any
threshold from your own data at your collection's load. See
[Abstention](../explanation/abstention.md).

The diagnostics are computed after the read, under a separate lock. A write
landing in between can make them describe a slightly newer state than
`result`.

| Status | Cause |
|---|---|
| `400 Bad Request` | unknown `strategy`, or a dimension mismatch |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/attention

Raw dot-product attention at a temperature you supply. Legacy:
`POST /collections/{name}/attention`.

| Field | Type | Notes |
|---|---|---|
| `query` | array | *required* |
| `scale` | float | *required* — inverse temperature β |
| `exclude_id` | integer | Optional. Drops one stored location (leave-one-out) |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/attention \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"scale":12.5}'
```

```json
{ "result": [ … ], "beta": 12.5,
  "contributors": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ] }
```

`contributors` is ordered by descending weight. `similarity` here is the raw
dot product `Q·Kᵢ`, **not** a cosine — unlike the identically-named field in
`analyze`.

| Status | Cause |
|---|---|
| `400 Bad Request` | dimension mismatch, or an engine validation error |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/attention/mdl

The same read, except β is not supplied: the engine picks it per query by
minimising the description length of the activated key set (leave-one-out KDE
bandwidth selection over a log grid in `[0.1, 500]`). Legacy:
`POST /collections/{name}/attention/mdl`.

| Field | Type | Notes |
|---|---|---|
| `query` | array | *required*. The only field read — a stray `scale` is ignored |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/attention/mdl \
  -H 'Content-Type: application/json' -d '{"query":[0.1,0.2,0.3]}'
```

```json
{ "result": [ … ], "beta": 37.2, "entropy": 0.41,
  "contributors": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ] }
```

`entropy` is the normalised Shannon entropy of the weights,
`-Σ w·ln w / ln n`, in `0..1` — `0` for fewer than two contributors. It is
**reported, never acted on**: attention entropy is a weak abstention signal on
its own (measured AUC 0.61–0.76, against 0.997 for the top-1 match margin), so
the threshold, if any, is the caller's.

| Status | Cause |
|---|---|
| `400 Bad Request` | dimension mismatch, or an engine validation error |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/attention/calibrate

Pick an attention temperature for the whole collection by leave-one-out
value-reconstruction description length. Legacy:
`POST /collections/{name}/attention/calibrate`.

| Field | Type | Default |
|---|---|---|
| `betas` | array of floats | 81 log-spaced points across `[0.5, 200]` |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/attention/calibrate \
  -H 'Content-Type: application/json' -d '{"betas":[1.0,10.0,100.0]}'
```

```json
{ "beta": 42.6, "dl": 1183.4 }
```

Feed the result back as `scale` on
[`attention`](#post-dbdbcollectionsnameattention).

| Status | Cause |
|---|---|
| `400 Bad Request` | an empty or invalid `betas` sweep, or too few locations to calibrate |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/analyze

A `read` with the full activation trace attached — the "why this?" endpoint.
Legacy: `POST /collections/{name}/analyze`.

| Field | Type | Default |
|---|---|---|
| `query` | array | *required* |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/analyze \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"strategy":"iterative"}'
```

```json
{ "iterations": 4, "converged": true,
  "activated_locations": [ { "id": 7, "similarity": 0.91, "weight": 0.63 } ],
  "result": [ … ] }
```

Here `similarity` **is** a cosine.

| Status | Cause |
|---|---|
| `400 Bad Request` | unknown `strategy`, or a dimension mismatch |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/batch_analyze

Many `analyze` calls in one request, run in parallel. Legacy:
`POST /collections/{name}/batch_analyze`.

| Field | Type | Default |
|---|---|---|
| `queries` | array of arrays | *required*, non-empty |
| `strategy` | `"iterative"` \| `"fast"` | `"iterative"` |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/batch_analyze \
  -H 'Content-Type: application/json' \
  -d '{"queries":[[0.1,0.2,0.3],[0.9,0.0,0.1]],"strategy":"fast"}'
```

```json
{ "results": [ { "iterations": 4, "converged": true,
                 "activated_locations": [ … ], "result": [ … ] } ] }
```

Results are in request order. Not audited — use `analyze` when you need the
log.

| Status | Cause |
|---|---|
| `400 Bad Request` | empty `queries`, more than `--max-batch-queries` (default 1 000), unknown `strategy`, or a dimension mismatch |
| `413 Payload Too Large` | body over `--max-body-size` |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |
