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

`iterative` re-weights the query against the activated addresses until it
settles (up to `t_max` passes), then reads the stored values once at the
settled point. `fast` is a single step. Use `fast` for lookups and anything
that feeds a confidence decision, and `iterative` to recover an item from a
heavily corrupted query. See [choose a read
strategy](../how-to/choose-a-read-strategy.md).

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

Here `similarity` **is** a cosine. With `iterative`, `similarity` and
`weight` are taken at the settled point, so they look alike for every query;
use `"strategy": "fast"` when the weights feed a confidence decision.

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

### POST /db/{db}/collections/{name}/recall

Recall with a selective-attention head. The database must have been created
with `selective_attention` naming the head. Legacy:
`POST /collections/{name}/recall`. Audited.

| Field | Type | Default |
|---|---|---|
| `query` | array | *required*. Values in the head's own fields are ignored |
| `head` | string | *required*. A head named at database creation |
| `contributors` | integer | `10`. How many contributing rows to list |

The stored rows are scored by attention-weighted squared distance over the
fields the head does not fill in, `Σ_d α_d (q_d − x_d)²`, with the head's
learned `α`. The nearest `k` (`selective_attention.k`) are blended by
softmax of the negated distance, and the blend is returned with the head's
fields filled in. Before the head has warmed up, attention is uniform.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/crm/collections/customers/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.4,-1.2,0.0,0.0],"head":"outcome","contributors":3}'
```

```json
{ "result": [0.38, -1.05, 0.21, 0.79],
  "contributors": [ { "id": 812, "distance": 0.04, "weight": 0.11 } ] }
```

`contributors` are nearest first; over all blended rows the weights sum
to 1.

| Status | Cause |
|---|---|
| `400 Bad Request` | the database has no selective attention, the head is unknown, or a dimension mismatch |
| `404 Not Found` | no such collection |

### GET /db/{db}/collections/{name}/next

The time-attended read of a collection's history: its rows weighted by the
database's learned per-age attention, newest first, summed and
unit-normalised. What the memory expects next. Legacy:
`GET /collections/{name}/next`.

```bash
curl -u admin:pw http://localhost:6380/db/movies/collections/user_42/next
```

```json
{ "result": [0.02, -0.11, 0.07] }
```

`{"result": null}` for an empty collection.

| Status | Cause |
|---|---|
| `400 Bad Request` | the database has no selective attention, or was created without `time` |
| `404 Not Found` | no such collection |

### POST /db/{db}/collections/{name}/novelty

How new each query is, relative to what is normal around it in memory.
About `1` means it fits its surroundings; higher means it stands out from
them. Novelty is relative, not absolute: the same distance can be ordinary
in a sparse region and alarming in a dense one. It is the default way to
ask "is this an anomaly?", "is this new?", "should I abstain?" or "has
normal moved?". Legacy: `POST /collections/{name}/novelty`. Audited when
`nearest` asks for stored rows back.

| Field | Type | Default |
|---|---|---|
| `queries` | array of arrays | *required*, non-empty, at most `--max-batch-queries` |
| `k` | integer | `50`. Neighbours compared |
| `nearest` | integer | `0`. How many nearest stored rows to return per query |

How it is computed today: the local outlier factor over the collection's
rows. Every stored row gets a *spread*, the distance to its `k`-th nearest
other row. A query's novelty is the mean familiarity of its `k` nearest
rows divided by its own, where familiarity is the inverse mean distance to
those neighbours, each distance floored at the neighbour's spread.
Distances are Euclidean over the stored rows, so store standardised
features.

Neighbours are found by the exact int8 scan (`Euclidean` metric), which
reads only a few dozen candidate rows back from storage in shadow
residency. The spreads are computed on the first call for a given `k`, by an
exact blocked all-pairs search, and kept in memory. After that, writes that
only append rows (plain collections) update them in place, so later calls
never rebuild; any other change (a bulk load, a delete, a merge) leaves
them stale and the next call rebuilds. Measured on credit-card fraud rows
(29 fields, `k` 50):

| Rows stored | First call | Each later query | Each appended row | Peak memory |
|---|---|---|---|---|
| 50,000 | 4 s | 0.2 ms | 0.6 ms | 200 MB |
| 284,315 | 152 s | 0.6 ms | 3.1 ms | 1.0 GB |

The first call grows with the square of the rows (one search per row), and
the kept state costs about 1.3 KB per row at 29 fields: the rows themselves,
each row's neighbour list, and who lists it. While the state exists, every
appended row costs one distance per stored row.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/risk/collections/normal_card_activity/novelty \
  -H 'Content-Type: application/json' \
  -d '{"queries":[[0.1,-0.3,0.2],[4.1,3.8,-5.0]],"k":50,"nearest":1}'
```

```json
{ "results": [ { "novelty": 1.02, "nearest": [ { "id": 812, "distance": 0.31 } ] },
               { "novelty": 7.9,  "nearest": [ { "id": 40,  "distance": 6.2 } ] } ] }
```

| Status | Cause |
|---|---|
| `400 Bad Request` | empty `queries`, more than `--max-batch-queries`, `k` of 0, fewer than 2 stored rows, or a dimension mismatch |
| `404 Not Found` | no such collection |

