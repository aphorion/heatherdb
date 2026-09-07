# Algebra endpoints

Vector algebra over whole collections. Each op reads one or two source
collections, computes, and writes the result into a target collection,
creating the target if needed. Sources must exist (`404` otherwise).

Every route here has a legacy form without the `/db/{db}` prefix that targets
the `default` database. Auth for all of them: Basic or Bearer, root or
`Database(db)` (legacy form: root or `Database("default")`). See
[Overview](overview.md) and [Vector algebra](../explanation/vector-algebra.md).
For stateless math on raw vectors, see [Vectors](vectors.md).

All the `/algebra/*` ops answer with:

```json
{ "collection": "target_name", "num_locations": 4096 }
```

The target write is version-checked (except `permute`): a concurrent write to
the target returns `409`, retryable.

`add`, `sub` and `bind` form an n×m cross product. The server refuses before
allocating when the estimated result (`n_a × n_b`, or `n_a × max_cross_k` when
limited) exceeds `--max-algebra-locations` (default 250 000), with a `400`
naming both numbers.

### POST /db/{db}/algebra/add

Superpose every pair of locations from two collections. Legacy:
`POST /algebra/add`.

| Field | Type | Default |
|---|---|---|
| `source_a`, `source_b`, `target` | string | *required* |
| `max_cross_k` | integer | `0` = full cartesian product |

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/algebra/add \
  -H 'Content-Type: application/json' \
  -d '{"source_a":"taste","source_b":"history","target":"blended",
       "max_cross_k":8}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | the estimated cross product exceeds the cap; a dimension mismatch between sources |
| `404 Not Found` | `source_a` or `source_b` does not exist |
| `409 Conflict` | a concurrent write to `target` — retryable |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/algebra/sub

The same cross product, subtracting `source_b`'s locations. Legacy:
`POST /algebra/sub`. Same fields, same statuses as
[`algebra/add`](#post-dbdbalgebraadd).

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/algebra/sub \
  -H 'Content-Type: application/json' \
  -d '{"source_a":"taste","source_b":"dislikes","target":"refined"}'
```

### POST /db/{db}/algebra/bind

Circular-convolution bind of every pair. Legacy: `POST /algebra/bind`. Same
fields, same statuses as [`algebra/add`](#post-dbdbalgebraadd).

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/algebra/bind \
  -H 'Content-Type: application/json' \
  -d '{"source_a":"roles","source_b":"fillers","target":"bound"}'
```

### POST /db/{db}/algebra/intersect

Keep only the cross pairs whose similarity clears a threshold. Legacy:
`POST /algebra/intersect`.

| Field | Type | Default |
|---|---|---|
| `source_a`, `source_b`, `target` | string | *required* |
| `threshold` | float | `0.95` |

A non-positive `threshold` admits every cross pair, so the cross-product cap
is applied in that case only.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/algebra/intersect \
  -H 'Content-Type: application/json' \
  -d '{"source_a":"taste","source_b":"history","target":"shared",
       "threshold":0.9}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | a non-positive `threshold` whose cross product exceeds the cap; a dimension mismatch |
| `404 Not Found` | `source_a` or `source_b` does not exist |
| `409 Conflict` | a concurrent write to `target` — retryable |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/algebra/scale

Multiply every location of `source` by `alpha`. Legacy:
`POST /algebra/scale`.

| Field | Type | Notes |
|---|---|---|
| `source`, `target` | string | *required* |
| `alpha` | float | *required*. Negative flips the sign |

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/algebra/scale \
  -H 'Content-Type: application/json' \
  -d '{"source":"taste","target":"faded","alpha":0.5}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | an engine validation error |
| `404 Not Found` | `source` does not exist |
| `409 Conflict` | a concurrent write to `target` — retryable |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/algebra/permute

Apply `ρ^k` to every location of `source`. Legacy: `POST /algebra/permute`.

| Field | Type | Default |
|---|---|---|
| `source`, `target` | string | *required* |
| `seed` | integer | mutually exclusive with `name` |
| `name` | string | hashed (SHA-256) to a seed |
| `power` | integer | `1`. Negative applies `ρ⁻¹` |

Exactly one of `seed`/`name` is required. Unlike the other ops, the target
write is **not** version-checked, so this never returns `409`.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/algebra/permute \
  -H 'Content-Type: application/json' \
  -d '{"source":"taste","target":"shifted","name":"sequence","power":1}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | neither or both of `seed`/`name`; an engine validation error |
| `404 Not Found` | `source` does not exist |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/algebra/unbind

Unbind a raw key vector out of every location of `source`. Legacy:
`POST /algebra/unbind`.

| Field | Type | Notes |
|---|---|---|
| `source` | string | The bundle to unbind |
| `key_vector` | array | The binding key, as a raw vector of length `d` |
| `target` | string | |

The key is a raw vector rather than a collection because the adaptive memory
means collections rarely hold exactly one location — clients hold binding keys
as plain vectors.

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/algebra/unbind \
  -H 'Content-Type: application/json' \
  -d '{"source":"bound","key_vector":[0.7,0.1,0.7],"target":"fillers"}'
```

| Status | Cause |
|---|---|
| `400 Bad Request` | `key_vector` is not of length `d` |
| `404 Not Found` | `source` does not exist |
| `409 Conflict` | a concurrent write to `target` — retryable |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/compose/read

Route one query across several collections at once and blend the answers.
Legacy: `POST /compose/read`. Unlike the algebra ops this writes nothing.

| Field | Type | Default |
|---|---|---|
| `collections` | array of names | *required*, at least 2 |
| `query` | array | *required* |
| `routing_sharpness` | float | `20.0` |

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/compose/read \
  -H 'Content-Type: application/json' \
  -d '{"collections":["movies","books"],"query":[0.1,0.2,0.3],
       "routing_sharpness":20.0}'
```

```json
{ "result": [ … ],
  "weights":     { "movies": 0.81, "books": 0.19 },
  "confidences": { "movies": 0.93, "books": 0.44 } }
```

| Status | Cause |
|---|---|
| `400 Bad Request` | fewer than 2 collections; a dimension mismatch |
| `404 Not Found` | one of the named collections does not exist |
| `500 Internal Server Error` | storage failure |
