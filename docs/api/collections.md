# Collection endpoints

Lifecycle and introspection of collections. A collection is one Adaptive
Elastic Associative Memory inside a database; it inherits the database's
dimension.

Every route here has a legacy form without the `/db/{db}` prefix that targets
the `default` database. Auth for all of them: Basic or Bearer, root or
`Database(db)` (legacy form: root or `Database("default")`). See
[Overview](overview.md).

### POST /db/{db}/collections

Create a collection. Idempotent. Legacy: `POST /collections`.

| Field | Type | Notes |
|---|---|---|
| `name` | string | *required* |

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/collections \
  -H 'Content-Type: application/json' -d '{"name":"taste"}'
```

```json
{ "name": "taste", "created": true }
```

`created` is `false` when the collection already existed.

| Status | Cause |
|---|---|
| `400 Bad Request` | the engine rejected the name, or the collection could not be created |

### GET /db/{db}/collections

List collection names. Legacy: `GET /collections`.

Request body: no body.

```bash
curl -u admin:pw http://localhost:6380/db/movies/collections
```

```json
{ "collections": ["taste", "history"] }
```

| Status | Cause |
|---|---|
| `500 Internal Server Error` | the registry could not be read |

### DELETE /db/{db}/collections/{name}

Drop a collection and everything in it. Legacy:
`DELETE /collections/{name}`.

Request body: no body.

```bash
curl -u admin:pw -X DELETE \
  http://localhost:6380/db/movies/collections/taste
```

```json
{ "dropped": true }
```

`{"dropped": false}` when there was no such collection — dropping is
idempotent, not a 404.

| Status | Cause |
|---|---|
| `500 Internal Server Error` | storage failure while removing the collection |

### GET /db/{db}/collections/{name}/stats

Counters for the memory. Legacy: `GET /collections/{name}/stats`.

Request body: no body.

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/stats
```

```json
{ "num_locations": 812, "total_writes": 10450.0, "current_eta": 0.0031,
  "avg_write_count": 12.87, "max_write_count": 96.0 }
```

`num_locations` is a diagnostic of data complexity, not a tunable.
`current_eta` is the decayed learning rate.

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection — this route does not create one |
| `500 Internal Server Error` | storage failure |

### GET /db/{db}/collections/{name}/config

The effective `EAMConfig` for this collection. Legacy:
`GET /collections/{name}/config`. See
[Configuration](../reference/configuration.md#eam-knobs) for every field.

Request body: no body.

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/config
```

```json
{ "config": { "d": 128, "l_0": 0, "k": 20, "eta_0": 0.01, "lambda": 0.9999,
  "eta_min": 0.001, "tau_split": 0.3, "tau_merge": 0.95, "gamma": 1.0,
  "tau_damp": 10.0, "tau_overload": 8.0, "beta": 5.0, "t_max": 10,
  "epsilon": 1e-6, "neighbor_cap": 32, "num_landmarks": 32,
  "mdl_gate": false, "competitive": true } }
```

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection |
| `500 Internal Server Error` | storage failure |

### GET /db/{db}/collections/{name}/locations

The hard locations themselves. Legacy: `GET /collections/{name}/locations`.

| Param | Type | Default | Notes |
|---|---|---|---|
| `full` | boolean | `false` | Include the raw `address` and `counter` vectors |

Summary form (`full=false`):

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/locations
```

```json
{ "locations": [ { "id": 0, "write_count": 12.0,
                   "avg_counter_magnitude": 0.041 } ] }
```

Full form (`full=true`) replaces `avg_counter_magnitude` with the raw vectors
— `L × 2D` floats, so it is large:

```json
{ "locations": [ { "id": 0, "write_count": 12.0,
                   "address": [ … ], "counter": [ … ] } ] }
```

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/collections/{name}/compress

Merge locations while it lowers the collection's description length, stopping
at the MDL minimum. Legacy: `POST /collections/{name}/compress`. Both fields
optional; an empty body `{}` is valid.

| Field | Type | Default |
|---|---|---|
| `kappa` | float | the collection's dimension — model cost per location, in bits |
| `lambda` | float | `1.0` — weight on the data-fit (variance) cost of a merge |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/compress \
  -H 'Content-Type: application/json' -d '{"kappa":128.0,"lambda":1.0}'
```

```json
{ "locations_before": 812, "locations_after": 640,
  "description_length_before": 91234.5,
  "description_length_after": 88012.1, "merges": 172 }
```

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection |
| `500 Internal Server Error` | storage failure during the merge |

### GET /db/{db}/collections/{name}/fingerprint

The collection's emergent self-summary: the write-count-weighted centroid of
every hard location's normalised pattern, refined by an iterative Hopfield
read so it lands on a real attractor rather than a bare average. Legacy:
`GET /collections/{name}/fingerprint`.

Request body: no body.

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/fingerprint
```

```json
{ "fingerprint": [0.13, -0.07, 0.26] }
```

`{"fingerprint": null}` when nothing has been written.

| Status | Cause |
|---|---|
| `400 Bad Request` | the engine rejected the fingerprint computation |
| `404 Not Found` | no such collection |
| `500 Internal Server Error` | storage failure |
