# Write endpoints

Two ways to put vectors into a collection: `write`, which learns
competitively, and `bulk_load`, which does not.

Both have a legacy form without the `/db/{db}` prefix that targets the
`default` database. Auth for both: Basic or Bearer, root or `Database(db)`
(legacy form: root or `Database("default")`). Both **create the collection if
it does not exist**. See [Overview](overview.md).

### POST /db/{db}/collections/{name}/write

Write vectors into the memory, with competitive learning: activation, novelty
splits, address migration. Legacy: `POST /collections/{name}/write`.

| Field | Type | Notes |
|---|---|---|
| `vectors` | array of arrays | Each must match the database's dimension. *Required* unless `vectors_b64` is sent |
| `vectors_b64` | array of strings | Instead of `vectors`: each vector's little-endian f64 bytes, base64-encoded (standard alphabet, padded). About half the bytes of JSON numbers and no float parsing. Send one of the two, not both |
| `metadata` | array of JSON values | Optional. Length must equal the number of vectors |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors":[[0.1,0.2,0.3],[0.4,0.1,0.9]],
       "metadata":[{"title":"Solaris"},{"title":"Stalker"}]}'
```

Without metadata the vectors are written as one batch (a single LMDB
transaction) and the response is just a count:

```json
{ "count": 2 }
```

With metadata each vector also becomes a retrievable
[document](documents.md). The adaptive writes run in order, exactly as
one-by-one writes would, but the whole request is persisted in one LMDB
transaction; the response carries the assigned ids:

```json
{ "count": 2, "ids": [17, 18] }
```

| Status | Cause |
|---|---|
| `400 Bad Request` | no vectors, both `vectors` and `vectors_b64`, a `vectors_b64` entry that is not base64 of whole f64 values, a dimension mismatch, a `metadata`/`vectors` length mismatch, or metadata that will not serialise as JSON |
| `413 Payload Too Large` | body over `--max-body-size` (default 2 MiB) |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |

### POST /db/{db}/collections/{name}/bulk_load

Replace a collection's hard locations atomically with a caller-supplied set.
Bypasses competitive learning entirely: no activation, no novelty splits, no
address migration. This is what makes snapshot algebra operate on known
operands. Legacy: `POST /collections/{name}/bulk_load`.

| Field | Type | Notes |
|---|---|---|
| `addresses` | array of arrays | *required*, non-empty. Each of length `d` |
| `counters` | array of arrays | *required*. Same length as `addresses`, each of length `d` |
| `addresses_b64`, `counters_b64` | arrays of strings | Instead of `addresses` / `counters`: base64 little-endian f64 rows, as `vectors_b64` on `write` |
| `write_counts` | array of floats | Optional. Same length as `addresses`. Defaults to `1.0` each |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/codebook/bulk_load \
  -H 'Content-Type: application/json' \
  -d '{"addresses":[[1.0,0.0,0.0],[0.0,1.0,0.0]],
       "counters":[[0.5,0.5,0.0],[0.0,0.5,0.5]],
       "write_counts":[10.0,3.0]}'
```

```json
{ "n_loaded": 4096, "dim": 128 }
```

The collection's existing config (dimension, EAM knobs) is preserved. The
replace is version-checked: if a concurrent write lands first, nothing is
changed — retry.

| Status | Cause |
|---|---|
| `400 Bad Request` | empty `addresses`; `counters` or `write_counts` a different length from `addresses`; a row whose length is not `d`; `addresses` over `--max-bulk-items` (default 100 000) |
| `409 Conflict` | a concurrent write bumped the collection version — retryable |
| `413 Payload Too Large` | body over `--max-body-size` |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |
