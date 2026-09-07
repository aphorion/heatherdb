# Document endpoints

A document is a vector written with metadata (see
[`write`](writes.md#post-dbdbcollectionsnamewrite)). It keeps its own id and is
retrievable and searchable independently of the hard locations that absorbed
it.

Every route here has a legacy form without the `/db/{db}` prefix that targets
the `default` database. Auth for all of them: Basic or Bearer, root or
`Database(db)` (legacy form: root or `Database("default")`). See
[Overview](overview.md) and
[Store and query structured documents](../how-to/structured-documents.md).

### GET /db/{db}/collections/{name}/documents

List every document in the collection. Legacy:
`GET /collections/{name}/documents`.

Request body: no body.

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/documents
```

```json
{ "documents": [ { "id": 17, "metadata": { "title": "Solaris" } } ] }
```

Documents whose metadata does not parse as JSON are skipped.

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection — this route does not create one |
| `500 Internal Server Error` | storage failure |

### GET /db/{db}/collections/{name}/documents/{doc_id}

One document by id. Legacy: `GET /collections/{name}/documents/{doc_id}`.

Request body: no body.

```bash
curl -u admin:pw \
  http://localhost:6380/db/movies/collections/taste/documents/17
```

```json
{ "id": 17, "metadata": { "title": "Solaris" } }
```

Unparseable metadata comes back as `null` rather than failing the call.

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection, or no such document |
| `500 Internal Server Error` | storage failure |

### DELETE /db/{db}/collections/{name}/documents/{doc_id}

Delete a document. Legacy:
`DELETE /collections/{name}/documents/{doc_id}`.

Request body: no body.

```bash
curl -u admin:pw -X DELETE \
  http://localhost:6380/db/movies/collections/taste/documents/17
```

```json
{ "deleted": true }
```

Deletion is idempotent: a missing document is `200 {"deleted": false}`, not a
404, so a tombstone replayed after a crash does not fail.

This removes the document and every posting-list reference to it, so it can no
longer be retrieved or cited. It does **not** subtract its contribution from
merged engrams — superposition cannot do that.

| Status | Cause |
|---|---|
| `404 Not Found` | no such collection |
| `500 Internal Server Error` | storage failure |

### POST /db/{db}/collections/{name}/documents/query

Similarity search over documents, using the hard-location posting list as the
index: the query activates locations, candidate document ids are gathered from
their posting lists, and exact cosine is computed on the candidates. Legacy:
`POST /collections/{name}/documents/query`. Creates the collection if it does
not exist. Audited.

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | array | *required* | Also the recall vector in every mode |
| `n` | integer | `10` | Results returned |
| `unbind_role` | array | — | Single-role scoring: unbind each candidate by this role and compare the recovered filler to `query` |
| `role_pairs` | array of `{role, filler}` | `[]` | Multi-role scoring. Mutually exclusive with `unbind_role` |
| `cleanup` | `"mdl"` \| `"off"` \| `{"beta": N}` | `"mdl"` | Cleanup of the recovered filler. `role_pairs` only |

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/documents/query \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"n":5}'
```

```json
{ "results": [ { "id": 17, "similarity": 0.88,
                 "metadata": { "title": "Solaris" } } ] }
```

With `role_pairs`, each hit also carries `role_scores` — one similarity per
requested pair, in request order, unweighted and unaggregated — and the
response carries the temperature cleanup actually ran at:

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/documents/query \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"n":100,
       "role_pairs":[{"role":[1,0,0],"filler":[0,1,0]},
                     {"role":[0,1,0],"filler":[0,0,1]}],
       "cleanup":"mdl"}'
```

```json
{ "results": [ { "id": 17, "similarity": 0.88, "metadata": { … },
                 "role_scores": [0.71, 0.34] } ],
  "cleanup_beta": 37.2 }
```

Both extra fields are omitted when not applicable, so a single-role response
is byte-identical to what pre-multi-role clients parse.

**Recall is not re-ranked by the pairs.** Candidate retrieval always uses
`query` and plain full-bundle cosine; `role_pairs` only score the set the
index already produced. A document that would rank well under your weighting
can fail to be recalled at all — ask for an `n` far larger than you intend to
display.

Betas below the engine floor (30.0) are raised to it. See
[Store and query structured documents](../how-to/structured-documents.md) for
the sharpness cliff this guards against.

| Status | Cause |
|---|---|
| `400 Bad Request` | `unbind_role` and `role_pairs` both present; a dimension mismatch on `query`, a role, or a filler |
| `500 Internal Server Error` | the collection could not be opened, or a storage failure |
