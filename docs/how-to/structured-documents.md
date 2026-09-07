# Store and query structured documents

A **document** is a vector written with metadata attached. It keeps its own id
and stays individually retrievable, searchable and deletable, alongside the
hard locations that absorbed its vector.

## Write documents

Add a `metadata` array to any write. It must be the same length as `vectors`,
and each entry is arbitrary JSON.

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/collections/films/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[…], […]],
       "metadata": [{"title":"Solaris","year":1972}, {"title":"Stalker","year":1979}]}'
```
```json
{"count": 2, "ids": [0, 1]}
```

The `ids` field only appears when metadata was supplied. Without metadata the
vectors go in as one batched transaction and you get `{"count": 2}` — faster,
but nothing to retrieve by id afterwards.

## List, fetch, delete

```bash
curl -u admin:pw http://localhost:6380/db/movies/collections/films/documents
curl -u admin:pw http://localhost:6380/db/movies/collections/films/documents/0
curl -u admin:pw -X DELETE http://localhost:6380/db/movies/collections/films/documents/0
```

Deletion is idempotent — deleting a document that is already gone returns
`200 {"deleted": false}`, not a 404, so a replayed tombstone never fails.

**Deletion removes retrievability, not influence.** The document and every
posting-list reference to it disappear, so it can no longer be returned or
cited. Its contribution to the merged engrams stays: superposition cannot be
un-added. If you need a vector's effect gone, rebuild the collection.

## Search

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/collections/films/documents/query \
  -H 'Content-Type: application/json' -d '{"query": [ … ], "n": 5}'
```
```json
{"results": [{"id": 2, "similarity": 1.0, "metadata": {"title":"Interstellar"}}]}
```

Retrieval goes through the hard-location posting list: the query activates
locations, candidate document ids are gathered from those locations' posting
lists, and exact cosine is computed on the candidates. It is an index, not a
scan — which also means a document whose vector is far from `query` is never a
candidate, however well it might have scored.

## Role/filler documents

The interesting case is a document built compositionally out of the
[vector algebra](../explanation/vector-algebra.md): bind each attribute to a
role vector, then bundle the pairs into one document vector.

```
film = bundle( bind(GENRE, scifi), bind(DIRECTOR, nolan) )
```

Fix one random unit vector per role (`GENRE`, `DIRECTOR`) and per value
(`scifi`, `nolan`) and keep them client-side; they are the vocabulary.

```python
# The two algebra primitives, as one-liners over the stateless /vec routes.
# `bind` pairs a role with its value; `bundle` superposes several pairs into
# one vector of the same width.
def bind(a, b):    return call("POST", "/vec/bind",   {"a": a, "b": b})["result"]
def bundle(vs):    return call("POST", "/vec/bundle", {"terms": [{"vector": v} for v in vs]})["result"]

films = [("Solaris", scifi, tarkovsky), ("Interstellar", scifi, nolan)]

# One vector per film: "genre is X" bound, "director is Y" bound, both bundled.
# The result is a whole record in a single vector — no columns were declared.
vectors = [bundle([bind(GENRE, g), bind(DIRECTOR, d)]) for _, g, d in films]

# Metadata rides alongside the vector; it comes back with hits but is not what
# the memory searches on.
call("POST", "/db/movies/collections/films/write",
     {"vectors": vectors, "metadata": [{"title": t} for t, _, _ in films]})
```

Query it by building the same shape from the attributes you want:

```python
# Build the query the same way you built the records: it is not a filter
# expression, it is a vector of the shape you are looking for.
q = bundle([bind(GENRE, scifi), bind(DIRECTOR, nolan)])

# `n` caps how many hits come back, ranked by similarity to that shape.
call("POST", "/db/movies/collections/films/documents/query", {"query": q, "n": 5})
```

An exact structural match scores 1.0; documents sharing one attribute land well
below it.

## Per-attribute scoring

Whole-bundle cosine has a confound: a document's score for one role is divided
by its own norm, which grows with every *other* role it carries. Richly
structured documents therefore rank lower for reasons unrelated to the query
(measured correlation between filled-slot count and score: −0.35 to −0.42).

`role_pairs` scores each criterion separately. Each pair carries its own role
*and* its own expected filler:

```python
# Same query vector, but ask for each criterion to be scored on its own.
# Recall and ranking still use `query`; `role_pairs` only adds per-role numbers
# to each hit, so keep `n` generous enough that the document you care about is
# recalled before you re-rank it yourself.
call("POST", "/db/movies/collections/films/documents/query", {
    "query": q, "n": 50,
    "role_pairs": [{"role": GENRE, "filler": scifi},
                   {"role": DIRECTOR, "filler": nolan}]})
```
```json
{"results": [{"id": 2, "similarity": 1.0, "metadata": {…},
              "role_scores": [0.089, -0.108]}],
 "cleanup_beta": 59.5}
```

Each hit gains `role_scores` — one similarity per pair, **in request order,
unweighted and unaggregated**. How to combine them is your decision; the engine
deliberately does not make it.

Three things to know before relying on this:

- **Ranking ignores the pairs entirely.** `similarity` is still the plain
  full-bundle recall score, and recall still uses `query`. A document that
  would rank first under your own weighting can fail to be recalled at all —
  ask for an `n` far larger than you intend to display, then re-rank client-side.
- **Role scores are small numbers.** Unbinding recovers the filler plus
  crosstalk from every other role in the bundle, and cleanup denoises against
  the collection's own codebook rather than your filler vocabulary. Read them
  as relative signal between candidates, not as absolute confidence.
- **`unbind_role` and `role_pairs` are mutually exclusive** (`400` if you send
  both). The older `unbind_role` reuses the recall `query` as the filler, which
  conflates two different vectors; `role_pairs` does not. Prefer `role_pairs`.

## Cleanup and the sharpness cliff

`cleanup` controls how the recovered filler is denoised before it is scored:

| Value | Meaning |
|---|---|
| `"mdl"` *(default)* | the engine picks the temperature per query from the codebook's own geometry |
| `{"beta": 100.0}` | an explicit inverse temperature |
| `"off"` | score the raw recovered filler |

Leave it alone unless you have a reason. Cleanup quality is brutally sensitive
to β and **the failure is quiet** — a soft β returns a plausible-looking
ordering that is simply wrong. Measured role score against a 0.095 ceiling:

| β | score |
|---|---|
| 10 | 0.057 |
| 30 | 0.077 |
| ≥100 | 0.091 |

Explicit betas below 30.0 are raised to 30.0. The temperature actually used
comes back as `cleanup_beta`, so you can always see what happened.

## What this is not

Metadata is stored and returned; it is not indexed. There is no filtering,
sorting or querying by metadata field — no `where year > 1970`. If you need
that, filter the returned results client-side with a generous `n`, or keep the
structured fields in a database that does filtering and use HeatherDB for the
vector half.
