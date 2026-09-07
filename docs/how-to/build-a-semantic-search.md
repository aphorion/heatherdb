# Build a semantic search

Retrieve passages by meaning, and keep retrieving them when the query is
partial, misremembered or noisy.

The conventional stack for this is an embedding model, a vector index
(FAISS, pgvector, a hosted ANN service), and a cross-encoder reranker on top
to repair the index's ranking. The memory replaces the index and the
reranker: one store holds the passages, answers exact queries with plain
cosine over its posting lists, and answers degraded queries by reconstructing
what the query was reaching for before it searches. The embedding model stays
— it is where "meaning" comes from.

## When this beats a plain vector index, and when it does not

Be honest about the boundary before you build on it.

| Query | Better tool | Why |
|---|---|---|
| Exact phrase from the corpus | plain index | the query already points at the answer; a reconstruction only drifts toward whatever attractor dominates that region |
| Well-formed, in-vocabulary question | either | the reconstruction lands near the query and returns the same hits |
| Partial, garbled, half-remembered | reconstruction | the query is a corrupted pattern, and completing corrupted patterns is what the read does |
| Something the corpus does not cover | reconstruction, for the refusal | first-contact similarity is low and stays low, so the system can decline |

Build both paths and let the read's own fidelity choose between them. That is
the design below.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| An item is | one passage — a paragraph, a note, a message | the unit you want returned verbatim |
| Similar means | embedding cosine, centred | see [Encode text](encode-text.md) |
| A pool is | one collection per corpus or tenant | capacity is per pool; tenants must not interfere |
| Writing | `write` **with metadata** | metadata is what mints a document id |
| Answer resolves to | `documents/query`, over-fetched then re-ranked | the verbatim text rides with the vector |
| Abstain when | first-contact similarity is below the derived cut | see [Calibrate a gate](calibrate-a-gate.md) |

## Encode passages

One passage, one vector. Chunk so that a chunk is a self-contained answer — a
paragraph, a note, a single exchange — not a fixed token count that splits
mid-argument.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")     # 384-d, CPU, ~90 MB

def embed(text: str) -> np.ndarray:
    # normalize_embeddings=True returns unit vectors, which is what a cosine
    # store wants; skip it and every downstream norm has to be recomputed.
    return model.encode(text, normalize_embeddings=True)
```

Two corrections are not optional, and both belong in one `to_store(v)` step:

```python
# One projection matrix, drawn once and persisted. Two matrices give two
# incomparable spaces, so a re-draw silently invalidates the whole corpus.
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(rng.standard_normal((384, DIM)))

def to_store(v):
    # Centre first: raw model output is anisotropic, and the large component
    # every vector shares inflates all scores equally while separating
    # nothing. The SAME mean must be applied to queries.
    # Then project down with the fixed matrix -- zero-padding is for going
    # up, projection is for coming down.
    x = (v - CORPUS_MEAN) @ Q
    return (x / (np.linalg.norm(x) + 1e-9)).tolist()
```

See [Center your vectors](center-your-vectors.md) and [Choose a
dimension](choose-a-dimension.md). The model version is part of your data
contract: a bump re-encodes the corpus, because vectors from two model
versions are not comparable.

## Write passages as documents

Metadata is what makes a vector retrievable. Write the verbatim text in it and
there is no second store to keep in sync.

```python
call("POST", "/db/kb/collections/passages/write",
     {"vectors": [to_store(embed(t)) for t in batch],
      # The payload rides with the vector. Nothing is recomputed at read
      # time, and the id in the response is the document's identity.
      "metadata": [{"text": t, "source": s, "url": u}
                   for t, s, u in batch_meta]})
# -> {"count": 256, "ids": [17, 18, ...]}
```

`write`, not `bulk_load`: `bulk_load` orphans the document index, so
`documents/query` would return nothing, and the competitive codebook is what
lets a query matching no passage exactly still land in the right region.

**Without `metadata` the response is a bare `{"count": N}` with no `ids`, and
those vectors are unretrievable as documents forever** — there is no backfill,
because superposition does not come apart. See [Resolve a reconstruction to a
thing](resolve-reconstructions.md).

## Retrieve in two passes

Pass one is the plain index. Pass two is the reconstruction. Fidelity decides
how much weight the second gets.

```python
def search(question, k=10):
    q = to_store(embed(question))

    # Pass 1 -- the query itself, straight at the posting lists. This is what
    # a plain vector index would return, and for an exact query it is the
    # answer.
    direct = call("POST", "/db/kb/collections/passages/documents/query",
                  {"query": q, "n": 50})["results"]

    # First contact: a single step, so the returned state is still near what
    # the query landed on rather than near the attractor the loop would drag
    # it to. This is the statistic that carries confidence -- post-iteration
    # similarity is ~1.0 by construction and separates nothing.
    fast = call("POST", "/db/kb/collections/passages/read",
                {"query": q, "strategy": "fast"})["result"]
    fidelity = cos(q, fast)

    if fidelity >= TAU_EXACT:
        # The query is already on an attractor. Reconstructing adds drift,
        # not recall. Return the index's answer.
        return direct[:k]

    if fidelity < TAU_FLOOR:
        # Nothing in the corpus is near this. Refuse rather than return the
        # 50 least-bad rows.
        return []

    # Pass 2 -- settle the query onto what the memory holds, then search
    # from there. The reconstruction's neighbours are what the memory makes
    # of a partial query, which is the whole reason for the second pass.
    r = call("POST", "/db/kb/collections/passages/read",
             {"query": q, "strategy": "iterative"})["result"]
    recalled = call("POST", "/db/kb/collections/passages/documents/query",
                    {"query": r, "n": 50})["results"]
    return rerank(q, direct, recalled, k)
```

**Over-fetch, always.** `n` caps what the posting-list scan returns, ranked by
plain full-bundle cosine to whatever you passed as `query`. Anything you
intend to re-rank must survive that first cut, so ask for several times what
you display.

```python
def rerank(q, direct, recalled, k):
    # Union by document id, scoring each hit against the ORIGINAL query. The
    # reconstruction is used to find candidates, never to score them --
    # scoring against it would rank passages by how typical they are of the
    # corpus rather than by how well they answer the question.
    pool = {h["id"]: h for h in direct}
    for h in recalled:
        pool.setdefault(h["id"], h)

    scored = [(cos(q, doc_vec[i]), h) for i, h in pool.items()]
    scored.sort(reverse=True, key=lambda t: t[0])
    return [h for _, h in scored[:k]]
```

Without passage vectors client-side, re-rank on `similarity` from pass one
and show pass-two-only hits as a separate "related" band — the lateral results
a plain index cannot produce.

## Derive the two cuts

`TAU_FLOOR` and `TAU_EXACT` are the same kind of object as any other gate, and
the procedure in [Calibrate a gate](calibrate-a-gate.md) applies unchanged:
score a held-out sample of queries the corpus can answer and queries it
cannot, and put the cut in the empty band between the groups. Report coverage
and error rate, never a single accuracy number.

The regime is visible in the statistic itself. Measured on the sample app's
transcript, reconstruction fidelity read 0.16 on a query opening a topic the
store held nothing on, and 0.66 on a query returning to a topic already
written — one number, read before any ranking, says which of the three
branches above a query belongs in. Calibrate on first contact rather than on
the iterative read: post-iteration similarity is ~1.0 by construction, so it
separates nothing.

Recalibrate as the corpus grows. A gate is valid only at the load it was
calibrated at: interference rises with occupancy, and a cut derived on three
thousand passages reads differently on eleven thousand.

## Do not expect `role_pairs` to steer recall

If passages are bundled role/filler records — `source` bound to a value,
`section` bound to another — `role_pairs` scores each criterion separately on
the hits it is given. It does **not** change which documents are retrieved:
candidate recall always uses `query` with plain full-bundle cosine. A document
that would rank well under your weighting can fail to be recalled at all, so
raise `n` far above what you display. See [Store and query structured
documents](structured-documents.md) and [Explain a
result](explain-a-result.md).

## Verify

`GET .../stats` is the existence check — it 404s on a collection that does
not exist, where a read against a typo would silently create one and then
answer from an empty memory. `total_writes` must match the passages you sent,
and `num_locations` tells you how far the codebook has compressed them.

| Symptom | Change |
|---|---|
| Pass 2 returns the same few passages for every query | the reconstruction is collapsing onto dominant attractors: raise `beta`, or lower `k` |
| Pass 2 returns nothing useful and fidelity is always high | the corpus covers your queries; drop pass 2 and ship the index |
| Recall misses passages you know are there | `n` is too low, or the chunk is too long to be one direction |
| Scores cluster tightly near 1.0 | vectors are not centred — see [Center your vectors](center-your-vectors.md) |

## Related

- [Encode text](encode-text.md) · [Center your vectors](center-your-vectors.md)
  · [Choose a dimension](choose-a-dimension.md)
- [Store and query structured documents](structured-documents.md) — metadata,
  ids, `role_pairs` · [Resolve a reconstruction to a
  thing](resolve-reconstructions.md)
- [Calibrate a gate](calibrate-a-gate.md) — deriving both cuts
- [Pair the memory with a language model](pair-with-an-llm.md) — feeding hits
  to a model, and sizing context by fidelity
- [Document endpoints](../api/documents.md) · [Read endpoints](../api/reads.md)

Derived from the `recall` sample app (conversational memory) and `cortex`
(associative note store) in `aphorion/heatherdb-samples`.
