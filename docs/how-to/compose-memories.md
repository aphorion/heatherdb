# Compose memories

Combine, weight and subtract whole bodies of knowledge with the collection
algebra, and verify each operation with a fingerprint before and after.

The ops in [`/algebra`](../api/algebra.md) act on *collections*, not on
vectors: each reads one or two sources, computes over their hard locations,
and writes the result into a target collection.

| Op | Design move |
|---|---|
| `add` | combine two bodies of knowledge |
| `sub` | remove an influence |
| `scale` | weight one body before combining it |
| `intersect` | keep only what two memories agree on |
| `permute` | move a memory into its own key space, so it stops colliding |

## Operands have mass

`add` and `sub` are pairwise cross products over locations. A collection with
more locations, or heavier ones, dominates the result. Subtracting a large
population from a small one annihilates the smaller one entirely — the target
is left holding the negation of the large collection with the small one as
rounding error.

Scale the subtrahend by roughly the mass ratio first. Measured on a small
collection minus a large one:

| α on the large source | Result |
|---|---|
| 0.02 | a no-op — the target is indistinguishable from `source_a` |
| 0.5 | clean — the influence is removed, the rest survives |
| 2.0 | noise — nothing recognisable in the target |

```python
# Mass ratio from the location counts, which is what the cross product
# actually iterates over. `num_locations` comes back on every stats call.
def n_loc(c):
    return call("GET", "/db/movies/collections/%s/stats" % c)["num_locations"]

alpha = n_loc("taste") / n_loc("everything_watched")   # e.g. 640/1280 = 0.5

# Weight the heavy source down to the light one's mass before subtracting.
call("POST", "/db/movies/algebra/scale",
     {"source": "everything_watched", "target": "_scaled", "alpha": alpha})

call("POST", "/db/movies/algebra/sub",
     {"source_a": "taste", "source_b": "_scaled", "target": "refined"})
```

α is a starting point, not an answer. Sweep it — 0.25, 0.5, 1.0 — and pick by
fingerprint, below.

## Validate with a fingerprint

`fingerprint` returns a write-count-weighted, Hopfield-refined identity vector
for a whole collection: the centroid of every hard location's normalised
pattern, weighted by how much was written into it, then settled onto a real
attractor. It is the cheapest description of what a collection *is*, and
comparing it before and after is how you check that an operation did what you
meant.

```python
def fp(c):
    return call("GET", "/db/movies/collections/%s/fingerprint" % c)["fingerprint"]

before = fp("taste")
call("POST", "/db/movies/algebra/sub", {...})
after = fp("refined")

# Three readings, and each one is a different verdict:
#   ~1.0  the operation changed nothing you can measure — α too small
#   ~0.0  the operation destroyed the collection's identity — α too large
#   in between, and moved AWAY from the thing you removed — it worked
print("kept identity   :", cos(before, after))
print("moved off target:", cos(fp("everything_watched"), after))
```

Do the same around `add`: a combined memory whose fingerprint is nearly
identical to one source has not combined anything.

## Intersect: what two memories agree on

`intersect` keeps only the cross pairs whose similarity clears a threshold, so
the target holds the shared region and nothing else.

```python
# threshold defaults to 0.95, which is strict. Lower it to widen agreement;
# a non-positive threshold admits every pair and re-arms the cross-product
# cap, which is usually a mistake rather than a setting.
call("POST", "/db/movies/algebra/intersect",
     {"source_a": "alice", "source_b": "bob", "target": "consensus",
      "threshold": 0.9})
```

An empty target is a real answer: the two memories agree on nothing above
that threshold.

## What `add` is not

`/algebra/add` superposes every pair of locations from the two sources. It is
a cross product, and its result is the pairwise sums — **not** the attractors
a single memory would have found if both bodies of data had been written into
it. Learning is a competitive process; combining two already-settled codebooks
does not re-run it.

When you need the attractors of the combined data, write the data:

| Want | Do |
|---|---|
| a blended vector space for reads | `algebra/add` into a target |
| the structure the combined data would learn | write both sources' vectors into one fresh collection |
| an answer routed across both, without a target | [`compose/read`](../api/algebra.md#post-dbdbcomposeread) |

`compose/read` asks several collections at once and blends the answers by
confidence, sharpened by `routing_sharpness` (default `20.0`) — higher routes
more of the answer to the most confident collection, lower blends more evenly.
It writes nothing.

```python
call("POST", "/db/movies/compose/read",
     {"collections": ["movies", "books"], "query": q,
      "routing_sharpness": 20.0})
# → {"result": [...], "weights": {...}, "confidences": {...}}
```

The `weights` and `confidences` are worth logging on their own: they say which
memory answered.

## The trap: algebra writes orphan the document index

Algebra results are written as raw vectors. There is no metadata to attach and
no document ids are minted, so a target collection carries locations that no
document points at. If the target already held documents, their posting-list
entries now sit in a codebook reshaped by an operation those documents were
not part of, and `documents/query` results become unreliable.

Make algebra targets **document-free collections**, named as such:

```python
# Convention: algebra targets are derived artefacts, never the collection
# your application queries for documents.
TARGET = "_derived_refined"
```

If you need documents on the far side of an operation, re-write them into a
fresh collection with their metadata rather than composing into a collection
that already has them.

## Related

- [Algebra endpoints](../api/algebra.md) — fields, caps, status codes
- [Vector algebra](../explanation/vector-algebra.md) — what the ops mean
- [Store and query structured documents](structured-documents.md) — why the
  document index needs its own collection
- [Bind](../terms/bind.md) · [Bundle](../terms/bundle.md) ·
  [Permute](../terms/permute.md) · [Superposition](../terms/superposition.md)
- [Tune a database](tune-a-database.md) — `--max-algebra-locations`,
  `--request-timeout`
