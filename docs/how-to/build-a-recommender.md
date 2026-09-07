# Build a recommender

Rank a catalogue for one person from what that person has already done, with
no training step and no nightly rebuild.

The conventional stack for this is a matrix-factorisation or two-tower model,
an interaction table to train it on, a serving step that materialises user
embeddings, and a retraining schedule so new users and new items appear. The
memory replaces the model and the schedule: a user's taste vector is read out
of their own collection, and it is current the moment their last interaction
is written. You still supply an item encoder and you still own the catalogue.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| An item is | one catalogue entry — a film, a track, a product | the unit you rank |
| Similar means | agreeing on weighted item features | see [Encode tabular data](encode-tabular-data.md) |
| A pool is | one collection per user | measured: per-user beats one global memory |
| Writing | `write` | the codebook must generalise beyond the items seen |
| Answer resolves to | the catalogue, ranked against the fingerprint | the catalogue stays application-side |
| Abstain when | the fingerprint is `null`, or history is too thin | a taste vector from two items is not a taste |

## Encode an item

An item is a row of attributes: a set column (genres), an identity column
(director), a magnitude (year), maybe free text (plot). Hash each typed
feature to a fixed random direction, weight the blocks, normalise once. No
model is required.

```python
import hashlib
import numpy as np

DIM = 128
# The weights ARE the metric: a block scaled by w contributes w^2 to the
# inner product, so genre agreement at 0.55 counts (0.55/0.15)^2 ~ 13x a
# decade agreement. Set them by asking what you want back.
W = {"genre": 0.55, "title": 0.30, "decade": 0.15}

def symbol(feature: str) -> np.ndarray:
    """A stable random direction per typed feature string, seeded from the
    hash rather than from process state. Namespace by prefixing the input:
    "genre:action" and "director:action" are different strings, so one column
    can never collide with another however alike the raw values look."""
    seed = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8],
                          "little") % (2**31 - 1)
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)

def symbol_set(prefix: str, values) -> np.ndarray:
    # Normalise the SET, not each member, so a four-genre film does not
    # outweigh a one-genre film on genre count alone.
    s = sum(symbol(f"{prefix}:{v.lower()}") for v in values)
    return s / (np.linalg.norm(s) + 1e-9)

def encode(item) -> list[float]:      # one unit vector per catalogue entry
    v = (W["genre"] * symbol_set("genre", item.genres)
         + W["title"] * symbol_set("title", item.title_tokens)
         + W["decade"] * symbol("decade:%d" % (item.year // 10 * 10)))
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()
```

For a catalogue with substantial free text, swap the title block for a
sentence embedding and project down to the store's dimension — see [Encode
text](encode-text.md) and [Choose a dimension](choose-a-dimension.md). Model
output must be [centred](center-your-vectors.md) before it is comparable.

## One collection per user

```python
# The user id is the collection name. A new user is a new collection,
# created implicitly on first write -- no schema step, no rebalancing.
def col(uid): return "/db/movies/collections/user_%s" % uid
```

**A single global collection loses.** Measured on MovieLens 1M at `d = 128`,
held out over 1 000 users: one collection holding every item, and one holding
every user's fingerprint, scored Hit@10 of 0.140 and 0.180 against a
raw-cosine baseline of 0.372 — a loss of roughly 2×. Collapsing everyone into
one memory denoises each query toward a global centroid and destroys the
seed-specific direction that makes a recommendation personal. Per-user, on the
same data and encoder, ties or beats the baseline — and deleting a user is
then `DELETE` on one collection, not a filtered delete across a shared
index.

## Write the history

`write`, not `bulk_load`. `write` is competitive: repeated interactions in one
region of taste raise that region's write count and migrate its address, which
is what the fingerprint keys off. `bulk_load` would freeze the taught points
as the only attractors and give up the generalisation.

```python
# Rating weight is repetitions, not a scalar. Writing a 5-star item three
# times deepens its basin three times as much; the engine has no per-write
# weight parameter, and this stands in for one.
def reps(rating: float) -> int:
    return 3 if rating >= 5.0 else (2 if rating >= 4.0 else 1)

vecs = [encode(r.item) for r in history for _ in range(reps(r.rating))]

# Batches of a few hundred. One request per batch beats one per item, and
# each batch is a single transaction.
for i in range(0, len(vecs), 256):
    call("POST", col(uid) + "/write", {"vectors": vecs[i:i + 256]})
# -> {"count": 256}
```

Two consequences to design around. **A write without metadata is not a
document**: the response is a bare `count` with no `ids`, and
`documents/query` will never return these vectors. That is fine here — the
catalogue is resolved application-side — but it is not reversible later
([Resolve a reconstruction to a thing](resolve-reconstructions.md)). And **a
write cannot be retracted**, because superposition does not come apart: to
remove an item from a history, drop the collection and rewrite the remainder,
which is cheap at a few hundred items per user.

## Read the taste vector

`fingerprint` is the collection's self-summary: the write-count-weighted
centroid of every hard location's pattern, refined by an iterative Hopfield
read so it lands on a real attractor rather than a bare average.

```python
fp = call("GET", col(uid) + "/fingerprint")["fingerprint"]
# -> {"fingerprint": [0.13, -0.07, 0.26, ...]}   or {"fingerprint": null}

if fp is None:            # nothing written yet
    return []

# Re-normalise client-side. The endpoint does not guarantee unit norm, and
# the ranking below is a dot product that assumes one.
fpv = np.array(fp, dtype=np.float32)
fpv /= np.linalg.norm(fpv) + 1e-9
```

Cache it per user and invalidate on every write. A stale fingerprint is the
most common bug in this design: without invalidation, every recommendation
after the first reflects the history as it stood at the first read.

## Resolve against the catalogue

The memory holds the taste; the catalogue stays in your process, as a matrix
of the same item vectors you wrote. Ranking is one matmul.

```python
# CATALOG: (n_items, DIM) float32, row i = encode(catalog[i]). These must be
# the exact vectors you wrote -- re-encoding with different weights resolves
# to noise.
sims = CATALOG @ fpv

# Exclude before the top-k, not after: set seen rows to -inf so they cannot
# occupy a slot. Filtering afterwards silently returns fewer than k.
for row in seen_rows(uid):
    sims[row] = -np.inf

top = np.argpartition(-sims, k)[:k]
top = top[np.argsort(-sims[top])]
return [(CATALOG_IDS[i], float(sims[i])) for i in top]
```

Two filters belong here, both editorial rather than confidence-based: a
popularity floor, so a fingerprint does not surface obscure long-tail rows
that merely point the right way, and a similarity floor exposed as an
adventurousness dial rather than a hidden threshold.

## Abstain on a thin history

There is no confidence gate on a fingerprint — it summarises whatever is
there, including two items. Gate on the history instead:

```python
# `stats` 404s when the collection does not exist, which is the only safe
# existence check: a read auto-creates a collection on a typo and then
# answers from an empty memory rather than failing.
try:
    n_written = call("GET", col(uid) + "/stats")["total_writes"]
except NotFound:
    return {"recommendations": [], "reason": "no history yet"}
if n_written < MIN_HISTORY:
    return {"recommendations": [], "reason": "watch a couple more first"}
```

Derive `MIN_HISTORY` rather than picking it: hold out one item per user, sweep
the seed size, and take the smallest seed at which held-out rank stops
improving — [Calibrate a gate](calibrate-a-gate.md) is the procedure.

## Verify it works

Measure Hit@K against a raw-cosine baseline on the same split — superpose the
seed items, normalise, rank the catalogue — and report both numbers. On
MovieLens 1M with a 30-item seed, per-user fingerprints scored Hit@10 of 0.275
against the baseline's 0.250 with rating-weighted writes, and 0.220 without
them. That is competitive, not a rout, and the reasons to build it this way
are that a new interaction is live immediately, that a user's memory is a
droppable object, and that every recommendation can be attributed. [Find your
metric's floor](find-your-metrics-floor.md) covers the null and the baseline.

## Related

- [Shard a large pool](shard-a-large-pool.md) — why one collection per entity
- [Explain a result](explain-a-result.md) — `analyze` and `batch_analyze` give
  "why this?" as shared location weight, with no second model
- [Compose memories](compose-memories.md) — `algebra/add` blends two users'
  collections, and operands of different sizes need scaling first
- [Resolve a reconstruction to a thing](resolve-reconstructions.md) ·
  [Calibrate a gate](calibrate-a-gate.md) · [Choose a
  dimension](choose-a-dimension.md)
- [Encode tabular data](encode-tabular-data.md) · [Encode text](encode-text.md)
- [Collection endpoints](../api/collections.md) — `fingerprint`, `stats` ·
  [Write endpoints](../api/writes.md)

Derived from the `cinema` sample app in `aphorion/heatherdb-samples` and the
MovieLens recommender in `aphorion/heatherdb-pi-demo`.
