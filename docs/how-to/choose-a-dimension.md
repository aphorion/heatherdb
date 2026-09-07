# Choose a dimension

Pick the one number you cannot change later. Dimension is set at database
creation, immutable afterwards, and it is an **interference knob** — not a
capacity setting you raise until the warnings stop.

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' -d '{"name":"movies","dimension":384}'
```

## What the two failure modes look like

| Dimension relative to data | Behaviour |
|---|---|
| Too high | reads degrade toward nearest-neighbour lookup; nothing completes, nothing reads as anomalous |
| Matched | reconstruction and error correction |
| Too low | everything smears; reads blend unrelated patterns into a plausible vector that matches none of them |

The high side is the one that hides. At 384 dimensions with a small corpus,
everything reconstructs at ~0.95 fidelity — including inputs that should look
strange. A store where every read succeeds is a store that has stopped
generalising: patterns are so far apart that superposition never engages, and
the anomaly signal you wanted has been diluted away. Fidelity near 1.0 for
*everything* is a symptom, not a pass.

## The capacity arithmetic

Two numbers govern the choice:

| Quantity | Bound |
|---|---|
| Patterns superposable into one pool | `≈ d / 32` |
| Distinct role/filler slots in one bundled schema | `≲ √(d / 32)` |

The square root on the second is why structured records run out of room long
before flat feature vectors do. Ten slots in one bundle wants `d ≥ 3200`; ten
thousand flat items in one pool wants `d ≥ 320000`, which is the wrong answer
— see the procedure below.

Typical dimensions in working systems:

| `d` | Fits |
|---|---|
| 128 | feature vectors, sensor windows, hashed encoders — anything from [Encode tabular data](encode-tabular-data.md) |
| 512–1024 | HRR bundling with headroom: several roles per record, or many items per pool |
| 2048 | structures that nest — bundles of bundles, sequences of records |

## Procedure

1. **Estimate items per pool.** Not items in the database, not items in the
   collection: items that will land in the *same* hard location. That is your
   natural grouping — one entity, one tenant, one cluster.
2. **Check against `d / 32`.** For a candidate `d`, `d / 32` is the ceiling.
   128 gives 4, 512 gives 16, 2048 gives 64.
3. **Count schema slots** if records are bundles, and check against
   `√(d / 32)`.
4. **If you are over the bound, shard before you widen.** Capacity is per
   pool, so more pools buys more than more dimensions does, and it buys it
   linearly. See [Shard a large pool](shard-a-large-pool.md).
5. **Verify by measurement.** Never by argument.

## Verify by measurement

Write a batch you believe is in-distribution, then read it back and score.
Also score random vectors. The gap is the whole answer.

```python
import numpy as np

def fidelity(vec):
    """Cosine between what you wrote and what comes back. This is the one
    number that tells you whether the dimension is right -- it measures
    reconstruction, which is the thing dimension controls."""
    got = call("POST", f"/db/{db}/collections/{coll}/read",
               {"query": vec, "strategy": "iterative"})["result"]
    a, b = np.array(vec), np.array(got)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

known = [fidelity(v) for v in held_out]        # should reconstruct well
noise = [fidelity(random_unit()) for _ in range(200)]  # should not

print(f"known {np.mean(known):.3f}  noise {np.mean(noise):.3f}")
```

Read the result:

| Observation | Diagnosis | Move |
|---|---|---|
| known ≈ noise ≈ 0.95 | `d` too high — everything looks familiar | lower `d` |
| known low, unstable | `d` too low, or the pool is over capacity | shard first, then raise `d` |
| known high, noise low, wide gap | correct | keep it |

Watch `num_locations` from
[`GET …/collections/{name}/stats`](../api/collections.md#get-dbdbcollectionsnamestats)
as you go. Locations growing roughly with data diversity is healthy; locations
pinned at one while writes pile in means the pool is absorbing everything, and
step 4 applies.

## Coming down from an embedding model

An embedding model's native width is the model's choice, not yours.
Sentence-transformer output at 384 into a 128-dimensional collection is a
routine reduction — by a **fixed-seed orthogonalised random projection**:

```python
import numpy as np

SEED, SRC, DST = 20260101, 384, 128

def projection() -> np.ndarray:
    """Build the one projection matrix this collection will ever use."""
    rng = np.random.default_rng(SEED)
    # QR of a random Gaussian gives an orthonormal basis. Orthogonalised rows
    # make this a rotate-and-truncate rather than a random smear, so pairwise
    # angles survive as well as 128 dimensions allow.
    q, _ = np.linalg.qr(rng.standard_normal((SRC, DST)))
    return q                                   # (384, 128), columns orthonormal

P = projection()

def reduce(vec_384: np.ndarray) -> np.ndarray:
    v = vec_384 @ P
    return v / (np.linalg.norm(v) + 1e-9)
```

Three properties make it safe, and one of them is a rule you must not break:

- **Orthogonalised** — the reduction preserves angles as closely as the target
  width permits.
- **One direction** — it reduces; it is not inverted. What a read returns
  lives in the reduced space.
- **The same matrix every time.** The seed is part of the collection's
  contract. A projection rebuilt with a different seed — a different library
  version, a different machine, an unpinned RNG — puts queries in a different
  space from the writes, and the store answers nothing well while looking
  perfectly healthy. Pin the seed, and rebuild the matrix from it rather than
  shipping the matrix around.

Widening is the opposite problem and has an easier answer: zero-pad. It
preserves the norm and every pairwise cosine exactly. Never project up.

## When you get it wrong

There is no in-place conversion — the stored codebook is dimensioned. Create a
new database at the new dimension and re-write the data:

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' -d '{"name":"movies_v2","dimension":512}'
```

Which is a reason to run the measurement above on a throwaway database, with a
representative sample, before the real ingest.

## Related

- [Shard a large pool](shard-a-large-pool.md) — the answer when the bound
  bites.
- [Tune a database](tune-a-database.md) — the knobs that are *not* immutable.
- [Encode text](encode-text.md) — where the 384 comes from.
- [What similarity means](../explanation/what-similarity-means.md) — dimension
  as an interference knob.
- [How the memory works](../explanation/associative-memory.md) — capacity, and
  where it bites.
