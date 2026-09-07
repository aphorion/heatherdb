# Encode tabular data

Turn a row of typed columns — an order, a transaction, a sensor reading, a
user profile — into one unit vector whose *direction* carries every
distinction you want reads to see.

## Classify every column first

Each column gets exactly one of five treatments. Decide the class before
writing any code; the class determines the encoding, and the encoding is the
only thing a read can ever see.

| Class | Example column | Encoding |
|---|---|---|
| Identity | `category`, `merchant`, `country` | hash-seeded random symbol |
| Magnitude | `amount`, `duration`, `count` | one-hot bucket, weighted |
| Cycle | `hour`, `weekday`, `heading` | `(sin, cos)` pair |
| Set | `genres`, `tags`, `roles` | bundle the symbols, then normalise |
| Free text | `description`, `title` | see [Encode text](encode-text.md) |

## Magnitude is invisible until you bucket it

Cosine reads direction only. Put a raw amount in a component and a \$5,000
transaction points the same way as a \$4.50 one — the store sees one event,
twice. Bucket edges make "how big" a direction:

```python
# Six buckets from five edges: <15, 15-60, 60-150, 150-600, 600-2000, >2000.
# Edges are roughly log-spaced because that is how spending is distributed --
# the gap between $4 and $40 matters, the gap between $4000 and $4040 does not.
BUCKETS = [15, 60, 150, 600, 2000]

def bucket(amount: float) -> int:
    for i, edge in enumerate(BUCKETS):
        if amount < edge:
            return i
    return len(BUCKETS)
```

Adjacent buckets are orthogonal, so \$59 and \$61 land as far apart as \$4 and
\$4,000. That step function is the cost of making magnitude visible at all; for
a gradient instead, use
[fractional power encoding](encode-quantities-and-time.md).

## Identity columns become namespaced random symbols

Hash the feature string to a seed, draw a unit vector from it. Two different
strings give near-orthogonal vectors; the same string always gives the same
vector, on every machine, with no state to persist.

```python
import hashlib
import numpy as np

DIM = 128

def symbol(feature: str) -> np.ndarray:
    """Deterministic unit vector for a typed feature string."""
    # SHA-256 rather than Python's hash(): hash() is salted per process, so
    # the same category would encode differently after a restart.
    seed = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8],
                          "little") % (2**31 - 1)
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)
```

**Namespace by prefixing the hash input**, not by keeping separate tables.
`symbol("country:georgia")` and `symbol("state:georgia")` are different
strings, so they are different vectors — one column can never accidentally
collide with another, however alike the raw values look.

## Multi-valued columns are bundled, then normalised as a group

```python
def symbol_set(prefix: str, values: list[str]) -> np.ndarray:
    """One vector for a whole set: sum the members, normalise once."""
    if not values:
        return np.zeros(DIM)
    v = sum(symbol(f"{prefix}:{x.lower()}") for x in values)
    # Normalise the *set*, not each member. A film with four genres then
    # carries the same total weight as a film with one -- otherwise genre
    # count alone would dominate every ranking.
    return v / (np.linalg.norm(v) + 1e-9)
```

## Weights are the similarity metric

A block scaled by `w` contributes `w²` to the inner product. Weights of 0.55
for genre and 0.15 for decade are not "genre a bit more important": genre
agreement counts `0.55² / 0.15² ≈ 13×` a decade agreement. Two same-decade
films from unrelated genres will not be recalled for each other; two
same-genre films from different decades will.

Set the weights by asking what you want back:

| You want | Do this |
|---|---|
| near-duplicates of a row | raise the identity weights |
| "things like this, roughly" | raise the set and text weights |
| size to dominate | raise the magnitude weight above the rest |
| a column ignored | weight 0, or drop the block |

## Reach the store's dimension by zero-padding

Zero-padding is norm-preserving and leaves every pairwise cosine exactly as
designed. Projecting up mixes components and only approximately preserves
them. Pad. Projection is for coming *down* — see
[Choose a dimension](choose-a-dimension.md).

## Worked example: a transaction

```python
CATS = ["groceries", "coffee", "restaurant", "gas", "retail",
        "electronics", "cash_advance"]

# Weights, in one place, because they are the metric. Amount dominates
# because the question is "does this charge fit the cardholder's normal";
# online is a weak hint, so it gets a fifth of amount's pull.
W = {"amt": 1.3, "dist": 1.0, "hour": 0.45, "cat": 0.75, "online": 0.3}

def encode(t: dict) -> list[float]:
    f = []

    # Magnitude: one-hot over six buckets (6 dims).
    amt = [0.0] * (len(BUCKETS) + 1)
    amt[bucket(t["amount"])] = W["amt"]
    f += amt

    # Ordinal, not cyclic: 0 = home city, 1 = domestic-far, 2 = overseas.
    # Scaled to [0,1] so the weight means the same thing as the others.
    f.append(W["dist"] * t["dist"] / 2)

    # Cycle: hour 23 and hour 0 must sit adjacent, so one scalar will not do.
    ang = 2 * np.pi * t["hour"] / 24
    f += [W["hour"] * np.sin(ang), W["hour"] * np.cos(ang)]

    # Identity: a small closed vocabulary, so plain one-hot beats hashing --
    # the categories are already orthogonal and stay human-readable (7 dims).
    f += [W["cat"] if c == t["category"] else 0.0 for c in CATS]

    # Boolean: one component, weighted like any other block.
    f.append(W["online"] * t["online"])

    # Normalise the assembled feature vector, then zero-pad to the store's
    # dimension. Padding after normalising keeps the vector unit-length.
    v = np.array(f) / (np.linalg.norm(f) + 1e-9)
    out = np.zeros(DIM)
    out[:len(v)] = v
    return out.tolist()
```

Write a handful of a cardholder's ordinary charges, then read any new charge
back and score the reconstruction. Everyday charges return at ~1.0; a \$4,200
overseas 3 a.m. electronics purchase returns at ~0.29, with an empty gap
between. No rule anywhere says "over \$1000 is fraud" — the geometry does it.

Put the threshold in the measured gap, not in a number copied from here. See
[Tune a database](tune-a-database.md) for what to change when the gap is
narrow.

## Related

- [Encode text](encode-text.md) — the free-text column.
- [Encode quantities and time](encode-quantities-and-time.md) — gradients
  instead of buckets.
- [Center your vectors](center-your-vectors.md) — required once a block comes
  from an embedding model.
- [Choose a dimension](choose-a-dimension.md) — what `DIM` should be.
- [What similarity means](../explanation/what-similarity-means.md) — why these
  rules hold.
