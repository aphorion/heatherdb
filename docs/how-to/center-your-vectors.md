# Center your vectors

Subtract your corpus mean before you compare anything that came out of a
learned model. Without it, cosine over embeddings is near-1 for every pair and
discrimination collapses.

## The symptom

Learned embeddings are anisotropic: they occupy a narrow cone rather than
filling the sphere. Every raw pairwise cosine then sits on top of a large
shared constant, and the part that carries meaning is a thin band above it.

```python
import numpy as np

# X: (n, d) raw model output, already L2-normalised.
S = X @ X.T
off = S[~np.eye(len(X), dtype=bool)]     # every pair except self-pairs
print(f"raw      mean {off.mean():.3f}  sd {off.std():.3f}  "
      f"range {off.min():.3f}..{off.max():.3f}")

# Centering: remove the component every vector shares, then re-normalise.
# The shared component is not information -- it is the same in the query and
# in every candidate, so it inflates all scores equally and separates nothing.
mu = X.mean(axis=0)
Xc = X - mu
Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9

Sc = Xc @ Xc.T
offc = Sc[~np.eye(len(X), dtype=bool)]
print(f"centered mean {offc.mean():.3f}  sd {offc.std():.3f}  "
      f"range {offc.min():.3f}..{offc.max():.3f}")
```

Raw output crowds into a high, narrow band — unrelated documents scoring
alike, and a ranking whose top and bottom differ in the third decimal.
Centered output spreads across the full range, with unrelated pairs near zero
and related pairs clearly above them. Same vectors, same model, same metric;
the constant is gone.

Read it as: raw cosine reports "these are both text". Centered cosine reports
"these are about the same thing".

## Which vectors need it

| Vector source | Centre it? | Why |
|---|---|---|
| Pooled sentence/document embeddings | **yes** | anisotropic by construction |
| Image or audio model embeddings | **yes** | same cone |
| Spectral features (FFT magnitudes) | **yes** | a large DC-like common component |
| Log-magnitude features | **yes** | shared offset from the log floor |
| Independently drawn random symbols | no | already centred in expectation |
| One-hot blocks, `(sin, cos)` pairs | no | not drawn from a cone; centring would distort a metric you designed |
| Bundles of random symbols | no | inherit the symbols' centring |

The dividing line: **anything a model produced, or anything with a shared
offset baked into it, needs the mean removed.** Anything you drew yourself
from a symmetric distribution does not.

A mixed encoder — hashed symbols concatenated with a model embedding block —
centres only the model block, before the blocks are concatenated and the whole
vector is normalised.

## Procedure

1. Encode a representative sample of the corpus. A few thousand items is
   plenty; the mean converges fast.
2. Compute the mean over that sample and persist it beside the collection.
3. Apply it identically on every write and every query.
4. Keep the un-centered vector if you need to return the original.
5. Recompute when the corpus shifts.

```python
import json

def fit_mean(sample: np.ndarray, path: str) -> np.ndarray:
    """Compute the corpus mean once and write it to disk. This file is part
    of the model: a collection written under one mean and queried under
    another is querying a different space."""
    mu = sample.mean(axis=0)
    with open(path, "w") as f:
        json.dump(mu.tolist(), f)
    return mu

MU = np.array(json.load(open("corpus_mean.json")))

def prepare(raw: np.ndarray) -> tuple[list[float], list[float]]:
    """Return (stored, original). The centered vector is what the store
    indexes; the raw vector is what you hand back to a caller who asked for
    the embedding itself, or feed to a downstream model that expects the
    model's own space."""
    c = raw - MU
    c /= np.linalg.norm(c) + 1e-9
    return c.tolist(), raw.tolist()

stored, original = prepare(model.encode(text, normalize_embeddings=True))
call("POST", f"/db/{db}/collections/docs/write",
     {"vectors": [stored], "metadata": [{"text": text, "raw": original}]})
```

Centring changes what gets indexed, not what gets stored as metadata — keep
both when the original still has a job to do.

## A running mean, for streams

When items arrive continuously and no fixed sample exists, keep the mean
online. Freeze it once it settles; a mean that drifts under live traffic makes
a vector written this morning incomparable with a query made this afternoon.

```python
class RunningMean:
    def __init__(self, d: int):
        self.mu = np.zeros(d)
        self.n = 0

    def update(self, v: np.ndarray) -> None:
        # Welford-style incremental mean: one pass, no growing buffer, and
        # numerically stable over millions of items.
        self.n += 1
        self.mu += (v - self.mu) / self.n

    def frozen(self) -> np.ndarray:
        # Snapshot the mean and stop updating. Everything written and queried
        # from here on shares one space -- which is the only property that
        # matters. Persist this snapshot; do not re-derive it.
        return self.mu.copy()
```

Warm it on the first few thousand items, freeze, then write. Anything written
before the freeze was written against a moving mean and should be re-encoded.

## Recompute when the corpus shifts

The mean is a model of what "typical" looks like, and it goes stale when the
data changes shape — a new document type, a new language, a new sensor, a
model version bump.

Recomputing means re-encoding: a collection written under the old mean and a
query centred under the new one live in different spaces. Treat a mean change
exactly like a model change — new mean, new encode, new collection.

Watch for it by tracking the mean similarity of recent queries to their
results. A steady climb toward 1.0 across the whole stream is the cone
re-forming around a mean that no longer matches the data.

## Related

- [Encode text](encode-text.md) — the main producer of vectors that need this.
- [Encode tabular data](encode-tabular-data.md) — which blocks to leave alone.
- [Choose a dimension](choose-a-dimension.md) — the projection step, which
  belongs *after* centring.
- [What similarity means](../explanation/what-similarity-means.md) —
  anisotropy, and the rest of the standing rules.
- [Cosine similarity](../terms/cosine-similarity.md) — what the number is.
