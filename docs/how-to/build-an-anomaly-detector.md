# Build an anomaly detector

Flag observations that do not fit a stream's normal, using only normal
examples, and report what normal would have looked like instead.

The conventional stack for this is an autoencoder or an isolation forest
trained on normal data, a validation set to pick a cut, a serving process, and
a retraining schedule for when normal drifts. The memory replaces the trained
model and the retraining: normal observations are loaded directly, the score
is the read's own reconstruction fidelity, and drift is a reload. You still
choose an encoding, and you still derive the threshold from held-out data.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| An item is | one observation — a request, a charge, a sensor window | the unit you want a verdict on |
| Similar means | agreeing on the encoded fields, weighted | see [Encode tabular data](encode-tabular-data.md) |
| A pool is | one collection per entity being modelled | fidelity means "normal *for this entity*" |
| Loading | `bulk_load` | the taught points must be the only attractors |
| Answer resolves to | the nearest cached normal to the reconstruction | that is the expected observation |
| Abstain when | the entity has too few normals loaded | a thin baseline reconstructs everything badly |

## Encode the observation

An observation is a row: identity columns, magnitudes, cycles. Build one
weighted feature vector per row and keep the weights in one place, because
they *are* the similarity metric — a block scaled by `w` contributes `w²` to
the inner product. [Encode tabular data](encode-tabular-data.md) has the
per-column rules and a worked encoder; [Encode quantities and
time](encode-quantities-and-time.md) covers rates and amounts when bucket
edges are too coarse. Two constraints specific to this job: **bucket every
magnitude, or encode it with FPE** — a raw scalar in one component is
invisible next to a one-hot block, so "847 requests" and "8 requests" land in
nearly the same direction — and **weight the field the question is about
above the rest**.

## One collection per entity

Capacity and the meaning of the score both argue for the finest partition you
can name: one collection per cardholder, per host, per device, named by the
entity id — `/db/ops/collections/normal_{entity}`. A new entity needs no
schema step and no rebalancing; its collection appears on first load.

A single global collection holding every entity's normal loses the signal
twice: interference between unrelated entities blurs the codebook, and a
reconstruction scored against *everyone's* normal answers a question you did
not ask. See [Shard a large pool](shard-a-large-pool.md).

## Load normals with `bulk_load`, not `write`

`write` grows a codebook that generalises: it splits, migrates and merges
locations until they cover the space the data lives in. That is the right
behaviour for a recommender and the wrong behaviour here, because a
space-filling codebook reconstructs *anything* well, including the anomaly.

`bulk_load` replaces the collection's locations with exactly the rows you
supply, so each taught point becomes its own attractor and nothing else does:
an observation far from every normal has nowhere to settle.

```python
# Same rows as addresses and counters: address is what a query is matched
# against, counter is what gets reconstructed. Identical rows mean "recall
# this observation verbatim when something looks like it".
rows = [encode(o) for o in normals_train]

call("POST", col(entity) + "/bulk_load",
     {"addresses": rows,
      "counters": rows,
      # Frequency belongs here, not in duplicate rows. A pattern seen 40
      # times outranks one seen once in the softmax without occupying 40
      # locations.
      "write_counts": [float(freq[o]) for o in normals_train]})
# -> {"n_loaded": 512, "dim": 128}
```

`bulk_load` replaces the whole collection, so every update is a full reload
from your own store of normals — cheap at a few thousand rows per entity, and
what keeps the attractor set exactly equal to the set you intend. It also
orphans the document index, so keep this collection document-free and resolve
against a sidecar cache instead: [Resolve a reconstruction to a
thing](resolve-reconstructions.md).

## Score by reconstruction fidelity

One read per observation. The score is the cosine between what you sent and
what came back — the caller computes it, there is no `fidelity` field on the
wire.

```python
def fidelity(entity, obs):
    q = encode(obs)
    r = call("POST", col(entity) + "/read",
             # "iterative" settles onto an attractor, so a normal observation
             # is pulled all the way onto a stored normal and scores near 1.0
             # while an anomaly is dragged somewhere it never asked to go.
             {"query": q, "strategy": "iterative"})["result"]
    return cos(q, r), q, r
```

High fidelity is normal; the anomaly score is its complement. No rule
anywhere says "over $1 000 is fraud" — the geometry does it.

## Derive the threshold from held-out normals

Hold out normals *before* you load them, as in [Calibrate a
gate](calibrate-a-gate.md). With no labelled anomalies you can still control
one side of the trade honestly.

```python
# Held-out normals were never loaded, so nothing scores against a codebook it
# contributed to.
held = [fidelity(entity, o)[0] for o in normals_heldout]
held.sort()

# The cut is a quantile of NORMAL, which fixes the false-positive rate by
# construction: at the 1st percentile, one normal in a hundred is flagged.
tau = held[max(0, len(held) // 100)]
print("tau %.3f  (n=%d held-out normals)" % (tau, len(held)))
```

A quantile of normal fixes the **false-positive rate** and says nothing about
detection rate, because no anomaly was measured. With even a handful of known
anomalies, stop using a quantile: put the cut in the empty band between the
two groups and report coverage and errors as [Calibrate a
gate](calibrate-a-gate.md) prescribes. Either way the gate is valid only at
its calibration load — recalibrate after any reload that changes the entity's
normal, and record the row count beside the threshold.

## Report the expected observation beside the observed one

The reconstruction is the explanation. Resolve it against a sidecar cache of
the loaded normals and print the nearest one next to the input: the diff
between the two is what the memory objects to.

```python
def explain(entity, obs):
    f, q, r = fidelity(entity, obs)
    cached = normals_cache[entity]          # [(label, vector), ...]

    # Nearest to the RECONSTRUCTION, not to the query. The query's neighbour
    # is whatever the anomaly happens to resemble; the reconstruction's
    # neighbour is what the memory expected to see instead.
    expected = max(cached, key=lambda kv: cos(r, kv[1]))

    return {"observed": obs,
            "expected": expected[0],
            "fidelity": round(f, 3),
            "verdict": "normal" if f >= tau else "anomaly"}
```

Rendering the two side by side — `expected: GET /api/users 200 from
10.0.1.15 - 8 req/60s` against `observed: POST /api/admin/users 403 from
185.220.101.42 - 847 req/60s` — is the whole explanation. Where the encoding
has a named basis, diff the field blocks rather than the strings and name the
fields that moved. [Explain a result](explain-a-result.md) covers the diff,
and `analyze` when you want the activated locations instead.

## Abstain rather than guess

```python
# `stats` 404s on a collection that does not exist, which is the only safe
# existence check -- a read would create it silently and then answer from an
# empty memory. Below MIN_NORMALS the held-out fidelity has no usable spread
# and every observation scores badly; take MIN from the calibration run, as
# the smallest n at which the held-out quantile is stable.
try:
    n = call("GET", col(entity) + "/stats")["num_locations"]
except NotFound:
    return {"verdict": "no baseline"}
if n < MIN_NORMALS:
    return {"verdict": "no baseline", "loaded": n}
```

Route abstentions to a coarser collection — the entity's cohort, or a global
fallback — and label the answer with which baseline produced it.

## Verify and tune

`GET .../stats` is the check — it 404s on a collection that does not exist,
and `num_locations` must equal the number of rows you loaded.

| Symptom | Change |
|---|---|
| Anomalies score as high as normals | `k` is too large — the read is averaging the whole collection. Lower `k` toward the number of normals a query should legitimately match |
| Everything scores low, including held-out normals | the encoding, not the engine: the fields that vary are outweighing the fields that identify. Re-weight, then re-derive `tau` |
| The gap narrows as you load more normals | the collection is covering the space. Split the entity by time window or mode, one collection each |
| `num_locations` exceeds what you loaded | something called `write` against this collection — it must only ever be `bulk_load`ed |

`k` and `beta` live in `db.toml` or the create call; see [Tune a
database](tune-a-database.md). Before trusting any separation, check what the
same score reads on data with no structure — [Find your metric's
floor](find-your-metrics-floor.md).

## Related

- [Calibrate a gate](calibrate-a-gate.md) — deriving `tau` from held-out data
- [Explain a result](explain-a-result.md) · [Resolve a reconstruction to a
  thing](resolve-reconstructions.md)
- [Encode tabular data](encode-tabular-data.md) · [Encode quantities and
  time](encode-quantities-and-time.md)
- [Shard a large pool](shard-a-large-pool.md) — one collection per entity
- [Write endpoints](../api/writes.md) · [Read endpoints](../api/reads.md)
- [Abstention](../explanation/abstention.md) · [Fidelity](../terms/fidelity.md)

Derived from the `sentinel` sample app (log-stream anomaly detection) and its
sibling `sieve` (near-duplicate triage) in `aphorion/heatherdb-samples`.
