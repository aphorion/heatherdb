# Build a deduplicator

Decide, for each incoming record, whether it is a duplicate of something the
system has already absorbed, a near-duplicate worth flagging, or new.

The conventional stack for this is either a similarity search — embed, index,
nearest-neighbour, threshold the distance — or a trained pairwise classifier
over labelled duplicate/not-duplicate pairs. This replaces both. There is no
index to rebuild as the corpus grows, no labelled pairs to collect, and the
verdict comes from one number the read hands you for free.

## The design

**An item is one record** — a ticket, a listing, a comment, a row. Whatever
your users would call "the same thing submitted twice".

**Two items are similar when their text means the same thing.** That is a
semantic judgement, so this is the case for an embedding model — see
[Encode text](encode-text.md), route 1. Hashed features would call "cannot log
in" and "unable to sign in" unrelated. Centre the output before writing it:
raw model output is anisotropic and every pair scores near 1.0 without it, and
a fidelity gate on top of that separates nothing. See
[Center your vectors](center-your-vectors.md). For rows rather than prose, use
[Encode tabular data](encode-tabular-data.md) and the same procedure below.

**A pool is a deduplication scope** — one queue, one tenant, one product line.
Capacity is about `d/32` patterns per pool
([Choose a dimension](choose-a-dimension.md)), and two records from different
scopes should never be able to interfere. One collection per scope beats one
global collection.

**The answer resolves twice.** The verdict comes from fidelity — the cosine
between the query and its own reconstruction, computed by you; there is no
`fidelity` field on the wire. The *evidence* — which stored record it matched
— comes from resolving the reconstruction against a sidecar of stored records.
See [Resolve a reconstruction to a thing](resolve-reconstructions.md).

**The abstention rule is the low band.** Below the unique threshold, the
memory has no basin for this record and returns no matches, rather than the
nearest thing it happens to hold.

## Fidelity as the verdict

A read runs the Hopfield loop from your query and returns the state it settles
into. If the record has been absorbed before, the query starts inside a basin
and the state barely moves: the reconstruction comes back close to the query,
and the cosine is high. If nothing like it was ever written, the state is
dragged toward whatever unrelated attractors are nearest and the cosine is
low. The distance travelled *is* the verdict.

```python
import json, sqlite3, struct, urllib.request
import numpy as np

DB, COLL = "intake", "tickets"

def call(method, path, body=None):
    req = urllib.request.Request(
        "http://localhost:6380" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0

def fidelity(vec):
    """cos(query, reconstruction). The engine returns a vector, never a
    score — this number is computed here, from the two vectors, and it is
    the whole verdict."""
    r = call("POST", f"/db/{DB}/collections/{COLL}/read",
             {"query": list(vec), "strategy": "iterative"})["result"]
    return cos(vec, r), r
```

## Repetition deepens the basin

Writes are not rows. Writing the same record ten times does not store ten
copies — it drives the same hard locations ten times, and their counters
converge harder on that pattern. `avg_write_count` and `max_write_count` in
`GET .../stats` are that history.

The consequence for a deduplicator is that the verdict is frequency-aware
without a frequency table: a complaint filed fifty times sits in a deep basin
and a near-miss variant of it still reads high, while a one-off record that
happens to resemble the incoming text reads lower. The gate is measuring "how
established is this pattern", not "how close is the nearest neighbour", and
those diverge exactly where a duplicate detector is asked to be useful.

It also means the order of ingestion matters at the margin: the first
occurrence of a record is always `UNIQUE`, by construction, because nothing
had absorbed it yet.

## Derive the bands

Two thresholds cut fidelity into three verdicts. **Derive both from held-out
labelled pairs; do not adopt the numbers from any sample app** — they depend
on your encoder, your dimension and your pool size. The procedure is in
[Calibrate a gate](calibrate-a-gate.md); what follows is its shape for two
cuts instead of one.

```python
# A few dozen labelled records is enough. `dupes` are records you know are
# restatements of something already ingested; `fresh` are records you know
# are new. Neither set may have been written to the collection.
dup_scores   = [fidelity(embed(t))[0] for t in heldout_dupes]
fresh_scores = [fidelity(embed(t))[0] for t in heldout_fresh]

# The duplicate cut goes in the empty band between the two groups. Below
# the highest fresh score there is no evidence of duplication at all.
tau_dup    = (max(fresh_scores) + min(dup_scores)) / 2
tau_unique = float(np.percentile(fresh_scores, 90))

# Report the trade at these cuts, not an accuracy. A deduplicator that
# merges two distinct tickets costs more than one that misses a duplicate,
# so the number to watch is false merges, at whatever coverage that buys.
auto = [s for s in dup_scores if s >= tau_dup]
bad  = [s for s in fresh_scores if s >= tau_dup]
print("auto-merged %d/%d dupes, %d false merges" %
      (len(auto), len(dup_scores), len(bad)))
```

The middle band between `tau_unique` and `tau_dup` is not a failure of
calibration. It is the population where a partial basin exists — same topic,
different incident — and it is the band a human or a downstream rule should
see. Size it deliberately: widening it trades throughput for safety.

```python
def verdict(f):
    """Three verdicts from one number, using the two derived cuts."""
    if f >= tau_dup:    return "DUPLICATE"      # merge, or reject on intake
    if f >= tau_unique: return "NEAR_DUPLICATE" # route for review
    return "UNIQUE"                             # accept
```

## Ingest a stream

Check first, then write. Checking after the write guarantees a duplicate
verdict for every record, since the record has just deepened its own basin.

```python
side = sqlite3.connect("dedupe.db")
side.execute("CREATE TABLE IF NOT EXISTS recs "
             "(id INTEGER PRIMARY KEY, text TEXT, vec BLOB)")

def remember(text, vec):
    """Sidecar: the vectors written to the engine, verbatim. Resolution is
    a cosine against these, so a re-encode with a different model version
    resolves to noise — store, do not recompute."""
    side.execute("INSERT INTO recs (text, vec) VALUES (?,?)",
                 (text, struct.pack(f"{len(vec)}f", *vec)))
    side.commit()

def cached():
    for t, b in side.execute("SELECT text, vec FROM recs"):
        yield t, np.array(struct.unpack(f"{len(b)//4}f", b))

def process(text):
    v = embed(text)                      # centred, unit-length
    f, recon = fidelity(v)
    v_ = verdict(f)

    # Evidence resolves against the RECONSTRUCTION, not the query. The
    # query's neighbours are what this text is nearest to; the
    # reconstruction's neighbours are what the memory recognised it as.
    matches = sorted(((cos(recon, cv), t) for t, cv in cached()),
                     reverse=True)[:3] if v_ != "UNIQUE" else []

    if v_ != "DUPLICATE":
        # Only genuinely new content is absorbed. Writing confirmed
        # duplicates would deepen basins for text you are discarding, and
        # inflate the pool against its d/32 capacity for nothing.
        call("POST", f"/db/{DB}/collections/{COLL}/write",
             {"vectors": [list(v)]})
        remember(text, list(v))

    return {"verdict": v_, "fidelity": round(f, 3), "matches": matches}
```

Deliberately writing a confirmed duplicate is the exception, not the rule:
do it when you want frequency to accumulate — a spam filter where the tenth
copy should read harder than the first — and skip it when the pool is near
capacity.

## Check it works

1. **Round-trip.** Ingest a record, then submit it again unchanged. The
   second submission must land in the `DUPLICATE` band. If it does not, the
   vectors are not being stored verbatim, or the encoder is not deterministic.
2. **Confirm the collection exists.** `GET /db/intake/collections/tickets/stats`
   returns `num_locations`, `total_writes`, `current_eta`, `avg_write_count`,
   `max_write_count`. Reads and writes auto-create a collection on a typo, so
   a `404` here — the one route that does not create — is how you find out you
   have been deduplicating against an empty memory.
3. **Recalibrate as the pool fills.** A gate is valid at the load it was
   calibrated at. Re-run the held-out procedure after every significant growth
   in `num_locations`, and treat drift in `tau_dup` as the signal to shard.

## Tune

| Symptom | Change |
|---|---|
| Everything reads as a duplicate | vectors uncentred, or `d` too high — check the raw pairwise cosine spread first |
| Nothing reads as a duplicate | pool near-empty, or `d` too low and every basin has smeared into one |
| No empty gap in the held-out sample | the encoder is not separating your duplicates — fix the encoding, not the cuts |
| Fidelity uniformly high, uninformative | the pool has exceeded `d/32` — [shard by scope](shard-a-large-pool.md) |
| Reads are slow | lower `k`, or use `strategy: "fast"` — then re-derive both cuts |

## Related

- [Calibrate a gate](calibrate-a-gate.md) — deriving both cuts
- [Encode text](encode-text.md) ·
  [Encode tabular data](encode-tabular-data.md) ·
  [Center your vectors](center-your-vectors.md)
- [Choose a dimension](choose-a-dimension.md) ·
  [Shard a large pool](shard-a-large-pool.md)
- [Resolve a reconstruction to a thing](resolve-reconstructions.md)
- [Read endpoints](../api/reads.md) ·
  [Collection endpoints](../api/collections.md)
- [Fidelity](../terms/fidelity.md) · [Attractor](../terms/attractor.md) ·
  [Interference](../terms/interference.md)

Derived from the `sieve` sample app in
[`aphorion/heatherdb-samples`](https://github.com/aphorion/heatherdb-samples).
