# Shard a large pool

When recall falls as a collection fills, split it into many collections and
route each query to one of them. Capacity is **per pool**, so the escape from
a capacity wall is more pools — not more dimensions.

## The measurement that settles it

Same total item count, same dimension, pool size 16, one collection versus
sharded:

| Items | One collection | Sharded |
|---|---|---|
| 128 | 62% recall | 100% |
| 512 | 7% | 100% |
| 2048 | 0% | 100% |

Recall collapses superlinearly in one pool and stays flat when sharded.
Doubling `d` doubles the `d / 32` ceiling once, and costs an immutable schema
change and a full re-ingest; doubling the shard count doubles capacity every
time, and costs a routing step.

## Procedure

1. **Partition by a natural key.** Entity, tenant, cluster, node, region,
   device — whatever already groups the data. A key that groups similar items
   beats a hash: shards then have distinct fingerprints, which is what makes
   routing work.
2. **Write each item into its own shard collection.**
3. **Fetch each shard's fingerprint** and keep them as the routing table.
4. **Route a query** to the nearest fingerprint, and read that shard.
5. **Blend across shards** with `compose/read` when the query genuinely spans
   several.

## Create shards and write to them

Shards are ordinary collections in one database; only a naming convention
marks them as shards.

```python
def shard_of(item: dict) -> str:
    """The partition key, as a collection name. Deriving it from the data
    means no directory or lookup table has to stay in sync -- the item
    always names its own shard."""
    return f"shard_{item['tenant_id']}"

# Writes are per-shard batches. Group before sending: one request per shard
# beats one request per item, and each shard's writes stay in one transaction.
by_shard = {}
for item in items:
    by_shard.setdefault(shard_of(item), []).append(encode(item))

for name, vectors in by_shard.items():
    call("POST", f"/db/{db}/collections/{name}/write", {"vectors": vectors})
```

## Build the routing table from fingerprints

A collection's
[fingerprint](../api/collections.md#get-dbdbcollectionsnamefingerprint) is its
emergent self-summary: the write-count-weighted centroid of every hard
location, refined by a Hopfield read so it lands on a real attractor rather
than a bare average. It is the shard's identity vector, and costs nothing to
maintain — the collection computes it from what it already holds.

```python
import numpy as np

def routing_table() -> dict[str, np.ndarray]:
    """One centroid per shard. Rebuild it after a bulk ingest; a fingerprint
    moves as its shard learns, and a stale table routes to the wrong place."""
    names = [c["name"] for c in
             call("GET", f"/db/{db}/collections")["collections"]
             if c["name"].startswith("shard_")]
    table = {}
    for n in names:
        fp = call("GET", f"/db/{db}/collections/{n}/fingerprint")["fingerprint"]
        # An empty shard answers null -- it has nothing to be similar to, so
        # it must not be a routing candidate.
        if fp is not None:
            table[n] = np.array(fp)
    return table

def route(query: np.ndarray, table: dict[str, np.ndarray], k: int = 1):
    """Rank shards by cosine to their fingerprints. k=1 for the routed read;
    k>1 to hand several shards to compose/read."""
    scored = sorted(
        ((float(query @ fp / (np.linalg.norm(query) * np.linalg.norm(fp))), n)
         for n, fp in table.items()), reverse=True)
    return [n for _, n in scored[:k]]
```

## Query the routed shard

```python
TABLE = routing_table()

target = route(q, TABLE)[0]
answer = call("POST", f"/db/{db}/collections/{target}/read",
              {"query": q.tolist(), "strategy": "iterative"})["result"]
```

One read against one small pool. The other shards are untouched — query cost
stops growing with total corpus size.

## Ask several shards at once

When the routing scores are close, or the query legitimately spans shards, let
[`compose/read`](../api/algebra.md#post-dbdbcomposeread) ask several and blend
by which one actually knows.

```python
res = call("POST", f"/db/{db}/compose/read", {
    "collections": route(q, TABLE, k=3),   # at least 2 required
    "query": q.tolist(),
    # routing_sharpness is the blend temperature: high concentrates the
    # answer on the confident shard, low averages across all of them.
    "routing_sharpness": 20.0,
})

print(res["weights"])       # {"shard_a": 0.81, "shard_b": 0.19, ...}
print(res["confidences"])   # per-shard recall confidence
```

The blend is weighted by each shard's own confidence, not by your fingerprint
scores — so a shard that looked plausible from its centroid but has nothing
relevant inside contributes almost nothing. The per-shard `weights` also make a
bad partition visible: one shard permanently at 1.0 means the key is not
partitioning anything.

## One collection per entity beats one global collection

The strongest version of this pattern is the finest partition: a separate
memory per user, per device, per account.

A single global memory holding everyone's history loses to plain cosine
similarity — interference between unrelated entities destroys more than the
memory adds. Per-entity memories win by 2× on the same data. The reason is the
capacity arithmetic, not the algorithm: one entity's items fit comfortably in
one pool, where everyone's items do not.

Practical consequences:

- **A new entity is a new collection**, created implicitly on first write.
  There is no schema step and no rebalancing.
- **Deleting an entity is deleting a collection** —
  `DELETE /db/{db}/collections/{name}` — rather than filtering rows out of a
  shared index.
- **Per-entity fidelity means something.** A read against one entity's own
  memory scores against that entity's normal — the anomaly signal a global
  store cannot give you.
- **Collections are cheap; databases are not.** Shard into collections inside
  one database. Reach for [multiple databases](multi-database.md) for
  tenancy, access scope, and differing dimensions — not for capacity.

## When sharding is not the answer

If a single *entity* has more items than `d / 32`, no partition by entity
helps — sub-shard it by time window or cluster, or raise the dimension.
[Choose a dimension](choose-a-dimension.md) says which case you are in.

## Related

- [Choose a dimension](choose-a-dimension.md) — the `d / 32` bound.
- [Work with multiple databases](multi-database.md) — the other axis of
  separation.
- [Tune a database](tune-a-database.md) — `tau_overload`, `k` and `beta` when
  one pool must hold more.
- [Collections](../api/collections.md) — `stats`, `fingerprint`, `compress`.
- [How the memory works](../explanation/associative-memory.md) — where
  capacity bites.
