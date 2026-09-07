# Capacity reference

The measured limits of superposition, and what to do at each wall.

Every number here is measured, not derived. Recall percentages are the fraction
of stored items a read returns correctly. Unless a row says otherwise, the
figures come from numpy models of the same arithmetic the engine runs; the
live-engine walls are one notch tighter (see [Live engine vs.
numpy](#live-engine-vs-numpy)).

**The escape from a capacity wall is sharding into more collections, not
raising the dimension.** Dimension buys a square-root; sharding buys a linear
factor and is not immutable.

## Per-pool capacity

How many patterns superpose into one location before per-read recall degrades.

| D | Items per pool at ≥90% recall |
|---|---|
| 512 | 32 |
| 1024 | 32 |
| 2048 | 64 |

The rule of thumb is `D/32`, which is what `tau_overload` defaults to
(`max(d/32, 8)`): a saturating location splits *as* it reaches the wall rather
than well past it.

The failure mode is a read that blends several stored patterns into a
plausible-looking vector matching none of them. Fidelity is the detector.

## Schema capacity

Role–filler slots recoverable from one bundle. The bound is `√(D/32)`, not
`D/32` — each slot costs a bind, and crosstalk grows with the number of slots
rather than the number of items.

At D = 1024:

| Slots | Recall |
|---|---|
| 2 | 100% |
| 3 | 100% |
| 4 | 94% |
| 6 | 22% |
| ≥8 | 0% |

Raising D restores it. At 8 slots:

| D | Recall |
|---|---|
| 1024 | 0% |
| 2048 | 15% |
| 4096 | 96% |
| 8192 | 100% |

Doubling D roughly doubles the recoverable slot count, so an 8-slot schema
wants D ≥ 4096 and a 16-slot schema is a re-design, not a bigger dimension.

## Sharding

Same total item count `N`, one collection versus shards holding 16 items each.

| Total N | Unsharded recall | Sharded at pool 16 |
|---|---|---|
| 128 | 62% | 100% |
| 512 | 7% | 100% |
| 2048 | 0% | 100% |

Sharding is the only lever that holds recall flat as `N` grows. Route across
shards with [`POST /db/{db}/compose/read`](../api/algebra.md), which weights
each collection by how well the query matches it.

## Multi-hop recall

Per-hop recall multiplies. A chain of `h` hops recalls at roughly `r^h`, where
`r` is the single-hop recall at that pool size.

| Pool N | 1 hop | 3 hops | 6 hops |
|---|---|---|---|
| 16 | 100% | 100% | 100% |
| 128 | 63% | 17% | — |

A traversal of any depth wants pool sizes at or under 16. Deep chains are a
sharding problem before they are a dimension problem.

## Live engine vs. numpy

The engine's walls sit one notch tighter than the pure arithmetic, because a
read also pays for competitive placement, activation over `k` locations, and a
finite softmax temperature.

At D = 1024, ≥90% recall:

| β | Engine holds to N | numpy holds to N |
|---|---|---|
| 5 (default) | 16 | 32 |
| 30 | exact-nearest-neighbour ceiling | — |

β is a softmax temperature and costs the same to evaluate at any value, so a
soft β buys nothing. β = 30 is also the clamp floor for role cleanup
(`MIN_CLEANUP_BETA`). Raise `beta` in `db.toml`, or read with
[`attention/mdl`](../api/reads.md) and let the engine pick β per query from the
codebook's geometry.

## Location saturation

A location saturates at roughly `d/32` writes, which is the `tau_overload`
default. Watch it with `GET …/collections/{name}/stats`:

| `stats.max_write_count` vs. `tau_overload` | State |
|---|---|
| < 0.5 × | Healthy. Locations are well below the wall |
| 0.5–1.0 × | Approaching. Splits are imminent for the hottest locations |
| ≈ 1.0 × and `num_locations` climbing | Splitting normally. Expected steady state |
| > 1.0 × and `num_locations` flat | Splits are not firing. Check `competitive = false`, or an `mdl_gate` that finds the traffic coherent |

`avg_write_count` far below `max_write_count` means the load is concentrated on
a few locations — a skewed key distribution, not a capacity problem. Reads
against the cold locations are fine; reads against the hot one blend.

## Choosing under a wall

| Symptom | Lever |
|---|---|
| Reads blend several stored patterns | Raise `beta`, or use `attention/mdl`; lower `k` |
| Recall falls as the collection grows | Shard. More collections, not more dimensions |
| A schema loses slots as you add fields | Raise D (√ scaling), or split the record across collections |
| A traversal loses the trail after two hops | Shrink pools to ≤ 16 |
| `max_write_count` past `tau_overload`, locations flat | Lower `tau_overload`, or turn off `mdl_gate` |
| Location count grows without bound | `POST …/compress` to merge to the MDL minimum |

## Related

- [Parameters](parameters.md) — every knob these walls respond to.
- [Behaviour](behaviour.md) — what a capacity failure looks like from the API.
- [How the memory works](../explanation/associative-memory.md) — why the walls
  are where they are.
- [Tune a database](../how-to/tune-a-database.md) — changing the knobs.
