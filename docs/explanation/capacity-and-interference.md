# Capacity and interference

Capacity is a property of a pool, not of a database: roughly `D/32` patterns can
be [superposed](../terms/superposition.md) into one pool before per-read recall
degrades.

## The linear wall

Superposing patterns into a single accumulator makes each one a signal against
the sum of the others as noise. Recall holds while the signal clears the
interference floor, and collapses over a narrow range once it does not.

| Dimension `D` | Measured wall |
|---|---|
| 512 | 32 |
| 1024 | 32 |
| 2048 | 64 |

The wall scales with `D`, which is why `tau_overload` defaults to
`max(d/32, 8)`: a hard location splits *as* it reaches the wall rather than well
past it. The number to keep is the ratio — a pool holds about one item per 32
dimensions, and the same pool at twice the width holds about twice as many.

The failure at the wall is not an error. Reads return a plausible vector that
blends several stored patterns and matches none of them.
[Fidelity](../terms/fidelity.md) is the detector: it falls when blending starts.

## Schema capacity is quadratic

A schema — a record built by [binding](../terms/bind.md) each field key to its
value and [bundling](../terms/bundle.md) the results — consumes capacity faster
than a flat pool, because each additional slot both adds a term to the bundle
and adds a competitor to every unbind. Reliable slot count therefore goes as the
square root of the pool budget:

```
n_slots ≲ √(D / 32)
```

At `D = 1024`, per-slot recovery against slot count:

| Slots | Recovery |
|---|---|
| 2–3 | 100% |
| 4 | 94% |
| 6 | 22% |
| ≥ 8 | 0% |

The square-root law says the cure is width, and it is — a fixed 8-slot schema
recovers as `D` rises:

| `D` | Recovery at 8 slots |
|---|---|
| 1024 | 0% |
| 2048 | 15% |
| 4096 | 96% |
| 8192 | 100% |

Note the shape: the transition is sharp in both tables. Capacity does not
degrade gently with slot count or dimension; it holds, then goes.

## The escape is sharding, not dimension

Raising `D` buys capacity linearly and costs storage and compute linearly, so it
is a bounded lever. Partitioning the same items across pools is unbounded,
because capacity is per pool and the total is the sum.

At pool size 16, with the same total item count `N` split across pools versus
superposed into one:

| Total `N` | Unsharded recall | Sharded recall |
|---|---|---|
| 128 | 62% | 100% |
| 512 | 7% | 100% |
| 2048 | 0% | 100% |

Sharded recall is flat in `N`. The unsharded column is the `D/32` wall
reasserting itself as the pool fills, and no dimension in reach closes the gap
at `N = 2048` — the width required grows with the item count, while the shard
count required does not change the *per-pool* load at all.

This is the standing design rule: **route into pools, do not superpose
everything**. A pool per entity, per tenant, per time bucket, or per schema type
keeps every read inside its own capacity budget. The cost is that the routing
key must be known at query time, which is a modelling constraint, not a tuning
one.

## Per-hop recall multiplies

A composed structure read across several levels succeeds only if every level
succeeds. Per-hop recall `r` over `h` hops gives end-to-end recall `r^h`, so
90% per hop is 59% at five hops and 35% at ten.

Two consequences follow. Depth is expensive in a way that width is not, and
per-hop recall must be driven near 1 rather than merely "good" —
[the cleanup loop](the-cleanup-loop.md) is what does that, by resetting the
noise floor between levels instead of letting it compound.

## Read temperature scales with pool size

The read's inverse temperature `β` sets how sharply the softmax concentrates
over the activated set. A fixed `β` that separates a small pool averages a large
one, because the number of competing near-matches grows with `N` while the
softmax's selectivity does not.

At `D = 1024`:

| `β` | Behaviour |
|---|---|
| 5 | ≥90% recall only to `N = 16` |
| 30 | attains the exact-nearest-neighbour ceiling |

So `β` is not a global constant to inherit from documentation; it is a function
of how many items compete in a pool. `attention/mdl` selects it per query from
the codebook's own geometry, and `attention/calibrate` selects one for a whole
collection — both express the same property, that the right temperature is a
consequence of the stored geometry rather than a free parameter.

## Reading the symptoms

| Symptom | Cause | Lever |
|---|---|---|
| Fidelity drops as writes accumulate | pool past `D/32` | shard; raise `tau_overload` headroom |
| Reads return plausible non-items | blending at the wall | shard; raise `β`; lower `k` |
| Deep composition fails, shallow works | per-hop recall multiplying | cleanup between levels |
| A schema loses slots as it grows | quadratic schema law | fewer slots per record, or raise `D` |
| Small pools sharp, large pools mushy | `β` fixed against growing `N` | `attention/mdl` or calibrate |

## Related

- [How the memory works](associative-memory.md) — `tau_overload`, splitting and
  the read loop.
- [What similarity means](what-similarity-means.md) — dimension as an
  interference knob on the encoder side.
- [The cleanup loop](the-cleanup-loop.md) — keeping per-hop recall near 1.
- [Boundary conditions](boundary-conditions.md) — interference, not distance, as
  the long-context wall.
- [Superposition](../terms/superposition.md),
  [orthogonality](../terms/orthogonality.md) — the arithmetic behind the wall.
