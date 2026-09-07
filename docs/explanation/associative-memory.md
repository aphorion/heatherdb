# How the memory works

HeatherDB is not a key-value store and not a nearest-neighbour index. It stores
patterns distributed across a self-organising network of hard locations and
reconstructs them from approximate queries. This page explains what that
actually means mechanically, and what it costs.

## Hard locations

The unit of storage is a **hard location**, and it carries three things:

- an **address** — a unit vector saying where in the space it sits,
- a **counter** — an accumulator holding the superposed content written there,
- a **write count** — how much evidence it has absorbed.

Reading returns *patterns*, which are normalised counters. Routing uses
addresses. Keeping the two separate is what lets a location move toward the
data that keeps arriving without losing what it has already stored.

The number of hard locations is not a setting. It is grown and pruned by the
write rule, and it is best read as a **diagnostic of data complexity**: 200
similar vectors might become 51 locations; 200 unrelated ones will become many
more.

## Writing: select, update, regulate

Every write runs three phases in a single pass. There is no separate training
step — the memory *is* the model, updated in place, on every write.

### Select

The *k* nearest locations to the incoming address are activated (cosine
similarity; addresses are unit vectors, so a dot product is enough). Activation
weights are `wⱼ = max(sim, 0) / max_j sim`, so the best match gets 1.0 and the
rest are relative to it.

One of the activated set is the **winner**, and it is not simply the closest.
Winner selection applies a *conscience*:

```
S_eff(j) = sim(x, aⱼ) − γ · nⱼ / Σ nᵢ
```

A location that has been winning a lot is penalised in proportion to its share
of the accumulated evidence. This is what stops one location from swallowing a
whole region while its neighbours stay empty — the classic dead-unit problem in
competitive learning.

On a cold start — an empty index, which is the default because `l_0 = 0` means
data-seeded rather than pre-seeded with random vectors — the first write simply
becomes the first location.

### Update

One pass over the activated set does two things at once.

**Counter accumulation.** Each activated location adds `wⱼ · counter` to its
counter and `wⱼ` to its write count. Content is superposed, not appended; this
is why a stored vector cannot later be subtracted out.

**Address migration.** Each activated location moves toward the input:

```
aⱼ ← normalize(aⱼ + lr · (x − aⱼ))
```

The winner moves at `η_eff = η / (1 + nⱼ / τ_damp)` — a rate that damps as it
accumulates evidence, so a well-established location stops drifting. Neighbours
move at 10% of that. The global learning rate decays multiplicatively by
`lambda` on every write, floored at `eta_min`.

### Regulate

Topology maintenance, which is where the location count actually changes.

**Novelty split.** If the best match was worse than `tau_split`, nothing in the
memory covers this input, so a new location is spawned from it. It inherits the
activated set as its initial neighbours.

**Overload split.** If the winner's write count has passed `tau_overload`, it is
saturated: a perturbed copy is spawned, and the parent's counter and write
count are halved between the two. The default `tau_overload` is `max(d/32, 8)`,
which tracks the measured per-read capacity wall — a location splits *as* it
reaches the wall rather than well past it.

**Local dedup.** If both splits fired and produced near-identical children, they
are merged. Children are never merged back into existing locations: overload
children are deliberately near their parent, and novelty children are far from
everything.

### The MDL gate

With `mdl_gate = true`, both thresholds are replaced by one rule. The cost of a
new location is `engram_bits = log2(L+1) + log2(d/32)`, and the residual
surprise of a write that matched at cosine `s` is `−log2(s)` bits. A location
spawns when a single contact is surprising enough to pay for its own engram
outright, and splits when *accumulated* surprise — added a write at a time to
whichever location absorbed it — clears the same bar.

Two properties fall out. The address term rises with `L`, so **the bar for a new
location self-anneals as the index fills**. And a high-traffic but *coherent*
location carries almost no debt, so it does not split just because it is
popular — which the write-count threshold cannot distinguish.

## The navigable graph, for free

Every write already computes which *k* locations co-activated. Recording that
costs `O(k²)` against an `O(L·d)` activation step — effectively nothing — and it
builds a graph that mirrors the data manifold.

Capacity per location adapts: `max((k−1)·⌈ln L⌉, 2k)`. The logarithmic term
supplies the long-range links that make the graph navigable in `O(log L)` hops;
the `2k` floor keeps it connected while `L` is small.

Reads use it once it is worth using — at least 100 locations and an average
neighbour count of at least `k/2`. Below that, brute-force activation over a
contiguous address matrix is faster than pointer-chasing, and the engine uses
that instead.

Graph search is a single greedy probe: enter at the landmark (one of the 32
most-written locations) closest to the query, then descend through neighbour
edges, gathering each hop's unvisited neighbours into one contiguous buffer and
scoring them as a single batched mat-vec. Cost is
`O(landmarks·d + hops·neighbor_cap·d)`, with hops `≈ O(log L)` — near-constant
query cost as the collection grows, because it follows the manifold instead of
scanning it.

The graph is not bolted on. It is what competitive learning was building all
along.

## Reading: a Hopfield loop over the activated set

The short version of this section is the glossary entry for the
[Hopfield read](../terms/hopfield-read.md).

Activation picks the *k* most relevant locations. Then:

```
ξ₀ = normalize(query)
repeat up to t_max:
    sims  = cosine(ξ, addressⱼ)
    α     = softmax(β · sims)
    ξ_new = normalize(Σ αⱼ · patternⱼ)
    stop if cosine(ξ, ξ_new) > 1 − epsilon
    ξ    ← ξ_new
```

The activation set is fixed at the start; only the weights over it move. Each
iteration re-weights toward whichever locations the *current estimate* resembles,
which is what carves basins of attraction: a noisy query is pulled toward the
nearest genuine attractor rather than staying where it landed.

`strategy: "fast"` does a single step instead of iterating. `analyze` returns
the same computation with `iterations`, `converged`, and the final weights
exposed.

That softmax is doing three jobs at once, and the identification is structural
rather than an analogy:

| Read it as | And it is |
|---|---|
| an attention head | `softmax(β · Q·Kᵢ)` — the same operation |
| a Hopfield network | basins of attraction; error-correcting recall |
| a posterior under a mixture model | soft clustering over prototypes, weighted by accumulated evidence |

## Fidelity

The cosine between what you queried and what came back is the **fidelity**, and
you get it on every read at no extra cost.

- **High** — a strong attractor exists here. The pattern is well represented.
- **Low** — novel, underrepresented, or in conflict with what is already stored.

That single number does the work of several purpose-built models: anomaly
detection, cold-start detection, regime-change detection, capacity monitoring.
Nothing extra is trained for any of them.

What counts as "high" is collection-specific and you should calibrate it, not
inherit a number from documentation. Measure fidelity across a batch you believe
is in-distribution, measure it across random vectors, and put the threshold in
the gap.

## Capacity, and where it bites

Superposition has a limit. Roughly `d/32` patterns can be superposed into one
location before per-read recall degrades — which is exactly why `tau_overload`
defaults to that number.

The failure mode is worth recognising: reads that blend several stored patterns
into a plausible-looking vector that matches none of them. Fidelity is your
detector; it drops when this is happening.

Three levers when it does:

- **Raise `beta`**, or use `attention` with an explicit temperature, so the
  softmax concentrates rather than averaging.
- **Use `attention/mdl`** and let the engine pick β per query from the
  codebook's own geometry, or `attention/calibrate` to pick one for the whole
  collection.
- **Lower `k`** so fewer locations enter the mix at all.

Note the asymmetry between the two `similarity` fields you will see:
`analyze` reports cosines, `attention` reports raw dot products `Q·Kᵢ`. Same
field name, different quantity.

## What it will not do

- **Deletion does not un-write.** Deleting a document removes it from the index
  and from every posting list, so it can no longer be retrieved or cited. Its
  contribution to the merged counters stays. Superposition cannot be un-added;
  rebuild the collection if you need the influence gone.
- **Dimension cannot change.** The codebook is dimensioned. Create a new
  database.
- **It is not a nearest-neighbour index.** A read returns a reconstruction,
  which may be a vector no one ever wrote. If you need "give me back exactly the
  row I put in", write documents and use `documents/query`, or set
  `competitive = false` to make the collection an exact growing key→value store.
- **It does not filter.** Metadata is stored and returned, never indexed. No
  `where`, no sort, no range query.

## Related

- [Dreaming](dreaming.md) — the same write rule, run against the memory's own
  contents while idle.
- [Vector algebra](vector-algebra.md) — composition on top of the memory.
- [Tune a database](../how-to/tune-a-database.md) — the knobs, symptom by symptom.
