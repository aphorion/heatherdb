# Consolidation

Replaying what a memory already holds through the rule that wrote it changes
what the memory holds. Nothing is ingested and nothing is queried, yet the
collection afterwards is not the collection before — because the write rule is
order-dependent, and a stored collection is partly a record of the order it
arrived in.

The operational side is [Use dreaming](../how-to/use-dreaming.md); the loop's
mechanics and its knobs are in [Dreaming](dreaming.md). This page is about why
the operation has an effect at all, and what its output measures.

## The write rule is a competition, and it can be re-run

Every write selects the *k* nearest locations, migrates their addresses toward
the input, accumulates into their counters, and then regulates topology —
spawning on novelty, splitting on overload, dedup'ing. Each of those decisions
is taken against the index **as it stood at that moment**. An early write
established a location against an empty neighbourhood; a later one arrived to
find the region already crowded. Normal operation never revisits either, because
a write only touches what it activated.

Re-presenting a stored address puts it back into that competition against the
finished index. Three consequences follow, and they are properties of the rule
rather than of the data:

- **Addresses move to where their evidence is.** A location that absorbed a
  spread of inputs sits wherever the first of them landed, damped by
  accumulated evidence. Replay migrates it toward the centre of what it actually
  represents.
- **Deferred splits fire.** A neighbourhood that has since become dense supports
  a novelty or overload split that the same input could not justify when it
  first arrived.
- **Converged locations collapse.** Locations that migrated together are
  redundant, and the merge pass folds them into one, enforcing capacity as it
  goes.

## What recurs deepens

Depth in this memory is [superposition](../terms/superposition.md) weighted by
write count. A pattern experienced many times has been accumulated many times
into the same counter, so it dominates that counter's direction and it is
represented by locations dense enough to survive competition. A pattern
experienced once holds a single location with a write count of one.

Replay preserves that ordering rather than flattening it: the recurring pattern
is re-presented at each of the locations that carry it, and each of those
re-competes and sharpens. The one-off is re-presented once, and if a stronger
neighbour has migrated over it, it merges away. Consolidation is not compression
that treats all stored items alike — it is competition run again, and repetition
is what wins a competition.

What it will not do is collapse things that genuinely differ. Three well
separated patterns, each written twice, plus a near-duplicate of the first: a
dream pass folds the near-duplicate away and leaves at least three locations,
and every distinct pattern still has a location within cosine 0.9 of it (the
invariant the engine's own `replay_merge_preserves_attractors_and_dedups` test
asserts). Merging is licensed by structure, not by a target size.

## Location count is a measurement

The number of [hard locations](../terms/hard-location.md) is not a setting, and
after a dream pass it is a reading: *this is how much distinct structure the
write rule can find in what you stored*. Two collections of the same size settle
at different counts, and the same collection settles at a different count after
consolidation than before.

Measured on ten episodes that share one context and carry two distinct laws, at
`D = 256`:

| Pass | Locations in | Locations out |
|---|---|---|
| Replay | 10 | 9 |
| One ladder rung | 10 | **2** |

Replay finds almost nothing to fold, because ten locations at one shared address
are already as merged as address similarity can make them. The rung finds two,
which is the number of laws present. Same data, same engine, two different
questions — and in both cases the count is an answer, not a configuration.

This is why the count is worth watching. A pass that leaves it unchanged with
`merged: 0` is reporting that there is no redundancy left to find; a rung that
returns as many families as it was given items is reporting that it found no
shared law. Both are results.

## Where merging meets the capacity wall

Consolidation trades location count against per-location load, and
[capacity](capacity-and-interference.md) sets the exchange rate. Roughly `D/32`
patterns superpose into one counter before per-read recall degrades, which is why
`tau_overload` defaults to that number. Merging fewer, fuller locations is free
only while each stays under the wall; past it, reads return a plausible blend
that matches nothing stored.

The two regulators pull in opposite directions on purpose. Merge removes
locations whose addresses have converged — a cheap gain, since near-identical
locations hold near-identical content. Overload split removes locations whose
counters have saturated, halving evidence between parent and child. A dream pass
runs both, so a collection settles where redundancy is gone *and* no counter is
over the wall. [Fidelity](../terms/fidelity.md) is the detector for the second
failure: it falls when blending starts.

## The ladder: strip the context, keep the law

A location is a pair. Its address says which situation it answers to; its
counter holds what was written there. Where the counter was formed by binding
the situation into a relation, the situation cancels:

```
law = unbind(counter, address)
```

What is left is the relation the location carries, with its context removed. Two
locations from different situations that carry the *same* law now hold nearly
the same vector, which they did not before — the context was the thing making
them look different.

Grouping those laws is the second half, and it needs a routing decision that
address similarity cannot make. In a world with two parameters — a law and a
shift, say — a situation is about equally similar to its law-mates and its
shift-mates, so an address tells you only who the candidates are. The gate
supplies the rest: among candidates close enough in address, a relation joins
the family whose accumulated law it **coheres** with, and spawns a new family if
it coheres with none. Incoherence is novelty even when the address looks
familiar.

A family is then a `(context prototype, accumulated law)` pair — the same
`(address, counter)` shape as its input. The operation is closed under its own
output, which is what makes it a ladder: episodes give laws, laws give laws
about laws, and each rung is the identical operation applied one level up.

Each rung reads its own boundaries off its own inputs rather than a constant —
the address cut from the largest gap in the upper region of the pairwise
similarities, the coherence cut from Otsu's threshold, the valley between the
within-family and cross-family modes. Those boundaries are therefore properties
of the level. On the ten-episode set above the rung calibrated to
`tau_cohere = 0.585` with `tau_split = 0.0` and returned two families; the same
episodes after a replay pass calibrated to `tau_split = 0.998`,
`tau_cohere = 0.996` and returned three. A self-scaling threshold reads the
distribution it is given, so what the previous pass did to that distribution
propagates.

## What the loop costs in accuracy

Consolidation is lossless with respect to the hand-labelled alternative and its
absolute level is set by capacity, not by the loop. Printed in the research
repository, on a four-family benchmark with one member held out per family and
six seeds: the stock write rule recovers exactly four clusters at 100% purity
and 100% routing, and held-out generalization from the engine-discovered
families equals the oracle's at every seed — **71% against 71%** at `D = 1024,
β = 5`, **75% against 75%** at `β = 30`. Widening to `D = 4096` lifts the same
loop to **96%** against a 100% oracle, the residual being recall mixture at
`β = 5` over the family pool.

Pointing the same operation at its own output holds too. On a
three-laws × four-shifts world with one whole shift per law held out — families
that produced no episodes anywhere — the gated rung reaches 3 clusters at 100%
purity and **100%** zero-shot derivation against a 100% episode oracle, where
address-only routing reaches 88%, the nearest trained family's law reaches 0%,
and chance is 6%.

Read those together: the loop adds no error of its own. Where it lands is
governed by pool width, read temperature and the gate — the same three
quantities that govern every other read in the system.

## Cleanup, again

A rung is composition in depth, so it inherits the composition rule: every
`unbind` leaves a residual, and a residual fed into the next level becomes that
level's input error. What resets the floor between rungs is the same associative
read that resets it between rollout steps — the family prototype a relation is
routed to is a stored attractor, not the raw residual. [The cleanup
loop](the-cleanup-loop.md) is the general statement; consolidation is one of its
instances, with idle time as the schedule.

## Related

- [Dreaming](dreaming.md) — the loop, the modes, the configuration.
- [Use dreaming](../how-to/use-dreaming.md) — when to run it, and the traps.
- [Capacity and interference](capacity-and-interference.md) — the wall merging
  works against.
- [The cleanup loop](the-cleanup-loop.md) — why a rung needs a memory beneath
  it.
- [How the memory works](associative-memory.md) · [World
  models](world-models.md)
