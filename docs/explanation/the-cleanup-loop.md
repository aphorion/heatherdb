# The cleanup loop

Every algebra operation leaves its result approximate, so composition is
reliable only when a cleanup read sits between levels to reset the noise floor.

## Why approximation compounds

[Unbinding](../terms/unbind.md) a role from a [bundle](../terms/bundle.md)
returns the filler plus the interference of everything else the bundle carries.
The result is a vector *near* the intended item, not the item. Feed that
approximate vector into the next operation and its residual becomes the next
level's input error, which the next unbind amplifies in turn.

The measured contrast between a raw unbind and its distractors sets the scale of
what cleanup has to work with:

| Query | Similarity to correct filler |
|---|---|
| Unbind result | ≈ 0.72 |
| Distractor items | ≈ 0 |

A single hop is unambiguous — 0.72 against nothing is a clean win. What fails is
composition, because 0.72 is not 1.0 and the gap is the seed of the next level's
noise.

## The wall, with and without cleanup

Per-hop recall multiplies, so composition is bounded by the *pool* each hop
reads against rather than by depth itself. Measured multi-hop traversal at
D=1024, as the pool grows:

| Items in the pool | 1 hop | 2 hops | 3 hops | 6 hops |
|---|---|---|---|---|
| 16 | 100% | 100% | 100% | 100% |
| 64 | 100% | 80% | 77% | 67% |
| 128 | 63% | 23% | 17% | 3% |
| 256 | 33% | 7% | 3% | 0% |

A pool of 16 holds to six hops without loss. A pool of 128 fails before the
first one completes. The depth a composition survives is therefore not a
property of depth — it is set by how crowded each read is, which is
[capacity](capacity-and-interference.md) again.

Two things follow. Cleanup between hops is what makes chaining possible at all,
because it resets the per-level noise floor to the cleanup's own accuracy
instead of letting error compound. And keeping each pool small is what makes
depth cheap: with cleanup and per-node sharding, a hierarchy of 32,767 nodes is
traversed root-to-leaf and leaf-to-root at 100%, where a single global map over
the same tree scores 0%.

<!--figure:cleanup-->

## Cleanup is what turns a vector into a symbol

A raw unbind result is a point in space. A cleanup read against the stored
codebook returns the [attractor](../terms/attractor.md) that point belongs to,
which is a stored item — an identifiable symbol with an identity, not a
neighbourhood.

That is the whole mechanism by which a
[vector symbolic architecture](../terms/vector-symbolic-architecture.md) is
symbolic at all. The algebra composes; the
[cleanup memory](../terms/cleanup-memory.md) discretises. Neither half is
sufficient. In HeatherDB the cleanup memory is not a bolted-on item list — the
associative read path *is* the cleanup, performed against the stored data.

## Store intermediates; do not recompute them

A composed structure produces intermediate vectors at every level. Two
disciplines are available and only one of them works.

**Recomputing** an intermediate from its parents reproduces the *uncleaned*
result, because the parents' cleanup is not part of the arithmetic that produced
it. Every recomputation re-introduces the noise the cleanup removed, and the
compounding returns.

**Storing** the cleaned intermediate makes it an item like any other: it has an
address, it participates in future reads, and every reference to it starts from
the cleaned value rather than the approximate one. The store is therefore load
bearing during composition, not just at the end of it — cleaned intermediates
are the state of the computation.

The rule: **each level writes its cleaned result before the next level reads
it.**

## Which read to use as cleanup

The read has two strategies, and they behave differently under composition.

| Strategy | Behaviour |
|---|---|
| First contact (`fast`, single step) | recall; reports how well the query matched |
| Iterative (`HopfieldIter`) | settles into an attractor; converges or oscillates |

**First-contact reads recall.** One step of the softmax-weighted sum over the
activated set moves the query onto the codebook without erasing how far it had
to move — the similarity between the query and the result is a measurement, and
it is the signal [abstention](abstention.md) is built on.

**Iterative reads oscillate inside a subtraction loop.** Used as a cleanup
operator within an outer loop that subtracts each recovered item from a residual
— the standard way to decompose a bundle into its terms — the iteration's
attractor dynamics and the outer subtraction fight: the iteration pulls toward
whatever the residual currently most resembles, and the subtraction moves the
residual away from it, so successive passes can cycle between the same few
attractors rather than draining the bundle.

The property to hold onto: **an operator that always converges is not a
measurement**. A read that settles to similarity ≈ 1.0 by construction has
discarded the distance that made it informative. Use the iterative read where an
attractor is what you want, and the first-contact read wherever the result feeds
another decision.

## Related

- [Vector algebra](vector-algebra.md) — the operations whose results need
  cleaning.
- [Vector symbolic architectures](vector-symbolic-architectures.md) — why
  cleanup is the missing half of the algebra.
- [Capacity and interference](capacity-and-interference.md) — per-hop recall
  multiplying over depth.
- [Abstention](abstention.md) — the signal first-contact reads carry and
  iterative reads destroy.
- [Cleanup memory](../terms/cleanup-memory.md),
  [attractor](../terms/attractor.md).
