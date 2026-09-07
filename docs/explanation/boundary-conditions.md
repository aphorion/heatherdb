# Boundary conditions

The regime in which a storage-native store is the right instrument, stated as
properties rather than caveats.

## Saturation looks like confidence

A pool past its capacity does not report doubt. It returns a blended vector that
matches no stored item, and the blend can sit close to the query, so the read
looks decisive.

This inverts the intuition that a system under strain hesitates. Superposition
degrades by *averaging*, and an average of many things is a smooth, plausible
thing. The detector is not the read's own confidence at one query; it is the
distribution of [fidelity](../terms/fidelity.md) across a batch, compared
against the same measurement taken when the pool was within budget. See
[capacity and interference](capacity-and-interference.md) for the walls, and
[abstention](abstention.md) for why a threshold has to carry its calibration
load.

## Route, do not superpose everything

Superposing all data into one pool makes reads regress toward the base rate,
because every item competes with every other.

| Arrangement | AUC |
|---|---|
| Single superposed pool | 0.664 |
| Frequency baseline | 0.703 |
| Routed over the same data | 0.924 |

The superposed arrangement does not beat simply predicting by frequency. The
same rows, partitioned so that each read touches only the relevant pool, reach
0.924. Nothing changed but the routing.

The same property appears in retrieval geometry: one global collection loses to
plain cosine over the raw rows by a factor of two, where per-entity collections
win. **Partitioning is not an optimisation; it is what makes the arrangement
informative.** A collection per entity, per tenant, per schema type, or per time
bucket is the default, and the requirement it imposes is that the routing key be
known at query time.

## Order is not recoverable from binding alone

Circular convolution commutes exactly: `a ⊛ b = b ⊛ a`. A structure built only
from [binding](../terms/bind.md) and [bundling](../terms/bundle.md) therefore
contains no information about the order of its operands — not "little", none.
The two orderings are the same vector.

Four different abelian encodings of a directed relation sit at exactly chance.
That is the expected result, not a shortfall of the encodings: a commutative
operator cannot represent an antisymmetric fact, and no choice of atoms changes
that.

Order requires a non-commutative operator. [Permutation](../terms/permute.md)
supplies one — permuting one operand before binding breaks the symmetry, and the
inverse permutation recovers the role. Any directed relation, sequence position,
or asymmetric predicate needs it.

<!--figure:permute-->

## Structure earns its keep on the store side

Structured keys — role/filler bundles built by binding — underperform plain text
embeddings when used as *retrieval* geometry:

| Key | AUC |
|---|---|
| Structured (bound roles) | 0.740 |
| Plain text embedding | 0.789 |

The mechanism is direct: binding multiplies atom mismatch. Two records that
differ slightly in one atom produce bound vectors that differ by the product of
that mismatch across every bound term, so near-matches are pushed apart rather
than kept near. A metric wants the opposite — graded degradation under small
differences.

The conclusion is not that structure is a mistake; it is that structure is a
*storage and composition* representation, not a similarity metric. Structured
keys support unbinding a field, substituting a role, and composing records.
Retrieval ranks better on the embedding. A system that uses each for what it is
good at writes structure and retrieves on embeddings.

## Interference, not distance, is the long-context wall

A large context does not fail because the relevant item is far away — a read
reaches any location in `O(log L)` hops through the neighbour graph, and cost is
near-constant in collection size. It fails because more stored items means more
terms competing in every superposition, and recall falls as occupancy rises
against the `D/32` budget.

The practical consequence is that the levers that fix a long-context problem are
capacity levers — sharding, dimension, read temperature — and not indexing
levers. Adding a faster search over a saturated pool retrieves the same blend
faster.

## What belongs in ordinary code

The engine is for **algebra and reconstruction**: composing structure, taking it
apart, completing partial patterns, and reporting how well a query matched.

Two neighbouring jobs do not belong in it.

**Ranking.** Scoring a modest candidate set by cosine and sorting is a few lines
of ordinary code, exact, and cheaper. A reconstruction read answers a different
question and returns a vector that may not be any stored row.

**Small exact lookups.** Retrieving a known row by a known key is a hash-map
operation. Superposition cannot promise exactness, and the store's answer to
"give me back exactly what I put in" is a document query or a
`competitive = false` collection — that is, a plain key→value store wearing the
engine's interface.

Two further hard boundaries, stated once:

- **Deletion does not un-write.** A deleted document leaves the index and every
  posting list, but its contribution to merged counters remains. Rebuild the
  collection to remove influence.
- **Dimension is fixed per database.** The codebook is dimensioned; changing it
  means a new database and a re-write.

## Related

- [Storage native intelligence](storage-native-intelligence.md) — the claim
  these boundaries bound.
- [Capacity and interference](capacity-and-interference.md) — the walls and the
  escape.
- [What similarity means](what-similarity-means.md) — encoding decisions that
  determine which of these boundaries you meet.
- [Abstention](abstention.md) — detecting the saturated regime.
- [Vector algebra](vector-algebra.md) — [permute](../terms/permute.md) and the
  non-commutative operator.
