# What similarity means

The encoding decides which two things count as similar, and that decision is
the entire design surface of a storage-native system.

A store that computes over arrangement can only express distinctions its
geometry carries. Whatever the encoder discards is invisible to every read,
forever, and no amount of tuning recovers it. This page is the set of standing
rules for building that geometry.

## Cosine reads direction, not magnitude

[Cosine similarity](../terms/cosine-similarity.md) is scale-invariant by
construction: `cos(x, 2x) = 1`. A vector's length carries no information into
any read.

The consequence is concrete. Put a transaction amount in a component and a
\$5,000 transaction points in the same direction as a \$4.50 one; the store
treats them as the same event. Magnitude becomes visible only when it is
encoded as *direction* — most simply by bucketing the amount and binding the
bucket as a one-hot symbol, so `amount:0–10` and `amount:1k–10k` are
[orthogonal](../terms/orthogonality.md) atoms rather than two lengths of the
same arrow.

The same rule covers counts, durations, prices and scores. If a difference in
size should change what a row is near, that difference has to change the row's
direction.

## Cyclic quantities need two components

Hour-of-day, day-of-week, angle, phase and heading wrap. A single scalar
component puts hour 23 maximally far from hour 0, which is the opposite of the
intended metric. Encoding the quantity as `(sin θ, cos θ)` makes distance on the
circle the distance in the embedding, so 23:00 and 00:00 sit adjacent and 06:00
and 18:00 sit opposed.

## Per-feature weights are the metric

Concatenating feature blocks and scaling each block by a weight is not
pre-processing. Under cosine, the weights *are* the similarity metric: a block
scaled by `w` contributes `w²` of the total inner product. Doubling the weight
on a block doubles the influence of that block's agreement on every ranking the
store produces.

This is the knob that replaces "choose a distance function". Setting it is a
modelling decision, and it is auditable — the weights sit in the encoder, not
inside fitted parameters.

## Reaching the store's dimension

A collection is dimensioned, and every write must match it. Two ways to move a
shorter feature vector into a wider store:

| | Effect on geometry |
|---|---|
| Zero-pad to `d` | norm-preserving; all pairwise cosines unchanged |
| Random projection up | mixes components; approximately, not exactly, preserving |

Zero-padding leaves the metric you designed exactly intact, which is why it is
the default choice for widening. Projection is the tool for the other
direction — reducing.

## Anisotropy and mean-centering

Learned text and image embeddings are anisotropic: the representations occupy a
narrow cone rather than filling the sphere, so every raw pairwise cosine sits
near 1 and the spread that carries meaning is a thin band on top of a large
constant. A store built over raw embeddings has a geometry in which everything
is similar to everything, and reads cannot discriminate.

Subtracting the corpus mean before writing removes the shared component and
restores the full range. Centering is not cosmetic; it is what makes the cone's
internal structure into the geometry the store actually indexes. The mean must
be computed once over a representative sample and then applied identically to
every write and every query — a query centered against a different mean lands
in a different space.

## Dimension is an interference knob

More dimensions means more room, and more room is not uniformly better.

With ample room, stored patterns barely interact, and a read degenerates toward
returning whichever single stored item is nearest — a lookup, not a
reconstruction. Reducing the dimension forces patterns to share structure, and
[superposition](../terms/superposition.md) starts doing work: reads complete
missing parts and correct noisy inputs, because the geometry has to generalise
to fit.

Reducing 384-dimensional sentence embeddings to 128 by a fixed-seed
orthogonalised random projection is routine. Three properties make it safe to
treat as a standard step:

- **Orthogonalised** — the projection's rows are made mutually orthogonal, so
  the reduction is a rotation-and-truncate rather than a random smear, and
  pairwise angles are preserved as closely as the target dimension allows.
- **Fixed seed** — the same matrix must be applied to every write and every
  query, so the seed is part of the collection's contract, not a runtime choice.
- **One direction only** — the projection reduces; it is not inverted. What
  comes back from a read lives in the reduced space.

The dimension trades two failure modes against each other:

| Dimension relative to data | Behaviour |
|---|---|
| Too high | reads converge to nearest-neighbour lookup; little completion |
| Matched | reconstruction and error correction |
| Too low | [interference](capacity-and-interference.md); reads blend unrelated patterns |

Capacity scales with dimension (`≈ d/32` per pool), so lowering the dimension
buys generalisation and spends headroom. The measured walls are in
[capacity and interference](capacity-and-interference.md).

## A checklist for an encoder

1. Every distinction that should affect retrieval changes *direction*.
2. Magnitudes that matter are bucketed or otherwise made directional.
3. Cyclic quantities are `(sin, cos)` pairs.
4. Block weights are set deliberately; they are the metric.
5. Learned embeddings are mean-centered against a fixed corpus mean.
6. Widening is zero-padding; narrowing is a fixed-seed orthogonalised
   projection.
7. Write path and query path apply byte-identical encoding steps.

Rule 7 is the one that fails silently. An encoder that centers on write but not
on query produces a store that looks healthy and answers nothing well.

## Related

- [Storage native intelligence](storage-native-intelligence.md) — why the
  encoding carries the whole design.
- [Capacity and interference](capacity-and-interference.md) — what the chosen
  dimension buys.
- [Vector symbolic architectures](vector-symbolic-architectures.md) — encoding
  structure rather than features.
- [Vector algebra](vector-algebra.md) — [bind](../terms/bind.md) and
  [bundle](../terms/bundle.md) as encoding primitives.
- [High-dimensional space](../terms/high-dimensional-space.md) — why
  near-orthogonality holds.
