# Cosine similarity

How closely two vectors point the same way, from 1 (identical direction) through
0 (unrelated) to -1 (opposite).

## Like you're twelve

Two people stand in a field and point at things. If they point at the same tree,
their arms are parallel — that is 1. If one points at a tree and the other at a
cloud on the far side, the arms are opposite — that is -1. If one points north
and the other east, the arms are unrelated — that is 0.

Notice that how *long* their arms are never came into it. Only the direction.

<!--figure:orthogonal-->

## Precisely

Cosine similarity is the dot product of two vectors divided by the product of
their magnitudes: the cosine of the angle between them, independent of length.
That length-independence is why it is the default measure of similarity for
vectors of learned features — a doubled-magnitude vector means the same thing.

It is the score behind every similarity search in HeatherDB, and it appears in
answers as well as queries: a read reports the similarity between what you asked
and what came back, which is the basis of the [[Fidelity|fidelity]] signal.

## Related

[[Orthogonality]] · [[Vector]] · [[Fidelity]]
