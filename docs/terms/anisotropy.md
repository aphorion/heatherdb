# Anisotropy

The tendency of learned vectors to crowd into a narrow cone, so that everything
looks similar to everything else.

## Like you're twelve

Ask a hundred people to point at "something they like". If they all end up
pointing roughly north-east, then measuring who agrees with whom is useless —
everyone agrees with everyone. The information is in the small differences, and
those differences are hidden by the fact that the whole crowd is leaning one
way.

Subtract the direction the crowd is leaning, and the disagreements appear.

<!--figure:orthogonal-->

## Precisely

Vectors produced by a trained encoder are not spread evenly over the sphere.
They occupy a narrow region, so pairwise [[Cosine similarity|cosine
similarities]] between unrelated items sit high — often 0.6 to 0.9 — and the
usable signal is a thin band on top of a large constant.

The correction is to subtract the mean of the population before comparing.
What remains is the deviation from the average, which is where the distinctions
live. This applies to pooled embeddings, spectral features and log-magnitude
features; independently drawn random symbols are already centred in
expectation and need nothing.

The mean is part of the model. Persist it, apply the same one at query time,
and recompute it when the corpus shifts. Keep the un-centred vector if you need
to return the original.

Anisotropy is the reason a similarity threshold that worked on random vectors
fails on real ones: the whole scale has moved, not just the noise.

## Related

[[Cosine similarity]] · [[Orthogonality]] · [[High-dimensional space]] ·
[[Vector]]
