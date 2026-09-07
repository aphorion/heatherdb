# Superposition

Several things stored in one vector at once, all still readable.

## Like you're twelve

A window at dusk shows you the garden outside and your own room reflected back,
overlaid in the same pane. Both pictures are fully there. Your eye can attend to
either one. Nothing was averaged, and nothing was filed in a separate place.

That is what a superposed vector is: several patterns occupying the same
storage, each recoverable, none of them in a slot of its own.

<!--figure:bundle-->

## Precisely

Superposition is the state produced by [[Bundle|bundling]]. It works because
[[High-dimensional space|high-dimensional]] vectors are near-orthogonal: each
term contributes almost nothing along the others' directions, so a query aimed
at one term is barely disturbed by the rest.

The trade is graceful degradation rather than a hard limit. As terms accumulate,
similarity to each individual term falls and the interference floor rises, until
a query can no longer tell a stored term from noise. Nothing breaks at that
point; results simply get less reliable — which is why [[Fidelity|fidelity]] is
reported rather than assumed, and why a [[Cleanup memory]] earns its place.

## Related

[[Bundle]] · [[High-dimensional space]] · [[Fidelity]]
