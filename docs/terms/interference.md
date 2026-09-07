# Interference

The noise every stored pattern contributes to every other one — the price of
keeping many things in the same space.

## Like you're twelve

Two people talking in a room is fine. Twenty people talking is a wall of sound,
and you cannot pick out any single conversation, even though every voice is
still physically there.

Nothing was deleted. There is just too much at once for any one thing to stand
out.

<!--figure:bundle-->

## Precisely

Stored patterns are not filed apart, so each one contributes a small amount to
what every read returns. That contribution is interference. It scales with how
many things share the space and with how similar they are, and it sets every
capacity limit in the system: a pool holds roughly `d/32` items before recall
degrades, and a single vector holds roughly `√(d/32)` distinct slots.

Interference is also the mechanism, not only the cost. Reconstruction works
*because* related patterns contribute together — the pattern you get back is
assembled from everything relevant, which is why it can be something no single
write contained.

Two dials control it. Dimension buys room: more directions means less overlap
between unrelated things, and too many means unrelated things stop interacting
at all, at which point the memory degrades toward nearest-neighbour lookup.
Pool size spends it: the escape from an interference wall is more pools, not
more dimensions.

## Related

[[Superposition]] · [[High-dimensional space]] · [[Orthogonality]] ·
[[Hard location]]
