# Elastic Associative Memory (EAM)

The memory model HeatherDB implements: an associative store whose structure
reshapes itself as data arrives.

## Like you're twelve

Imagine a shelf that rearranges itself while you use it. Put three cookbooks on
it and it quietly slides them together; add a pile of maps and it makes a
region for those too, moving the cookbooks a little to make room. Nobody
reorganises the shelf on a Sunday. The rearranging *is* how things get put away.

That elasticity is the difference between a memory that learns and a filing
cabinet that fills.

<!--figure:learning-->

## Precisely

EAM is an [[Associative memory]] whose representational capacity and internal
organisation adapt to the data written into it, rather than being fixed at
build time. Writes are [[Online learning|online learning]] steps: each one
places a pattern, adjusts the neighbourhood it lands in, and updates the
navigable structure used to reach it.

The consequences are the ones the paradigm advertises — no separate index to
rebuild, no retraining to absorb drift, no ceiling declared before the data
arrives, and no [[Catastrophic forgetting]] when new patterns differ from old.

The mechanism is described in [[How the memory works]], and the formal account
is in the paper linked from the [documentation index](../index.md).

## Related

[[Associative memory]] · [[Online learning]] · [[Sparse Distributed Memory (SDM)]] · [[Pentti Kanerva]]
