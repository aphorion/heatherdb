# Hard location

One entry in the codebook the memory grows for itself — an address it listens
at, and the pattern it has accumulated there.

## Like you're twelve

A library does not keep a shelf per book. It keeps a shelf per *subject*, and
every book that arrives goes to the shelf it most belongs on, nudging what that
shelf is about. Two hundred books might need fifty shelves. Nobody decided
fifty; it is what the books turned out to need.

A hard location is a shelf. You never create one, and you never say how many.

<!--figure:attractor-->

## Precisely

Each hard location holds an *address* — the direction it responds to — and a
*counter*, the accumulated pattern of everything written near it. A write finds
the locations nearest its input, picks a winner, adds to that winner's counter
and migrates its address slightly toward the new data. The set of all locations
is the codebook, and it is grown from the data rather than configured.

The count is a measurement, not a setting. Writing 1000 vectors drawn from five
underlying patterns produces a few hundred locations; writing 1000 unrelated
vectors produces more than a thousand, because there is more genuinely distinct
material to account for. Reported as `num_locations` in a collection's stats.

Locations saturate. Past roughly `d/32` writes a location blends what it holds,
and the engine splits it rather than letting it smear — which is why
[capacity](../reference/capacity.md) is counted per pool.

## Related

[[Attractor]] · [[Associative memory]] · [[Elastic Associative Memory (EAM)]] ·
[[Interference]]
