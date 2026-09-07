# Bundle

Adding several vectors into one of the same size that still resembles all of
them.

## Like you're twelve

Play three notes at once. What comes out of the piano is one sound, and a
trained ear can still pick out each note inside it. The chord is not the average
of the notes — it is all of them, at the same time, in one signal.

Bundling is the chord. Three ideas go in, one vector comes out, and the vector
is still recognisably each of the three.

<!--figure:bundle-->

## Precisely

Bundle is weighted summation, normalised by default. The result has non-trivial
similarity to every input, which is the exact opposite of what [[Bind|binding]]
does — and the two together are what make records representable: bind each
role to its filler, bundle the bindings, and you have one vector for the whole
record.

Weights may be negative, which is subtraction rather than an error: bundling
with a negative weight removes a concept from a mixture.

Capacity is finite. Each additional term adds interference, so a bundle of a
handful of items reads back cleanly and a bundle of hundreds does not. This is
[[Superposition]], and the ceiling on it is set by
[[High-dimensional space|dimensionality]].

## Related

[[Superposition]] · [[Bind]] · [[Cleanup memory]]
