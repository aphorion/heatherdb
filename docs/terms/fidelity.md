# Fidelity

How much the memory had to change your query to make it into an answer.

## Like you're twelve

You hum a tune at a friend. If they name the song instantly, your humming was
close. If they get there after a long pause and some squinting, it was rough —
same answer, less confidence, and it would be useful to know which of the two
just happened.

Fidelity is that "how sure was that". It costs one line to compute from the
answer you already have, instead of leaving you to guess.

<!--figure:reconstruct-->

## Precisely

Fidelity is the cosine between the query you submitted and the state the memory
settled into — the caller computes it from the read's own output; there is no
`fidelity` field on the wire. It scores their agreement — high when the input was already close to a stored
[[Attractor|attractor]], low when reconstruction had to travel.

It is what makes reconstruction safe to build on. A low-fidelity answer is not a
wrong answer; it is an answer the system is telling you not to lean on, which is
the same signal you would want from a classifier that can abstain. Thresholding
on it is how anomaly detection, cleanup, and "I don't know" all get expressed
without a second model.

## Related

[[Attractor]] · [[Associative memory]] · [[Cosine similarity]]
