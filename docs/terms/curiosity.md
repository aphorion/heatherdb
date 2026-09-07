# Curiosity

Choosing where to go by finding where the memory is emptiest.

## Like you're twelve

Drop a child into a new playground and they do not wander at random, and they
do not need to be told what to look at. They go to the thing they have not seen
yet. When that becomes familiar they move on. Nobody hands out points; the
unfamiliarity itself is the pull.

## Precisely

An associative memory reports how well it recognises each query — the
[fidelity](fidelity.md) of the read. Used one way, that number supports
[abstention](../explanation/abstention.md): refuse to answer when the state is
unfamiliar. Used with its sign flipped, the same number is a direction: move
toward the least familiar state you can reach.

What comes out is directed exploration with no reward function, no map, and no
new mechanism — the signal was already being computed for another purpose. An
agent that always steps to its least familiar neighbour covers a space
substantially faster than one stepping at random.

Two ends of the scale are degenerate, and both are diagnosable from the same
number. A saturated memory recognises everything, so the gradient flattens and
curiosity stalls. An empty memory recognises nothing, so every direction looks
equally attractive and the choice carries no information. Curiosity is useful in
between, and it stops being useful when coverage plateaus — which is a signal to
stop exploring, not a failure.

## Related

[[Fidelity]] · [[Plan]] · [[World model]] · [[Held-out evaluation]]
