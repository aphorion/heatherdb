# Rollout

Running a prediction forward more than one step by feeding each answer back in
as the next question.

## Like you're twelve

Before you play a move in chess you imagine it, then imagine the reply, then
your answer to that. Each guess rests on the one before, which is why thinking
six moves ahead is much harder than thinking one — a small mistake early makes
everything after it wrong.

That chain of imagined steps is a rollout, and the reason your chess brain gets
fuzzy at depth is the same reason a memory does.

## Precisely

A rollout applies a [world model](world-model.md) repeatedly: predict the next
state, then predict from that prediction. Each step's error becomes the next
step's input, so accuracy decays with the horizon — a rollout is a compounding
of approximations, not a lengthening of one.

A [cleanup](cleanup-memory.md) step between steps is what makes any depth
reachable. Without it, each prediction carries the last one's noise forward and
the chain degrades within a few steps. With it, each step is snapped back onto a
stored state before continuing, which resets the noise floor rather than
accumulating it.

How far a rollout survives is set by how crowded each read is rather than by
depth itself. In a pool of 16 items, six chained steps recover perfectly; in a
pool of 128, the first step is already unreliable — per-step recall multiplies,
so [capacity](../reference/capacity.md) is the real limit.

## Related

[[World model]] · [[Cleanup memory]] · [[Plan]] · [[Interference]]
