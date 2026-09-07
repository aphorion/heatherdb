# Plan

A sequence of actions chosen by imagining where each one leads.

## Like you're twelve

Working out how to get to a friend's house across town, you do not try every
route with your feet. You picture a few — bus then walk, bike the whole way,
ask for a lift — and score them in your head before moving. The imagining is
cheap; the walking is not.

## Precisely

Given a [world model](world-model.md), planning is search over sequences of
actions: apply an action, clean up, apply the next, and score the state you land
in. There is no separate policy to train — the plan is produced by using the
memory, so it changes the moment the memory does.

This makes planning as good as the model underneath it and no better, with two
consequences. A long plan compounds approximations, because it is a
[rollout](rollout.md), so the useful horizon is bounded by how fast prediction
decays. And a plan is only trustworthy where the memory has been: over a region
it has never seen, the model will still return confident-looking states, which
is what the [fidelity](fidelity.md) of each predicted step is for.

The complement is [curiosity](curiosity.md). Planning exploits what the memory
holds; curiosity decides where to go and gather what it lacks.

## Related

[[World model]] · [[Rollout]] · [[Curiosity]] · [[Fidelity]]
