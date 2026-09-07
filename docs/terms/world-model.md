# World model

Something that answers one question: if the world looks like *this* and I do
*that*, what happens next?

## Like you're twelve

You know what happens if you push a glass off a table. You have never done a
physics calculation about it, and you do not need to watch to find out — you
already know the glass ends up on the floor. Somewhere in your head is a
rehearsal of the world that runs faster than the world does.

That rehearsal is a world model, and you use it constantly. It is how you catch
a ball, how you decide a gap in traffic is too small, and how you notice that
someone has moved the furniture.

<!--figure:attractor-->

## Precisely

A world model is a function from a state and an action to the next state. In an
associative memory that function is a stored [binding](bind.md): the state is a
vector, each available action is a role, and the transition for a state is a
[bundle](bundle.md) of bindings, one per action —

```
M_s = Σ_a bind(a, s'_a)
```

Prediction is [unbinding](unbind.md) the action you are considering and then
[cleaning up](cleanup-memory.md) the result. Nothing is added to the store to
make this work; a memory of transitions *is* a simulator, and the same read
path that recalls a pattern predicts a consequence.

Two bounds follow. Prediction is approximate, so accuracy falls as a
[rollout](rollout.md) lengthens. And because binding commutes, two actions
applied in either order produce the same vector — order needs a
non-commutative operator such as [permutation](permute.md), or a sequence is
unrecoverable in principle rather than merely noisy.

## Related

[[Rollout]] · [[Plan]] · [[Bind]] · [[Cleanup memory]] · [[Curiosity]]
