# Hopfield read

HeatherDB's read operation: a query is re-weighted against the memory over and
over until it stops moving, and where it stops is the answer.

## Like you're twelve

Squint at a blurry photo of someone you know. Your first guess is vague, but
the moment you commit to it your brain fills in the face it thinks it is
seeing, which makes the guess sharper, which fills in more. A few rounds of
that and you are certain — and you did not flip through a photo album to get
there.

A read works the same way. Your query is the first vague guess. The memory
answers with what that guess most resembles, that answer becomes the next
guess, and after a few passes it settles somewhere and stops changing.

<!--figure:attractor-->

## Precisely

A read starts by activating the *k* stored locations most similar to the
query. From then on the activation set is fixed and only the weights over it
move:

```
ξ₀ = normalize(query)
repeat up to t_max:
    sims  = cosine(ξ, addressⱼ)
    α     = softmax(β · sims)
    ξ_new = normalize(Σ αⱼ · patternⱼ)
    stop if cosine(ξ, ξ_new) > 1 − epsilon
    ξ    ← ξ_new
```

Each pass re-weights toward whatever the *current* estimate resembles rather
than what the original query resembled, which is exactly what carves basins:
a noisy query is pulled toward the nearest genuine [[Attractor|attractor]]
instead of staying where it landed. The loop stops when a pass barely moves
the estimate, or at `t_max`.

Two strategies are exposed. `iterative` (the default) runs the loop to
convergence; `fast` does a single step, which is cheaper and adequate when the
query is already close. `analyze` runs the identical computation but reports
its working — iterations, whether it converged, and which locations carried
what weight.

The softmax in the middle is not an analogy for attention. It is
`softmax(β · Q·Kᵢ)`, the same operation a transformer attention head performs,
which is why a memory read and an attention step are the same thing seen from
two directions. Compare the query with what came back and you have
[[Fidelity|fidelity]].

## Related

[[Attractor]] · [[Hopfield network]] · [[Associative memory]] · [[Fidelity]] ·
[[Cleanup memory]]
