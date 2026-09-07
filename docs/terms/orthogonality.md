# Orthogonality

Two vectors pointing in unrelated directions — at right angles, sharing nothing.

## Like you're twelve

Push a shopping trolley north while your friend pushes it east. Neither of you
helps or fights the other; your efforts are independent. That is orthogonal.
Now push north while your friend pushes north-east — some of their push adds to
yours. That is correlation, and it is what makes stored things bleed into each
other.

<!--figure:orthogonal-->

## Precisely

Two vectors are orthogonal when their dot product is zero — cosine similarity 0,
90° apart. In practice nothing is exactly orthogonal; what matters is that
random high-dimensional vectors are *nearly* orthogonal, with similarities
hovering near zero.

Near-orthogonality is what buys interference-free storage. When you
[[Bundle|bundle]] several vectors, each one's contribution is recoverable
because the others contribute almost nothing along its direction. When you
[[Bind|bind]], the result is near-orthogonal to both inputs, which is exactly
what makes a binding act like a sealed container rather than a blend.

The failure mode is worth naming: if you encode two genuinely different things
with correlated vectors, no amount of clever reading separates them again. The
space is only as honest as the encoding.

## Related

[[High-dimensional space]] · [[Cosine similarity]] · [[Bind]]
