# Permute

Shuffling a vector's components in a fixed, repeatable way — the trick that
encodes order.

## Like you're twelve

Take a row of coloured cards and shift every card one place to the right. Do it
again and everything moves another place. The cards are the same cards; only
their arrangement changed, and you can always undo it by shifting back.

Now use "shifted once" to mean *first*, "shifted twice" to mean *second*, and
you can store a sequence without needing a separate box per position.

<!--figure:permute-->

## Precisely

Permutation is a fixed reordering of a vector's components, usually a cyclic
shift. It produces a vector near-[[Orthogonality|orthogonal]] to the original —
so "A in position 1" and "A in position 2" do not resemble each other — and it
is exactly invertible, unlike [[Unbind|unbinding]].

That combination makes it the standard way to encode order and nesting in
vector symbolic systems: bundle ρ¹(a) + ρ²(b) + ρ³(c) and you have a sequence
in one vector, readable by permuting back.

## Related

[[Bind]] · [[Bundle]] · [[Vector symbolic architecture]]
