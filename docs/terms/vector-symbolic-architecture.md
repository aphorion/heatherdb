# Vector symbolic architecture

Doing symbolic computation with high-dimensional vectors and arithmetic, instead
of symbols and pointers.

## Like you're twelve

Normally, to say "the film's director is Nolan" a computer stores a little box
labelled *director* with *Nolan* inside, plus a pointer to the film. VSA says:
multiply *director* by *Nolan*, add the result to the film's vector, and the
sentence is now a single number-list you can do arithmetic on. Want the
director back? Divide by *director*.

Structure, without a single box or pointer.

<!--figure:bind-->

## Precisely

A VSA represents everything — atoms, roles, records, sequences, whole graphs —
as fixed-width vectors in one [[High-dimensional space|high-dimensional space]],
and builds structure with three operations: [[Bind]] for association,
[[Bundle]] for superposition, [[Permute]] for order. Composition never grows the
representation, which is why the family is also called Holographic Reduced
Representations.

The catch is that every operation is approximate, so a VSA is only usable with a
[[Cleanup memory]] behind it — the part conventional implementations bolt on and
Heather already is.

The full argument is in [[Vector symbolic architectures]]; the operations and
their surface are in [[Vector algebra]].

## Related

[[Bind]] · [[Bundle]] · [[Permute]] · [[Cleanup memory]]
