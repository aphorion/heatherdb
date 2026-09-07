# Bind

Combining two vectors into a third that resembles neither, and can be taken
apart again.

## Like you're twelve

Put a letter in an envelope and seal it. The sealed envelope does not look like
the letter, and it does not look like the envelope you started with — it is its
own thing. But hand it to someone holding the right opener and the letter comes
back out.

That is binding. It is how you say *colour is red* in one vector: bind COLOUR to
RED and you get a thing that means the pairing, not either half.

<!--figure:bind-->

## Precisely

Bind is circular convolution, computed through the FFT. The result is
near-[[Orthogonality|orthogonal]] to both operands, which is what distinguishes
it from [[Bundle|bundling]] — a bundle resembles its parts, a binding hides
them.

Binding is what gives a role/filler slot without a schema. `bind(GENRE, scifi)`
is a vector meaning "genre is scifi", and it collides with nothing else in the
space. Bind several such pairs, bundle them together, and one vector holds a
whole record.

The inverse is [[Unbind]], and it is approximate — which is why every serious
use of binding also needs a [[Cleanup memory]].

## Related

[[Unbind]] · [[Bundle]] · [[Permute]] · [[Vector symbolic architecture]]
