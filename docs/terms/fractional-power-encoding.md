# Fractional power encoding

Turning a continuous quantity into a vector by raising a base vector to a
fractional power, so nearby values give nearby vectors.

## Like you're twelve

A clock hand at 3:00 and a clock hand at 3:01 point almost the same way; at
9:00 it points somewhere else entirely. You did not need a separate symbol for
every minute — one hand, rotated by however much time has passed, encodes them
all, and closeness in time is closeness in angle.

<!--figure:permute-->

## Precisely

Give each dimension a random phase and advance all of them in proportion to the
value. A quantity *x* becomes `base^x`, computed in the frequency domain, which
makes the encoding continuous: the similarity between two encoded values falls
off smoothly with their numeric distance instead of dropping to zero at a
bucket boundary.

Two properties follow. Addition of quantities becomes a single [[Bind|bind]],
because multiplying in the frequency domain adds phases. And the scheme extends
to position and degree — a fractional [[Permute|permutation]] rotates by a
non-integer amount, which encodes continuous position the way integer
permutation encodes discrete order.

Choose it over one-hot buckets when you want a gradient rather than a step
function: buckets give hard boundaries where two values either share a bucket
or share nothing, while this gives a graded answer everywhere. Choose buckets
when the boundaries are real — a price band, a legal age — and the gradient
would be a lie.

Exposed as `/vec/pow` and `/vec/rotate`.

## Related

[[Bind]] · [[Permute]] · [[Vector]] · [[Cosine similarity]]
