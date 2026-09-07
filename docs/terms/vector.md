# Vector

A list of numbers that stands for a thing — and, read as coordinates, a
direction in space.

## Like you're twelve

Describe a song with five numbers: how fast, how loud, how sad, how much
guitar, how danceable. Two songs with nearly the same five numbers are nearly
the same song, and you can tell that without listening to either. You did not
compare the songs; you compared their lists.

Now use two thousand numbers instead of five and the same trick describes
anything — a face, a sentence, a customer, an hour of sensor readings. That list
is a vector.

<!--figure:vector-->

## Precisely

A vector is an ordered array of numbers of fixed width. That width is the
[[High-dimensional space|dimensionality]] of the space, and in HeatherDB it is fixed per database: every
vector in one database has the same number of components, because comparing
lists of different lengths is meaningless.

The numbers themselves come from wherever you like — an embedding model, a
sensor, a hand-built encoding, or random assignment for symbols that only need
to be distinguishable. Nothing about the store cares. What matters is that
similar things get similar lists, because closeness is the only thing the memory
can see.

## Related

[[High-dimensional space]] · [[Cosine similarity]] · [[Bundle]]
