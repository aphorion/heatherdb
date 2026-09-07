# Pentti Kanerva

Almost everything in this system descends from the work of one researcher, and
the project is named after him.

## The name

*Kanerva* is Finnish for heather — the low, hardy shrub that covers open
moorland. Heather is the English translation of Pentti Kanerva's surname, and
the project carries it as an acknowledgement rather than a coincidence. HeatherDB
is an implementation of ideas he set out decades before anyone had a use for
them.

## What he worked out

Kanerva's *Sparse Distributed Memory* (1988) asked what memory would look like
if it were addressed by content in a space too large to enumerate — an address
space of, say, 2¹⁰⁰⁰ possible addresses with a few million real locations
scattered through it. Writing activates every location within a radius of the
address and adds the pattern to all of them. Reading activates the same
neighbourhood and sums what it finds.

Two properties fall out of that arrangement, and both are load-bearing here.
Storage is distributed, so no location holds an item and losing some degrades
the memory gracefully instead of destroying records. And retrieval is by
similarity, so an approximate address returns the pattern anyway — which makes
SDM the first explicit [[Cleanup memory]], years before the algebra that needed
one had a name.

SDM arrived six years after the [[Hopfield network]], and answers the same
question differently. Hopfield stores patterns in dense recurrent weights and
runs out of room at roughly 0.14 patterns per unit; Kanerva spends address space
instead, which is why capacity stops being the thing that limits the idea.

He later gave that algebra its name too. *Hyperdimensional computing* (2009) set
out the case that computation in very high-dimensional spaces is a paradigm in
its own right, resting on a single robust fact: random vectors in such a space
are almost always near-[[Orthogonality|orthogonal]]. Distinct things stay
distinct without anyone assigning identifiers, and [[Superposition|superposed]]
vectors stay readable. His binary spatter codes (1994) are one of the concrete
[[Vector symbolic architecture|vector symbolic architectures]] built on that
fact.

<!--figure:orthogonal-->

## What Heather takes, and where it leaves

Four commitments come straight across, largely unaltered. Memory is addressed by
content rather than location. Storage is distributed across many places rather
than filed in one. Retrieval tolerates approximate input, because the read is a
settle rather than a fetch. And dimensionality — not indexing cleverness — is
the resource that keeps unrelated things apart.

The departure is what the word *elastic* is doing in
[[Elastic Associative Memory (EAM)]]. SDM fixes its scaffold in advance: the
locations are drawn at random up front, and the memory's structure does not
depend on what gets written into it. That is a reasonable choice for a model of
human memory and an awkward one for a database, where the data arrives before
anyone knows its shape and keeps arriving afterwards. In Heather the structure
adapts to the data as it lands — each write places a pattern, adjusts the
neighbourhood it falls into, and updates the paths used to reach it — which is
what removes the separate index, the rebuild, and the retraining.

The rest of the distance is engineering rather than theory: durable storage,
per-database dimension, concurrent access, backups, an audit log. None of it
changes the idea. It makes the idea something you can run.

## Read the originals

- Pentti Kanerva, *Sparse Distributed Memory* (MIT Press, 1988) — the book.
- Pentti Kanerva, "Hyperdimensional Computing: An Introduction to Computing in
  Distributed Representation with High-Dimensional Random Vectors", *Cognitive
  Computation* 1(2), 2009 — the shortest route into the paradigm, and still the
  best one.
- Pentti Kanerva, "Binary Spatter-Coding of Ordered K-tuples" (1994) — binding
  and bundling in binary form.

## Related

- [How the memory works](associative-memory.md) — the read path that does the
  cleanup.
- [[Sparse Distributed Memory (SDM)]] — the glossary entry, explained twice.
- [Vector symbolic architectures](vector-symbolic-architectures.md) — the algebra
  his work makes possible.
