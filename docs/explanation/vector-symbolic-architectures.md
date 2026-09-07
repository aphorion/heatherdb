# Vector symbolic architectures

A vector symbolic architecture is a way of doing symbolic computation without
symbols — no parser, no graph, no neural network. Structure is represented as
arithmetic over high-dimensional vectors, and the structure can be taken apart
again by the inverse arithmetic. It is the tradition HeatherDB's algebra comes
from, and this page is the reasoning behind it. For the operations themselves,
see [vector algebra](vector-algebra.md).

## Three commitments

**Meaning is distributed.** A concept is not a field or a slot; it is a pattern
spread across every component of a vector. No single dimension carries the idea,
so no single dimension can destroy it. Degradation is graceful: lose a tenth of
the components and you have a slightly noisier concept, not a corrupt record.

**Dimensionality is the resource.** In a space of a few thousand dimensions, two
randomly drawn vectors are almost exactly orthogonal, and stay that way. That
near-orthogonality is what makes the rest possible: unrelated things do not
collide by accident, and a sum of several vectors keeps every term recoverable
instead of blurring into an average. Where a symbolic system spends memory,
this one spends dimensions.

**Composition is arithmetic.** Building structure is addition and multiplication
rather than pointer-chasing. Bind a role to a filler and you get a vector
meaning "this role has this value" that resembles neither operand. Bundle
several of those and you get one vector meaning all of them at once — a record,
a set, a sentence — the same width as its parts. Permute to encode order.
Nothing grows as you compose, which is why the tradition calls the results
*reduced* representations.

## Why it stayed a footnote

Every operation is approximate. Unbind a role from a bundle and what comes back
is not the filler; it is the filler plus the interference of everything else in
the bundle. Compose a few levels deep and the noise accumulates. The answer is
always a vector *near* the thing you wanted.

That is tolerable only if you can finish the job: take the approximate result
and snap it back onto the clean item it was reaching for. That step is called
cleanup, and it is where conventional implementations get expensive. The usual
answer is an item memory — a list of every clean vector the system knows —
scanned by similarity on every operation. It works at demo scale. It is a
liability at data scale, and it has to be maintained separately from wherever
the real data lives.

<!--figure:cleanup-->

## Where Heather changes the picture

An associative memory's read path is cleanup. Give HeatherDB a noisy vector and
it settles onto the pattern that vector belongs to and reports the fidelity of
the result — which is precisely the operation a VSA needs after every unbind,
performed against the stored data rather than a bolted-on list.

So the two halves fit together the other way round from usual. The algebra is
not a layer over the database; the database is the missing half of the algebra.
Composition happens in the [algebra operations](vector-algebra.md), cleanup
happens in the memory, and neither needs a model.

The practical effect is that symbolic operations are queries. You can ask for
the value of a role inside a stored record, subtract a concept from a bundle and
read what is left, or score a query against several roles at once and let the
memory clean up each answer. These are ordinary operations against ordinary data,
which is what "storage native" means when it stops being a slogan.

## Related

- [Vector algebra](vector-algebra.md) — bind, unbind, bundle, permute, and the
  routes that expose them.
- [How the memory works](associative-memory.md) — the read path that does the
  cleanup.
- [Store and query structured documents](../how-to/structured-documents.md) —
  role/filler bundles used in anger.
