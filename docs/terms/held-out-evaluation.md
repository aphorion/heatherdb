# Held-out evaluation

Scoring a memory with the answer barred from participating — the only way a
score means anything once the data is already stored.

## Like you're twelve

Asking a friend to recite a poem while they are holding the book open is not a
memory test. Close the book and it is.

A memory that has absorbed an item will hand that item back nearly perfectly.
That number tells you the write worked, not that the memory generalises.

## Precisely

Every self-query against stored data returns a high score by construction, so
it carries no information about whether anything was learned. An honest
measurement excludes the item being asked about — and, where the store holds
near-duplicates, its whole family, since one near-copy answers as well as the
original.

The engine exposes this as leave-one-out reads: a read with a single stored
engram excluded, or with a family excluded. Most of that family lives in the
Rust library rather than on the HTTP surface, so an HTTP client typically holds
out a split before writing: keep a portion of the data unwritten, score against
that, and report coverage alongside accuracy.

The same discipline applies to every threshold. A gate chosen by judgement is
an untested guess; a gate derived from held-out scores has a measured error
rate at a measured coverage. Thresholds are valid at the load they were
calibrated at, and drift as the pool grows.

## Related

[[Null model]] · [[Fidelity]] · [[Online learning]]
