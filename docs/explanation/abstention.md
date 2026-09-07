# Abstention

A read that reports its own confidence gives refusal for free: the same number
that answers the query says whether the query should have been answered.

Conventional systems bolt refusal on — a separate classifier, a calibration
model, an out-of-distribution detector consuming the same rows a second time.
A reconstruction read already carries the quantity, because reconstructing
requires measuring how far the query was from what is stored.

## Two kinds

**Structural abstention.** A key that was never composed has no address in the
store. Unbinding it returns noise that resembles nothing, so a miss cannot
return a confident wrong answer — the geometry has no item there to return. No
threshold exists to tune, and no calibration set is required, because the
refusal is a consequence of the encoding rather than a decision about a score.

**Calibrated abstention.** Where the question is "is this stored pattern close
enough", the answer is a threshold, and a threshold is a fitted quantity. It is
derived from held-out data: measure the score across inputs known to be in
distribution, measure it across inputs known not to be, and place the cut in the
gap between them.

The two differ in what they cost to maintain. Structural abstention holds for as
long as the encoding holds. Calibrated abstention holds for as long as the
conditions it was calibrated under hold — and one of those conditions is load.

## Thresholds are valid only at their calibration load

A score's separation between in- and out-of-distribution inputs is a function of
how much the store holds, because interference grows with occupancy. A gate
calibrated on a lightly loaded collection reports a different distribution on a
heavily loaded one.

An entropy gate over the attention weight distribution:

| Collection size | ROC AUC |
|---|---|
| 3,285 items | 0.76 – 0.80 |
| 10,940 items | 0.32 – 0.39 |

Below 0.5 the gate is anti-correlated: at the higher load it is worse than a
coin. The gate is not broken — it is being read outside the regime it was
measured in. **A threshold carries its calibration load as part of its
definition**, and a collection that has grown past that load needs the threshold
re-derived, exactly as a capacity budget does.

## Where the signal lives

Not every similarity a read exposes carries information about confidence.

**First-contact similarity carries the signal.** The cosine between the query
and the single-step read result measures how far the query had to move to reach
the codebook. On a task separating queries that the store can answer from
queries it cannot:

| | Mean first-contact similarity |
|---|---|
| Applies | 0.818 |
| Does not apply | 0.304 |

ROC AUC 1.0000, with an empty band at [0.61, 0.80] — no observation of either
class falls inside it. A gap that wide is what makes a threshold cheap to place
and stable to hold: any cut in the band separates perfectly.

**Post-iteration similarity carries none.** An iterative read runs until the
estimate stops moving, so its final similarity is ≈ 1.0 by construction, for
every input, including inputs the store has nothing to say about. The iteration
did not fail to detect anything; it converged, which is what it is for. The
number it leaves behind is a statement about convergence, not about the query.

<!--figure:memory-->

## The counter-property

A cleanup that returns a nearest neighbour for any input has no distance signal.
It always answers, so its answer is uninformative about whether it should have
answered — the answer's existence carries zero bits.

Stated as a rule: **a search that cannot fail has not succeeded.** The ability
to return nothing is what makes returning something a measurement. This is why
the read's output is a pair — reconstruction *and*
[fidelity](../terms/fidelity.md) — rather than a bare vector, and why exposing
only the vector to an application throws away the half that governs whether the
other half should be used.

## Building a gate

1. Use the **first-contact** read, not the iterative one, wherever the result
   feeds a decision.
2. Prefer **structural** abstention where the encoding allows it — an address
   that does not exist needs no threshold.
3. Where a threshold is needed, derive it on **held-out** data at the
   collection's **operating load**, from both classes.
4. Record the load the threshold was calibrated at, and re-derive when the
   collection moves materially past it.
5. Report the band, not just the cut. An empty band says the gate is safe; a
   band that has closed says the gate has stopped separating.

## Related

- [How the memory works](associative-memory.md) — fidelity as a free by-product
  of the read.
- [The cleanup loop](the-cleanup-loop.md) — why iterative reads discard the
  signal.
- [Capacity and interference](capacity-and-interference.md) — the load that
  moves the threshold.
- [Boundary conditions](boundary-conditions.md) — the case where saturation
  looks like confidence.
- [Fidelity](../terms/fidelity.md),
  [cleanup memory](../terms/cleanup-memory.md).
