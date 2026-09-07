# Curiosity

An honest memory answers "I have not seen this" and stops. A curious agent
reads the same sentence as an instruction: *go there*. Curiosity is
[abstention](abstention.md) with the sign of its own signal flipped.

## The signal is already there

A read reports how far the query had to move to reach something stored. That
number — the first-contact similarity between the query and what the read
landed on — is what a gate thresholds to decide whether an answer is worth
returning. Nothing about it is specific to answering:

```
familiarity(s) = max cosine(s, everything already written)
novelty(s)     = 1 − familiarity(s)
```

Thresholded, familiarity gates. Maximised, it exploits. **Minimised, it
explores.** One quantity, three uses, and no second model to train — the
epistemic value of a state is measured by the same read that would have
answered a query about it.

This is what makes exploration cheap here. A conventional agent needs a
learned novelty estimator, a count table, or an ensemble disagreement score
standing beside the policy. A store whose read is a distance measurement is
already the novelty estimator.

## The novelty gradient

The signal alone is a scalar over states. A *gradient* is that scalar
evaluated over the states an action can reach:

1. Enumerate the states reachable in one step.
2. Score each one's familiarity against memory.
3. Move to the least familiar.
4. Write what is observed there, which raises that state's familiarity.

Step 4 is what turns a static field into a gradient that flows. Every visit
flattens the peak the agent just climbed, so the frontier recedes ahead of it
and the walk keeps moving outward instead of orbiting. There is no reward, no
map and no derivative — the memory's own occupancy is the potential.

The behaviour it produces is directed. On a 6 × 6 grid an agent must map by
walking, starting from the centre with an empty memory, coverage after 20
steps is 19.0 cells for a curious walk against 11.4 for a random walk, and
half the world is covered in 19 steps against 44. The cost of that direction
is one cosine per reachable state.

## What the state is

Familiarity is only as good as the encoding it is measured over. Grid
positions encoded by
[fractional power encoding](../terms/fractional-power-encoding.md) —
`bind(x_role, B^x) + bind(y_role, B^y)` — make nearby cells similar and
distant cells near-orthogonal, so an unvisited cell *adjacent* to a visited
one reads as partly familiar. That is a feature: novelty then measures
distance to the explored region rather than mere set membership, and the
agent generalises its ignorance across states it has never occupied.

With a one-hot or hash encoding the same loop degenerates into visit counting,
which works but learns nothing from the neighbourhood.

## The boundaries

Novelty is a difference between the memory and the world, so it vanishes when
either side is degenerate.

**A saturated memory reports everything familiar.** Once the store holds
enough that every query lands near something, familiarity is high everywhere,
the gradient is flat, and the agent has no preference between actions. The
agent that has finished mapping and the agent whose memory has run out of
[capacity](capacity-and-interference.md) are indistinguishable from inside —
both see a uniform field. Distinguishing them requires reading the load, not
the signal.

**An empty memory reports everything novel.** With nothing written, every
state is equally unfamiliar and the argmin is a tie. A curious agent at
step zero is a random agent, necessarily: it has no basis for a preference
yet. Curiosity is a *second-order* drive — it needs some history to point.

**Neither extreme produces behaviour.** Useful curiosity lives in the band
between them, where memory holds enough for the field to have structure and
not so much that it has flattened. That band is the same operating regime a
[gate](../how-to/calibrate-a-gate.md) is calibrated in, for the same reason:
both depend on the score still separating.

A further boundary sits inside the loop. A deterministic argmin over a
finite neighbourhood admits limit cycles — the agent shuttles between two
cells whose familiarity keeps swapping. Coverage plateaus at 26 of 36 cells
on the grid above when ties are broken deterministically, and reaches 34.2
by step 80 when they are broken at random. The gradient is directed; it is
not by itself ergodic.

## Novelty alone is not a stopping rule

"Explore the most novel state" says where to go, never when to stop. A fixed
novelty threshold is a knob with no natural scale, and it is wrong in two
directions at once: it chases rare states that will never be revisited, and
it skips mildly surprising states that are visited constantly.

Description length has a scale — bits. Storing a state as an exception costs
`b` bits once (an address plus a value) and removes that state's residual
surprise from every future encounter, so the exploration frontier is the set
of states where

```
surprise(s) × recurrence(s)  >  b
```

Measured over a synthetic encounter stream with three state types (well
predicted, rare-and-surprising, frequent-and-mildly-off) at `b = 12.9` bits,
the bit-accounted rule reaches a total description length of 484 bits against
819 for the best of six fixed novelty thresholds, and it does so by storing
the 20 frequently visited states and only 1 of the 40 rare ones. Shift the
visitation distribution — rare states five times rarer, frequent ones twice
as hot — and the same rule with nothing retuned reads 416 bits while the
threshold tuned for the previous world reads 819.

The frontier under this rule is finite and shrinks as it is consumed: 21
states carry positive bits-to-save at the start, and when none remain the
agent stops. **Abstention returns as the terminal condition of curiosity** —
not "I don't know", but "there is nothing left worth learning".

## Curiosity and planning

A plan is a rollout through a [world model](world-models.md), and the model
is the same object as the memory: an action's effect learned as a transform
vector, `M_a = Σ unbind(next state, state)`, applied by binding it onto the
current state. Predicting an outcome and deriving a procedure are one
operation.

Planning and curiosity meet on the score. A rollout can be scored by how much
it would resolve — the summed novelty of the states it passes through —
rather than by reward, which makes an experiment a first-class plan. The
same machinery scores both drives, so an agent can hold them in one field:

```
V(x) = need · proximity-to-known-goal(x)  +  curiosity · novelty(x)
```

When the need term is zero the agent explores; when need dominates it exploits
what it has already mapped. The switch is not a mode selector, it is which
term is larger. An agent on a foraging world with a preferred energy level
holds mean energy 12.3 against 10.0 for a random walker and starves 3.5 times
in 800 steps against 35.3, spending 64% of its steps exploring — behaviour
that was never specified, only implied by a preference and a novelty signal.

## Related

- [Abstention](abstention.md) — the same number, used to refuse.
- [World models](world-models.md) — what a rollout runs through.
- [Build a curious agent](../how-to/build-a-curious-agent.md) — the loop, with
  the measured comparison.
- [Capacity and interference](capacity-and-interference.md) — why saturation
  flattens the field.
- [Boundary conditions](boundary-conditions.md) ·
  [Mnemonics](mnemonics.md)
- [fidelity](../terms/fidelity.md) ·
  [fractional power encoding](../terms/fractional-power-encoding.md) ·
  [cosine similarity](../terms/cosine-similarity.md)
