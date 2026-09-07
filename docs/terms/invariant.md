# Invariant

A quantity that does not change while everything else does.

## Like you're twelve

Watch a swing go back and forth. Height changes, speed changes, direction
changes — high and slow at the top, low and fast at the bottom. But there is a
combination of height and speed that stays exactly the same all the way through,
and finding that combination tells you more about the swing than any single
snapshot does.

The thing that refuses to change is usually the thing worth knowing.

## Precisely

Given states observed along a trajectory, an invariant is a direction in which
nothing changes between one state and the next. Lift the state into features,
form the matrix of feature *differences* across every observed transition, and
an invariant is a direction in that matrix's null space — a weighting of
features whose value is the same before and after.

No equations are supplied and no model class is chosen. The data is the
trajectories, and the true quantity, where one is known, is used only to grade
the answer.

The property that makes this trustworthy is what it does when there is nothing
to find. A damped system conserves nothing, and the same computation run on it
must return a weak result rather than a plausible one — a method that reports a
law for every input has not found one. Grading therefore means three numbers:
the result, its score on a system with a known law, and its score on a system
with none.

Where nothing is conserved, the useful reading is a rate rather than a
constant — how fast the system decays, and at what frequency.

## Related

[[Null model]] · [[Held-out evaluation]] · [[World model]] · [[Fidelity]]
