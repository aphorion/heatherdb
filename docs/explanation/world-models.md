# World models

A world model answers one question: `(state, action) → next state`. That is a
[binding](../terms/bind.md) held in an associative memory, so a collection that
stores transitions **is** a world model. No module is added to the engine to get
one.

## The transition function is a bundle of bindings

Fix a state `s`. Every action available from it produces a successor. Bind each
action role to the successor it produces and
[bundle](../terms/bundle.md) the results:

```
M_s = Σ_a bind(a, s'_a)
```

`M_s` is the transition function *of that state*, held in one vector. Reading it
is prediction:

```
ŝ' = cleanup( unbind(M_s, a) )
```

[Unbinding](../terms/unbind.md) recovers the bound successor plus the
[interference](../terms/interference.md) of the other actions in the bundle; the
cleanup read snaps that residual onto the stored state it belongs to.

This is the shape a [hard location](../terms/hard-location.md) already has. Its
address is a routing vector; its counter is a superposed accumulator. Store the
state code in the address and the transition bundle in the counter and the
memory's own read is the predictor. Learning is one write per experienced
transition — no training loop, no separate fitting step.

## A rollout is predict, clean up, repeat

Imagination is the same read run in a loop. Each step unbinds the next action
out of the current state's map, cleans the result up to a stored state, and uses
that state to address the next map.

The cleanup is load-bearing, not cosmetic. Both arms below address identical
per-state maps over 81 states at `D = 1024`; they differ only in what the
rollout carries between steps — a state snapped onto its
[attractor](../terms/attractor.md), or the raw unbind residual fed forward
through a soft read at the stock `beta = 5`:

| Horizon | Snap between steps | Raw residual, β = 5 | Raw residual, β = 15 |
|---|---|---|---|
| 1 | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% |
| 3 | 100% | 71% | 100% |
| 4 | 100% | 17% | 100% |
| 8 | 100% | 3% | 100% |
| 32 | 100% | 2% | 100% |

*(measured here; 400 rollouts per cell)*

Without a step that discretises, a rollout walls at depth two to four — the same
wall composition hits in [the cleanup loop](the-cleanup-loop.md), now in the
time axis. The snap resets the per-step noise floor, and depth stops being a
compounding quantity. A sharp enough read is a cleanup: `beta = 15` recovers the
whole horizon.

## The read returns a distribution, not a guess

Real dynamics branch. A state-action pair experienced with two outcomes
superposes both into the same counter, weighted by how often each occurred, and
the [Hopfield read](../terms/hopfield-read.md) hands the weights back:

- `strategy: "iterative"` collapses to the mode — commit to one successor, which
  is what a rollout step wants.
- `strategy: "fast"`, or `analyze`, keeps the spread — the predictive
  distribution itself, as activation weights over candidate successors.

Both come out of the read rule. Neither is a trained output head.

## Three boundaries

### Prediction degrades with horizon, in proportion to coverage

A map written by wandering holds only the transitions it has felt. Rollout
accuracy on an 81-state grid, against the trivial baseline "the state does not
change":

| Horizon | 135/324 written | 247/324 written | 324/324 written | Stay-put |
|---|---|---|---|---|
| 1 | 46% | 76% | 100% | 14% |
| 2 | 31% | 66% | 100% | 27% |
| 4 | 11% | 49% | 99% | 14% |
| 8 | 5% | 34% | 99% | 8% |
| 16 | 1% | 24% | 96% | 4% |
| 32 | 1% | 16% | 94% | 5% |

*(measured here)*

Decay is multiplicative: a rollout is correct only if every step was, so a
per-step error rate `e` gives roughly `(1−e)^H`. Full coverage still drifts —
94% at H = 32 — because visit counts are uneven and the bundle is weighted by
them. The horizon at which a model stops being useful is a measurement, not a
constant, and the [stay-put baseline](../how-to/find-your-metrics-floor.md) is
what makes the number mean anything: at H = 2 the baseline reads 27%, so a
31% model has found almost nothing.

### The encoder decides whether dynamics generalize

Per-state maps memorise. Give each state an independent random code and the
model knows the states it has visited and nothing about the rest. Give the
states a structured latent — a [fractional-power
encoding](../terms/fractional-power-encoding.md) of the coordinates, so nearby
states share structure — and "go East" becomes a single operator
`s' = bind(T_E, s)` that holds everywhere. The transition function collapses
from one map per state to one operator per action.

Operator accuracy on states never visited, 9×9 grid, chance ≈ 1.2%:

| Training states | Random codes | Structured latent |
|---|---|---|
| 2 / 81 | 1% | 100% |
| 8 / 81 | 1% | 100% |
| 40 / 81 | 2% | 100% |

*(measured here, from `exp_latent_states.py`)*

Two states are enough to fix all four operators, and they predict the other 79
exactly. Random codes admit no such operator, because there is nothing for two
states to share. Structure in the encoding is where generalization lives; the
memory is the easy half.

### Composed actions need a non-commutative operator

A per-action operator applied by binding is a circular convolution, and
convolutions commute. The learned operators therefore commute whatever the world
does — measured composition-order cosine **1.00** on both an abelian and a
non-abelian world. A world factors into such operators exactly when its own
actions commute:

| World | Order-sensitive pairs | Held-out step | 2-step composition |
|---|---|---|---|
| Cyclic (abelian) | 0% | 100% | 100% |
| Dihedral (non-abelian) | 50% | 26% | 55% |

*(measured here, from `exp_nongrid.py`)*

Order does not survive a commuting encoding. Rollout is unaffected — a per-state
map stores a graph, and both worlds roll out at 100% to depth 16 — so the
boundary is on *factoring*, not on prediction.

Escaping it means changing the operator family, not the world. A learned
orthogonal map `exp(S − Sᵀ)` need not commute, and at `N = 64`, `D = 32` it
moves the dihedral world's order-sensitive composition from 42% to **73%** and
held-out prediction from 28% to **49%**, with operator commutation falling from
1.00 to **0.53** (measured here, from `exp_nonabelian_fix.py`). The direction is
right and the abelian ceiling is not reached at that width: non-commuting
dynamics are outside the regime the shipped binding covers.

## What a state can be

Nothing in the mechanism requires a state to be a position, and nothing
requires an action to be a movement. A state is whatever you encode; an action
is whatever transforms it. The grid examples in the how-to are convenient, not
representative.

| Domain | A state is | An action is | A rollout answers |
|---|---|---|---|
| Navigation | where the agent is | a move | where a path ends |
| Interface | what is on screen | a click | what a sequence produces |
| Process control | sensor readings | a setpoint change | where the plant settles |
| Biology | a measured profile of a sample | a perturbation applied to it | what a course of treatment leaves behind |
| Markets | a regime, as measured | a shock | how the regime propagates |

The biological row is the one that most changes what the machinery is for, and
it is worth being precise about. Given before-and-after measurements of samples
under known perturbations, the transition function is exactly the object above:
each perturbation is a role, each resulting profile is its filler, and one
vector per starting state holds all of them. Predicting the effect of a
perturbation on a state nobody has tested is an unbind followed by a cleanup —
the same two calls that predict a move on a grid.

A transformation between two states is the degenerate case: a world model with
a single action. Encoding "diseased profile" and "healthy profile" and taking
the transform that carries one to the other gives a vector that can be applied
to a *new* diseased profile, and the result is a prediction about a sample the
transform was never derived from. That is a rollout of length one.

## Why fidelity matters more here than on a grid

On a grid, a wrong prediction is visible immediately — the agent walks into a
wall. In a domain where the ground truth costs a laboratory a month, a
confident wrong answer is expensive and silent, so the important property is
not the prediction but the number attached to it.

Two guards apply, and neither is optional outside a toy problem:

**Every predicted step carries its own [fidelity](../terms/fidelity.md).** A
prediction about a state well inside the region the memory has seen scores
high; one about a state outside it scores low, and that separation is
measurable in advance rather than discovered afterwards. A model built on part
of a space and asked about the rest shows a clean split — high scores on the
covered region, low on the uncovered — which is the signal to stop and
measure rather than to act.

**A fitted transition function is valid only over the region the observations
covered.** This is not a caution about accuracy degrading gracefully at the
edges; outside the covered region the model will return a plausible,
well-formed state that means nothing. The [rollout](../terms/rollout.md) does
not know it has left the map. Its fidelity does.

Neither guard makes a prediction true. They make an untrustworthy prediction
identifiable, which is the difference between a model that generates hypotheses
worth testing and one that generates hypotheses indistinguishable from noise.

## Latent versus observed states

Nothing above requires states to be observable things. A state is whichever
vector the encoder produces, and rollout fidelity is indifferent to which: with
maps enumerated over all states, random latents and structured latents both roll
out at 100% to horizon 20. What the encoding changes is what can be factored,
and therefore what transfers to states never visited.

The honest limit is the arrow's other end. The model maps state to state. An
observation that does not determine the state — partial observability, aliasing,
history-dependent dynamics — is outside what a per-state map represents, and no
measurement here characterises that regime. The available move is to make the
encoded state carry whatever history disambiguates it, which pushes the problem
back into the encoder where the second boundary already lives.

## Related

- [Build a world model](../how-to/build-a-world-model.md) — the task-shaped
  version of this page.
- [The cleanup loop](the-cleanup-loop.md) — the same wall in composition rather
  than in time.
- [How the memory works](associative-memory.md) — the read that does the
  cleanup, and the write that does the learning.
- [Capacity and interference](capacity-and-interference.md) — how many
  transitions one map holds.
- [Vector algebra](vector-algebra.md) · [Consolidation](consolidation.md)
