# Discover a conservation law

Hand the method raw trajectories — pairs of states, nothing else — and it
returns the combination of features that does not change along the flow. No
equations, no model class, no candidate law. The true quantity, where one is
known, enters only to grade the answer.

The object recovered is an [invariant](../terms/invariant.md): a function
`H(s)` with `H(s') = H(s)` on every observed transition. Lift the state into
features `φ(s)` and that condition is linear — `w · (φ(s') − φ(s)) = 0` for
every transition — so `H = w · φ` for any `w` in the null space of the matrix
of feature changes. The smallest right-singular directions of that matrix are
the candidates.

## The recipe

1. **Lift.** Map each state into features — random Fourier features for a
   black-box read, an interpretable polynomial/log/trig basis to read the law
   back as symbols.
2. **Difference.** Stack `φ(s_{t+1}) − φ(s_t)` over every observed transition,
   whitened by each feature's spread so the null direction is not merely the
   quietest feature.
3. **Decompose.** Take the bottom right-singular vectors of that matrix.
4. **Select** among the bottom few by **held-out drift** — how constant a
   candidate stays on unseen trajectories. Selection never touches the truth.
5. **Grade twice**: correlation against the true quantity where one exists, and
   drift at an energy or initial condition never trained on.
6. **Run a dissipative control through the identical pipeline.**

## Lift, difference, decompose

```python
import numpy as np

D_FEAT = 400
rng = np.random.default_rng(0)
# Random Fourier features: cos(W·s + b) has the property that the inner
# product of two feature vectors approximates a Gaussian kernel, so a linear
# function of phi is a smooth nonlinear function of the state. Nothing about
# the system is encoded here -- W and b are drawn before any data is seen.
W = rng.standard_normal((D_FEAT, 2)) * 2.5
b = rng.uniform(0, 2 * np.pi, D_FEAT)
SCALE = np.array([1 / np.pi, 1 / 2.0])      # put (q, p) on roughly unit scale

def phi(S):
    S = np.atleast_2d(S) * SCALE
    return np.sqrt(2.0 / D_FEAT) * np.cos(S @ W.T + b)

def discover(trajs):
    """Conserved quantities = the least-changing directions of the flow."""
    dphi = np.vstack([phi(t[1:]) - phi(t[:-1]) for t in trajs])
    Phi = np.vstack([phi(t) for t in trajs])
    # Centre both: without dropping the DC component, the constant function is
    # trivially conserved and wins every time.
    Phi = Phi - Phi.mean(0)
    dphi = dphi - dphi.mean(0)
    sd = Phi.std(0) + 1e-9                      # whiten by feature spread
    _, sv, Vt = np.linalg.svd(dphi / sd, full_matrices=False)
    return Vt, sv, sd                           # Vt[-k] are the candidates
```

## Selection and the two grades

```python
def H_of(w, S, sd):
    return (phi(S) / sd) @ w

def drift(w, trajs, sd):
    """Spread of H ALONG a trajectory over its spread ACROSS the data set. A
    real invariant is flat on each orbit and differs between orbits: near
    zero. A direction that merely tracks position is not."""
    within = np.mean([np.std(H_of(w, t, sd)) for t in trajs])
    return within / (np.std(H_of(w, np.vstack(trajs), sd)) + 1e-9)

# Pick among the bottom singular directions by drift on a validation split of
# the FIT trajectories. The true energy is not consulted at any point here.
best = min((Vt[-k] for k in range(1, 8)), key=lambda w: drift(w, fit[::2], sd))
```

Both grades are then measured at energies never fitted — [held-out
evaluation](../terms/held-out-evaluation.md) in the coordinate that matters for
a law, the level set rather than the sample.

## Measured: held-out energies, and the control

Fit on release amplitudes `0.4, 0.7, 1.0, 1.3, 1.9, 2.2, 2.4`; grade on
`0.55, 1.15, 1.6, 2.05`. The damped oscillator differs from the harmonic one
by a single friction term and goes through byte-identical code.

| System | \|corr(H, true E)\| | drift | null corr (20 draws) |
|---|---|---|---|
| harmonic oscillator | **0.978** | 2.9 × 10⁻¹¹ | 0.200 ± 0.135 (max 0.454) |
| pendulum (nonlinear) | **1.000** | 2.7 × 10⁻¹¹ | 0.084 ± 0.057 (max 0.184) |
| damped oscillator (control) | **0.465** | 5.3 × 10⁻¹ | 0.247 ± 0.106 (max 0.390) |

The control declines on both grades at once: correlation 0.465 sits inside the
null's range, and drift 0.53 is indistinguishable from the null's 0.64 ± 0.06,
against conservative readings ten orders of magnitude lower. A dissipative
system has no conserved energy, and the measurement says so rather than
returning a plausible-looking law.

Nonlinearity is not the difficulty. The pendulum's invariant,
`½p² + (1 − cos q)`, is recovered at correlation 1.000 — a cleaner read than
the linear oscillator's 0.978, whose circular level sets have more nearly
degenerate feature images.

**The singular value certifies nothing.** The bottom singular values are
`9.4 × 10⁻¹⁵` (harmonic), `9.2 × 10⁻¹⁵` (pendulum) and `1.2 × 10⁻¹⁴` (damped)
against top values near `3 × 10¹`: a 400-feature difference matrix has a
numerically empty null space whatever the physics is. Only the held-out grades
separate the three.

## The null that works, and the one that does not

The claim is that transitions stay inside a level set. The null must destroy
exactly that:

```python
# RIGHT: every successor is drawn from the pooled states of all trajectories.
# Level-set structure is gone; the state distribution is untouched.
pooled = np.vstack(fit)
null_trajs = [np.stack([t[0]] + list(pooled[rng.integers(0, len(pooled),
                                                         len(t))]))
              for t in fit]

# WRONG: shuffling states WITHIN a trajectory. Every state stays on its own
# level set, so the structure being claimed survives the shuffle intact.
null_trajs = [t[rng.permutation(len(t))] for t in fit]
```

Through the identical pipeline the within-trajectory shuffle returns **0.964**
(harmonic) and **1.000** (pendulum) — a null as strong as the result, because
it preserves the thing it was meant to remove; the cross-trajectory null floors
at 0.200 and 0.084. [Find your metric's floor](find-your-metrics-floor.md) and
[null model](../terms/null-model.md) give the general form.

## Measured: held-out initial conditions, six systems

The same pipeline over a wider zoo — six trajectories fitted per system, two
initial conditions held out, the true invariant regressed onto the best four
candidates:

| System | recovered | held-out drift |
|---|---|---|
| harmonic oscillator | E: **1.000** | 5.8 × 10⁻⁴ |
| pendulum (nonlinear) | E: **1.000** | 5.9 × 10⁻⁴ |
| Lotka–Volterra | V: **1.000** | 7.1 × 10⁻³ |
| Kepler orbit (2-body) | E: **1.000**, L: **1.000** | 6.1 × 10⁻³ |
| damped oscillator | E: **0.730** | 1.0 × 10⁻¹ |
| Lorenz attractor | E: **0.350** | 9.2 × 10⁻¹ |

Kepler carries two independent invariants and both come back — the method is
not limited to one direction per system. The two systems with nothing to
conserve sit two to three orders of magnitude higher in drift.

## Reading the law back as symbols

Swapping the random features for a named basis — `{v, v², vᵢvⱼ, ln|v|, cos v,
sin v}` per coordinate — costs generality and buys legibility. Same selection
rule, 30 trajectories per system, 70/30 split by trajectory:

| Field | System | recovered terms | corr | held-out drift |
|---|---|---|---|---|
| physics | pendulum | `+1.00·cos q − 0.84·p²` | 1.000 | 0.000 |
| relativity | free particle | `+1.00·E² − 0.84·p²` | 1.000 | 0.000 |
| astronomy | orbit | `−1.00·x·vy + 1.00·y·vx` | 1.000 | 0.000 |
| thermodynamics | ideal gas | `−1.00·ln P − 0.63·ln V` | 0.958 | 0.000 |
| ecology | Lotka–Volterra | `−1.00·ln y + 0.89·y + 0.83·x` | 1.000 | 0.000 |
| epidemiology | SIR | `−1.00·S² − 0.44·S − 0.28·S·I` | 0.998 | 0.000 |
| — | damped (control) | `−1.00·q² − 0.76·cos q + 0.74·q` | **0.925** | **0.650** |

`E² − p²` and angular momentum `x·vy − y·vx` come back as their textbook forms,
term for term. The gas law is recovered in log coordinates, where `PV/T` is
additive, at 0.958 — the lowest of the six and the only one whose top terms do
not spell the closed form exactly.

**The control is where the two grades earn their keep.** In this basis the
damped oscillator scores 0.925 against the true energy — publishable, on that
number alone — because collapsing spirals make `q²` track `½(q² + p²)` across
the pooled states. It is not conserved *along* any orbit, and drift says so:
0.650, against 0.000 for every system with a law. In the random-feature basis
the decline appears in the other grade instead, 0.465 at drift 0.53. Which
grade catches a false law depends on the basis; requiring both makes the
decline reliable.

## Boundary conditions

- **Constancy on the fitted trajectories is not a law.** A candidate selected
  and scored on the same orbits is a curve fit; every grade above is measured
  on energies or initial conditions the fit never saw.
- **Sampling must cover more than one level set.** At a single energy the level
  index is itself constant and any function of it scores perfectly. The fit
  sets above span 4–8 levels.
- **Chaotic flows have no invariant to recover.** Lorenz reads 0.350 at drift
  0.92 — correctly nothing; its law is a rate.
- **The state must be observed, not inferred.** A hidden coordinate makes the
  flow appear non-deterministic and drives drift up on every candidate.

## Related

- [Model a dynamical system](model-a-dynamical-system.md) — the general case,
  where the eigenvalue is not 1 and the law is a decay rate.
- [Find your metric's floor](find-your-metrics-floor.md) · [Forecast a time
  series](forecast-a-time-series.md)
- [Invariant](../terms/invariant.md) · [Null model](../terms/null-model.md) ·
  [Held-out evaluation](../terms/held-out-evaluation.md) ·
  [World model](../terms/world-model.md)
