# Model a dynamical system

Most systems conserve nothing. Their law is still readable from raw
trajectories: fit the operator that advances the state, eigendecompose it, and
each eigenvalue is a decay rate and a frequency.

A conserved quantity is the special case `g(s') = g(s)`. The general object is
`g(s') = λ·g(s)`, and `μ = log(λ)/Δt` splits into two physical numbers:

```
Re μ = 0   undamped oscillation   — the eigenvalue sits ON the unit circle
Re μ < 0   dissipation at rate |Re μ| — INSIDE the circle
Im μ       the frequency of that mode
```

So "no conserved law" is never "no law". A damped oscillator's law is a spiral,
and the spiral has a measurable rate.

## Fit the operator

Extended dynamic mode decomposition: lift the state, then solve one linear
least-squares problem for the matrix that maps this feature vector to the next.

```python
import numpy as np

def koopman(trajs, phi, dt, ridge=1e-6):
    """K is the operator in feature space: phi(s') ~ K^T phi(s)."""
    Phi  = np.vstack([phi(t[:-1]) for t in trajs])   # every state
    Phip = np.vstack([phi(t[1:])  for t in trajs])   # its successor
    # Ridge keeps the Gram matrix invertible when features are near-collinear,
    # which random features on a low-dimensional attractor always are.
    G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    K = np.linalg.solve(G, Phi.T @ Phip)
    lam = np.linalg.eigvals(K)
    lam = lam[np.abs(lam) > 1e-6]                 # drop numerically dead modes
    return np.log(lam.astype(complex)) / dt       # continuous-time eigenvalues
```

The lift is the same one a [conservation
law](discover-a-conservation-law.md) uses — 400 random Fourier features on the
state, scaled by the data's own spread. Nothing about the system enters the
features.

## Select the mode you mean

A trajectory set produces hundreds of eigenvalues, and most are numerical
debris or harmonics. Select the **slowest** oscillatory mode — the one closest
to the imaginary axis — not the lowest-frequency one:

```python
def dominant_mode(mu, fmax=5.0):
    """The system's natural spiral: oscillatory, and the slowest to decay."""
    osc = mu[(np.abs(mu.imag) > 0.05) & (np.abs(mu.imag) < fmax)
             & (mu.imag > 0)]
    return osc[np.argmin(np.abs(osc.real))] if len(osc) else None
```

Selecting by frequency instead reads noise. The damped oscillator's
lowest-frequency modes are `freq 0.1315, rate 0.351` and `freq 0.1714, rate
149.3` — the second has `|λ| = 0.00057`, a mode that dies inside one step.
Selecting by `|Re μ|` returns `rate 0.0998, freq 0.9951` instead.

## Measured: rate and frequency against the truth

Six systems, raw trajectories only, one fit each:

| System | \|λ\| | decay rate | frequency | true rate | true freq |
|---|---|---|---|---|---|
| harmonic oscillator | 1.00000 | −0.0000 | 1.0000 | 0.0000 | 1.0000 |
| pendulum (nonlinear) | 0.99996 | 0.0007 | 0.6412 | — | — |
| Lotka–Volterra | 1.00000 | 0.0000 | 0.6605 | — | — |
| Kepler orbit (2-body) | 0.99999 | 0.0019 | 0.7728 | — | — |
| damped oscillator | 0.99502 | **0.0998** | **0.9951** | 0.1000 | 0.9950 |
| Lorenz attractor | 1.01078 | −1.0727 | 0.3142 | — | — |

The damped oscillator `ẍ + 2ζω₀ẋ + ω₀²x = 0` at `ζω₀ = 0.1, ω₀ = 1` has
friction `0.1` and damped frequency `ω₀√(1 − ζ²) = 0.99499`. The operator reads
`0.0998` and `0.9951` off the trajectories, with no equation supplied and no
knowledge that friction exists. The four conservative systems sit on the
imaginary axis to four decimals; that flatness is the eigenvalue-1 case, which
is the [invariant](../terms/invariant.md) the other page recovers.

## The spectrum resolves nonlinearity as structure

A nonlinear oscillator has no single frequency — the pendulum's period grows
with amplitude. Fitting one operator on five small swings (`|q₀| ≤ 0.6`) and
listing its near-axis modes:

| measured freq | 0.9973 | 0.9929 | 0.9910 | 0.9848 | 0.9776 |
|---|---|---|---|---|---|
| amplitude | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
| true freq (elliptic) | 0.9975 | 0.9944 | 0.9900 | 0.9844 | 0.9775 |

One eigenvalue per fitted amplitude, tracking the elliptic-integral period
across the band: three of the five agree to within 0.0004 and the widest
disagreement is 0.0015. A second family appears at `1.9551, 1.9698, 1.9831` —
the harmonics `2ω` of the same motion, not new physics. Amplitude enters as a
*set* of eigenvalues rather than as a parameter: a linear operator in a
nonlinear feature space represents the nonlinearity as spectrum.

## Validity is bounded by the region the trajectories covered

The fit is a statement about the visited part of state space and nothing else.
The small-swing operator above, applied to swings it never saw:

| amplitude | 0.4 (in region) | 1.0 | 2.0 | 2.8 |
|---|---|---|---|---|
| one-step feature error | **0.0011** | 6.5070 | 3.0768 | 3.0298 |
| true frequency | 0.9900 | 0.9378 | 0.7525 | 0.4949 |

Relative one-step error rises by three orders of magnitude at the first
amplitude outside the fitted band. Nothing in the spectrum announces this: the
eigenvalues still look clean, because they describe a region the query is no
longer in. Report the region a fitted operator covers alongside its rates, and
re-fit on a rolling window when the system can leave that region.

## The chaotic case: measure divergence, not a law

Chaos has no dominant mode worth quoting. The useful reads are geometric —
phase-space contraction and the maximal Lyapunov exponent — both measured from
the same trajectories:

```python
def divergence(f, S, h=1e-5):
    """Mean rate of phase-space volume change: 0 conservative, <0 leaking."""
    d = S.shape[1]; div = np.zeros(len(S))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        div += (np.array([f(s + e)[i] for s in S])
                - np.array([f(s - e)[i] for s in S])) / (2 * h)
    return div.mean()

def max_lyapunov(f, s0, dt, T=150.0, d0=1e-8):
    """Benettin: track a shadow trajectory, renormalise its separation each
    step, and average the log growth. Integrate many periods -- regular
    orbits only converge to ~0 over long horizons."""
    s = s0.copy(); sp = s0 + d0 * rng.standard_normal(len(s0)); acc = 0.0
    for _ in range(int(T / dt)):
        s  = rk4(f, s,  1, dt)[0]
        sp = rk4(f, sp, 1, dt)[0]
        dvec = sp - s; dist = np.linalg.norm(dvec)
        acc += np.log(dist / d0 + 1e-18)
        sp = s + (d0 / (dist + 1e-18)) * dvec    # renormalise, keep direction
    return acc / (int(T / dt) * dt)
```

| System | divergence | λ_max | character |
|---|---|---|---|
| harmonic oscillator | +0.0000 | +0.0043 | conservative |
| pendulum (nonlinear) | +0.0000 | +0.0051 | conservative |
| Lotka–Volterra | −0.0139 | +0.0098 | conservative |
| Kepler orbit | +0.0000 | +0.0295 | conservative |
| damped oscillator | **−0.2000** | −0.1015 | dissipative |
| Lorenz attractor | **−13.6667** | **+0.8648** | chaotic |

Both dissipative readings are exact: the damped system's Jacobian has trace
`−2ζω₀ = −0.2`, and Lorenz's is `−(σ + 1 + β) = −13.6667`. The measured
`λ_max = 0.8648` against the published `≈0.906` is the classifying number —
positive means nearby trajectories separate exponentially, so a state estimate
loses one significant figure roughly every `ln(10)/0.865 ≈ 2.7` time units. For
such a system the honest deliverable is that horizon, not a forecast beyond it.

The three characters partition on two numbers: `λ_max > 0` is chaotic;
otherwise `divergence < 0` is dissipative and `≈ 0` is conservative.

## A fitted operator on a real series

The same construction on a rolling window of a market state vector (5
macro-features, 2 310 days, VAR(1) refitted on a trailing year) reports a mean
spectral radius of `0.956` and a self-reported horizon `−1/ln ρ` of `27.2`
days. Those numbers are not informative: correlation between the self-reported
horizon and the realised 5-day forecast skill is `+0.096`, and between spectral
radius and forward 10-day volatility `−0.016`. The same state's raw volatility
coordinate predicts its own future at `+0.369`.

A near-unit-root operator has a spectrum that barely moves, so its
eigenvalues carry no self-knowledge even while the fit succeeds. Physics
systems separate on the spectrum because their eigenvalues are far from the
unit circle in a repeatable way; the boundary is the data's, not the method's.

## Related

- [Discover a conservation law](discover-a-conservation-law.md) — the
  eigenvalue-1 case, and the control that must decline.
- [Forecast a time series](forecast-a-time-series.md) — prediction when the
  operator is not the useful object.
- [Build a world model](build-a-world-model.md) — the same prediction problem
  stored as bindings, with rollout decay measured.
- [Invariant](../terms/invariant.md) · [World model](../terms/world-model.md) ·
  [Held-out evaluation](../terms/held-out-evaluation.md)
