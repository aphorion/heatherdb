# A physicist's guide to HeatherDB

HeatherDB stores vectors. This guide is about the discovery that the
*operations on those vectors are physics*: binding is the exponential
propagator, the spectrum bins are your Fourier modes, and the memory
machinery (consolidation, compression) is a working scientific method.
Everything below has been verified against analytic ground truth on a
live engine; the numbers quoted are measured, not estimated. The
companion scripts live in `heather_research/math/spiral_paper/`.

Requires the engine from branch `feat/spiral-plane-algebra`
(`/vec/pow`, exact unbind, raw bind) and, for §6, branch
`feat/native-operator-fit` (`/vec/fit`).

---

## 1. The one idea

A real vector of dimension D carries D/2 complex Fourier bins. Put a
**generator** s = σ + iω in each bin's exponent and you have encoded a
point on the complex-frequency plane. The engine's bind operation
(`/vec/bind`) is circular convolution — elementwise *multiplication of
spectra* — so:

```
bind( e^{s·t1}, e^{s·t2} )  =  e^{s·(t1+t2)}
```

Binding **adds the exponents**. Everything else follows from where you
put the generator:

| generator location | regime | what binding does |
|---|---|---|
| imaginary axis, s = iω (unit circle) | reversible | arithmetic, function algebra, translation |
| off-axis, s = σ + iω (log spiral) | irreversible | growth, decay, **time evolution** e^{tA} |

The slogan, earned below: **time becomes a query parameter**. Store a
generator once; any past or future state is a read.

## 2. The dictionary

| physics | substrate object | engine call |
|---|---|---|
| field / state u(x) | real vector (samples) | — |
| Fourier mode k | spectrum bin k | — |
| evolution operator e^{Δt·A} | "clock" vector | `POST /vec/bind` (`normalize:false`) |
| e^{n·Δt·A} (any horizon) | clock to the n-th power | `POST /vec/pow` |
| d/dx, resolvents, filters | "operator" vectors | `POST /vec/bind` |
| exact inverse / division | spectral division | `POST /vec/unbind` (`exact:true`) |
| system identification (Wiener/DMD) | operator fit from pairs | `POST /vec/fit` |
| measurement / nearest-pattern | EAM read | `POST /collections/{c}/read` |
| experience → memory | write | `POST /collections/{c}/write` |
| abstraction (MDL) | compression | `POST /collections/{c}/compress` |

Two conventions you must respect (each cost us a debugging session):

- **The π cap.** `/vec/pow` exponentiates on the principal branch, so a
  bin's effective frequency is ω wrapped into (−π, π]. This is the real
  carrier's Nyquist limit in generator space. Keep |ω| < π or your
  encoding silently aliases.
- **The transform sign.** Under numpy's FFT convention the derivative
  multiplier is **+iκ**. Get it backwards and your call deltas come out
  negative (ask us how we know).

## 3. Ten lines to a damped oscillator

Build a clock whose every interior bin is e^{s·Δt} with
s = −γ + iω_d, bind repeatedly, and you are integrating
x″ + 2γx′ + ω₀²x = 0 — **exactly**:

```python
import numpy as np, requests
d, g, w0, dt = 8000, 0.25, 2.0, 0.4
wd = np.sqrt(w0**2 - g**2); s = -g + 1j*wd
mk = lambda bins: np.fft.irfft(np.r_[0, bins, 0], n=d)   # DC=Nyq=0
clock = mk(np.full(d//2-1, np.exp(s*dt)))
state = mk(np.full(d//2-1, 0.5))                          # a0 = x0/2, v0=0
post = lambda p, b: np.array(requests.post(f"http://localhost:8419{p}",
                                           json=b).json()["result"])
for n in range(25):
    state = post("/vec/bind", {"a": state.tolist(), "b": clock.tolist(),
                               "normalize": False})
x_t = 2*np.real(np.fft.rfft(state)[1:d//2].mean())        # readout
```

Measured: max error **8.3e-16** over 25 steps at a step size where
explicit Euler diverges to ~60. The reason is structural, not
numerical: a timestepper truncates the Taylor series of e^{s·Δt};
binding *is* e^{s·Δt}. There is no truncation error to accumulate.
And the semigroup identity collapses the loop entirely:

```python
horizon = post("/vec/pow", {"a": clock.tolist(), "t": 25})   # φ(Δt)^25 = φ(25Δt)
```

— the whole trajectory point in one call (measured: 6.3e-16).

## 4. Calculus

Encode a function as a superposition of encoded points,
F = Σ f(x_k)·φ(x_k). Then in the spectrum, differentiation is
*multiplication by a fixed vector* — which is to say, **the derivative
is something you bind with**:

- **differentiate**: `bind(F, S)` where S's bins are the derivative
  multiplier (measured corr vs analytic: 0.9997);
- **integrate**: `unbind(F, S, exact=true)` — spectral division. The
  DC bin cannot be divided; `eps` zeroes it; that zeroed bin **is the
  constant of integration**, falling out of the algebra (0.9987).

Operator algebra generally: any function of d/dx that is diagonal in
Fourier — resolvents (sI−A)⁻¹, transfer functions, filters, fractional
derivatives — is just another vector. Applying it is one bind. We
priced options this way (§5) and computed forced responses at corr
0.993 by binding a forcing with a resolvent.

## 5. PDEs

Every constant-coefficient PDE that is diagonal in the Fourier basis
is a clock:

| equation | clock spectrum (per Δt) | notes |
|---|---|---|
| transport u_t = −c u_x | e^{−icκΔt} | pure phase |
| heat u_t = α u_xx | e^{−ακ²Δt} | contractive; **exactly reversible on clean data**, and reversing amplifies noise by e^{2απ²t} — the ill-posedness of backward heat, measured |
| wave / Schrödinger | unitary phases | circle regime |
| Black–Scholes (log-price) | e^{(−σ²κ²/2 + i(r−σ²/2)κ − r)Δt} | **volatility = magnitude decay, drift = phase, discounting = uniform contraction**; price = one bind, each maturity = one pow, delta = one more bind (measured: 1.6e-4 on price, 7e-6 on delta) |

Higher dimensions: separable multipliers (heat, derivatives) apply as
row binds then column binds. Non-separable ones decompose — most
beautifully the **Poisson solve**:

```
(−∇²)⁻¹ = ∫₀^∞ e^{t∇²} dt   ≈   Σ_j w_j · (heat clock at time s_j)
```

The inverse Laplacian is the time-integral of diffusion, so an
elliptic solve is a weighted **sum of heat clocks** (28 log-spaced
nodes fit 1/λ to rel. error 1.3e-9; fit the weights by least squares
with rows scaled by λ, not 1/λ).

**Nonlinear PDEs** don't have one-clock propagators (that's physics,
not a tool gap — the cascade creates information). They step, the way
production pseudo-spectral CFD steps, and every ingredient is native:

- derivatives: binds with D (per axis);
- the stiff viscous half: an **exact** clock — no step-size penalty,
  ever;
- quadratic nonlinearity u·∇u: pointwise products ("gates");
- 2/3-rule dealiasing: a mask bind. **Not optional** — omit it and the
  simulation NaNs within half a time unit.

Measured: 2-D Navier–Stokes vortex merger, 403,200 live binds, corr
**1.00000** against an exact-Poisson RK4 reference at 20× finer
timestep. And when a transform linearizes the nonlinearity exactly —
Cole–Hopf for Burgers, Koopman lifts for finite-subspace systems — the
clock comes back: Burgers to 1e-10 vs 15,000 RK4 steps in **12 calls**;
a Koopman-lifted nonlinear system to 7.4e-11.

## 6. Learning physics from data

This is where the database half and the physics half meet.

**Transitions are unbinds.** If s₁, s₂ are consecutive snapshots, then
`unbind(s₂, s₁, exact=true)` *is* the propagator over that interval —
one noisy local glimpse of the law. Practical detail: store the
**generator increment** (propagator − identity); propagators of
different laws are all δ-plus-small and nearly parallel as vectors,
while increments point in genuinely different directions.

**Dreaming is law discovery.** Bulk-load engrams, run MDL compression
(`/compress`), read the surviving prototypes. Measured: 128 engrams
from two hidden advection–diffusion worlds → exactly **2 prototypes,
perfectly split**, 45× compression — and the coefficients read
straight off the prototype's bins (log of bin k over Δt =
−νk² − icκ): ν to four digits.

**Refinement is in-store gradient descent.** The backward pass through
a bind is a bind with the involution (conjugate spectrum). Two
subtleties worth their own lines:

- plain **SGD beats Adam** here: SGD's per-bin rate |S_k|² *is*
  inverse-variance weighting, so its fixed point is the
  Wiener-optimal operator. Adam's normalization destroys that
  weighting and amplifies noise bins. Stability bound:
  lr < 2/max|S_k|².
- the converged fixed point has a closed form, which is why it became
  an engine endpoint: `POST /vec/fit {sources, targets, ridge}` →
  per-bin Σ T S̄ / (Σ|S|² + ridge). One call replaces the training
  loop.

**Symbolic extraction.** Fit the dreamed generator over the dictionary
{(iκ)⁰ … (iκ)⁵} with MDL term selection (data bits + 32 bits per
term, exhaustive over subsets). Measured: 5/5 hidden equations
recovered with the correct *sparsity pattern* — dispersion (u_xxx)
distinguished from diffusion (u_xx) from hyperdiffusion (u_xxxx), no
hallucinated terms, coefficients to 3-4 digits. The substrate ends the
pipeline by *writing down the PDE*.

**Hierarchy.** Abstract over many dreamed laws and the law-space
collapses: 12 advection–diffusion worlds → singular spectrum with a
**154× rank-2 gap**, and the 2-D basis *is* {∂²/∂x², ∂/∂x} (subspace
residuals 0.0036 / 0.0008). The substrate discovered the differential
operators as the basis of all the physics it had seen — after which a
brand-new world is learned from **one transition** by projecting onto
the law manifold (12.6× error reduction, corr 1.00000).

**Turbulence closures.** Learn on the *residual* (what the resolved
coarse solver misses), never on the full dynamics — a linear model
asked to replace the resolved cascade loses by 33×. Hebbian DMD (two
correlation bundles + one divide) on coarse Burgers turbulence gives a
subgrid operator whose per-bin readout is the **textbook eddy-viscosity
cusp**, and whose LES restores the resolved energy budget to ratio
1.000 vs the no-model run's 1.034 (95× less budget error). Judge
closures statistically (energy, spectra) — shock positions are chaos,
not error.

## 7. Native deep learning (for the physicist who also trains models)

A bind layer is a circulant matrix (diagonal in the spectrum); a
time-domain elementwise gate is its complement. Alternating them is
universal for linear maps (Huhtanen–Perämäki) — and is *also* exactly
the **Fourier Neural Operator**, the architecture ML converged on for
PDE surrogates. Backprop is native: adjoint of a bind = bind with the
involution; adjoint of a gate = elementwise. Quadratic activations are
self-binds, and the product rule **is a bind**:
δ_u = 2(δ_q ⊛ involve(u)). Measured: deep gate–bind nets train through
the commutative wall (12× separation); Volterra nets learn nonlinear
interaction terms where every linear model scores exactly zero.

## 8. The singularity observatory

Because flows are stored as spectra, the **analyticity strip** δ(t) —
the classical Sulem–Sulem–Frisch blowup diagnostic — is read directly
off the bins as the slope of the log spectral tail. Finite-time blowup
⟺ δ → 0. Paired with MDL model comparison on 1/‖ω‖∞ (finite-time
touchdown c(T−t) vs exponential approach — whichever compresses
better), the substrate *classifies* blowup vs survival from data.
Measured on Constantin–Lax–Majda (exact blowup known since 1985):
blowup time 2.0010 vs theorem's 2, exponent 0.999 vs theorem's 1,
strip touchdown at 2.0001 by the independent observable.

Caveat that bit us: the spectral tail has an engine round-trip noise
floor (~1e-13). Fit δ only on bins above ~1e-9, or flat noise
masquerades as strip collapse.

## 9. The receipts

All live-verified, with scripts and figures in
`heather_research/math/spiral_paper/`:

| claim | number | script |
|---|---|---|
| arithmetic by binding | cos = 1.0 exact | `calculus_vsa.py` |
| derivative / integral | corr 0.9997 / 0.9987 | `calculus_vsa.py` |
| ODE by repeated bind | 8.3e-16; Euler diverges | `ode_binding.py` |
| whole trajectory in one pow | 6.3e-16 | `ode_binding.py` |
| Koopman-lifted nonlinear flow | 7.4e-11 vs RK4 | `spiral_all.py` |
| heat equation (image) | 1e-15; reversible | `heat_image.py` |
| Black–Scholes + delta | 1.6e-4 / 7e-6 | `black_scholes.py` |
| Burgers via Cole–Hopf | 1e-10, 12 calls | `burgers.py` |
| 2-D Navier–Stokes | corr 1.00000 | `navier_stokes.py` |
| in-store gradient descent | 0.999996 vs closed form | `intrinsic_backprop.py` |
| deep linear, commutative wall | 12× separation | `deep_linear.py` |
| Volterra interaction learning | 5×; linear class scores 0 | `volterra.py` |
| law discovery by dreaming | 2 laws, 4-digit coefficients | `dream_the_law.py` |
| law-space basis = {∂², ∂} | 154× rank gap | `dream_the_calculus.py` |
| eddy viscosity by dreaming | budget 1.000 vs 1.034 | `ns_eddy.py` |
| symbolic equation extraction | 5/5, no spurious terms | `symbolic_extraction.py` |
| singularity detection (CLM) | T = 2.0010 vs 2 | `clm_blowup.py` |

## 10. Sharp edges, collected

1. FPE bandwidth caps at **π** (principal branch) — generators outside
   wrap and alias silently.
2. Fractional `pow` needs **non-negative DC/Nyquist** bins; build FPE
   bases with DC = Nyquist = 1, never 0 (0^negative explodes).
3. `normalize:false` on `/vec/bind` is **mandatory** off the circle —
   normalization erases exactly the growth/decay that is the
   computation.
4. Default `/vec/unbind` is correlation (conjugate): exact *only* on
   unit-magnitude spectra. Off the circle use `exact:true` —
   conjugation multiplies magnitudes where division is needed.
5. The derivative multiplier is **+iκ** in numpy's convention; a sign
   flip phase-scrambles everything downstream.
6. Quadratic terms need **2/3-rule dealiasing** or pseudo-spectral
   stepping NaNs.
7. Learn closures on the **residual** of the resolved solver; compare
   chaotic systems **statistically**.
8. SGD over Adam for spectral operator fitting (inverse-variance);
   lr < 2/max|S_k|²; or skip the loop entirely with `/vec/fit`.
9. Spectral-tail measurements have a ~1e-13 noise floor; threshold at
   1e-9.
10. `bulk_load` **replaces** the collection — one call per dataset.
11. Compare against analytics at the **grid node's own coordinates**;
    nearest-node snapping fakes errors orders above the solver's.
12. Engine f64 serialization turns inf/NaN into JSON null — a learning
    rate that diverges comes back as `None`, not an exception.

## 11. Where this is going

The arc this guide documents ran: arithmetic → calculus → ODEs → PDEs
→ in-store learning → law discovery → symbolic extraction →
singularity detection — every step on the same two operations. The
open frontiers, with working instruments already pointed at them: the
optimization basins of deep substrate networks, learned PDE surrogates
on the native FNO architecture, and the Navier–Stokes program (strip
laws, monotone-functional search, self-similar profile hunting).

The paper behind all of this: *"Arithmetic, Calculus, and Dynamics as
Binding: A Unified View of Vector-Symbolic Computation on the
Complex-Frequency Plane."* The engine is the running proof.
