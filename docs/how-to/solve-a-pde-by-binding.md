# Solve a PDE by binding

Evolve a field — an image, a temperature grid, an option price surface — to any
time you ask for, in one round trip per timestamp, with no timestepping loop.

A linear evolution operator that commutes with translation is *diagonal* in the
frequency basis: it multiplies each frequency by a number and mixes nothing.
[`bind`](../terms/bind.md) is circular convolution, which is exactly
multiplication in that basis, so the whole evolution is one bind with one
kernel vector. Time enters as a fractional power of that kernel, so
[`/vec/pow`](../api/vectors.md#post-vecpow) hands you any timestamp — `t = 0.3`,
`t = 137`, `t = −2` — at the same cost as `t = 1`.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| The field is | a `d`-length array of samples on a uniform grid | bind operates on raw vectors; the grid *is* the vector |
| The operator is | one **generator vector** whose spectrum is the symbol at `t = 1` | every later timestamp is a power of it |
| A timestamp is | `pow(generator, t)` | fractional powers of a diagonal operator are exact, not interpolated |
| Evolution is | `bind(field, clock)` with `normalize: false` | the spectrum magnitude *is* the decay; normalising would erase it |
| Boundaries are | **periodic** | circular convolution wraps; this is a constraint on the problem, not an approximation of a better one |
| Cost per timestamp | one `pow` + one `bind` (1-D) | independent of how far away `t` is |

**Periodic boundaries are the price of admission.** The left edge of the field
is the right edge's neighbour. A problem with Dirichlet or Neumann walls needs
its domain mirrored or padded until the wrap is harmless, and a field with a
jump across the seam rings. The Black–Scholes example below pays this with a
taper; a photograph pays it with a border of dead pixels.

## The recipe

1. Write the operator's **symbol** — its multiplier at each frequency, at
   `t = 1`.
2. Inverse-transform the symbol once to get the generator vector. Store it.
3. For a timestamp, `pow(generator, t)`.
4. Bind the field with the result.

Everything except step 4 happens once per operator, not once per query.

## Heat, end to end

`u_t = α ∇²u` has symbol `e^{−αω²}` — a real, sub-unit multiplier at every
frequency, largest at `ω = 0`. High frequencies die first, which is what
diffusion *is*.

```python
import numpy as np
import requests

ENGINE = "http://localhost:6393"
S = requests.Session()

def call(path, body):
    """Every /vec route answers {"result": [...]}."""
    r = S.post(ENGINE + path, json=body, timeout=120)
    r.raise_for_status()
    return np.asarray(r.json()["result"])

def bind(a, b):
    # normalize=False is mandatory here: the heat kernel's spectrum has
    # magnitude < 1, and that magnitude is the decay. Normalising would
    # rescale the field back up and destroy the physics.
    return call("/vec/bind", {"a": list(map(float, a)),
                              "b": list(map(float, b)),
                              "normalize": False})

def vpow(a, t):
    return call("/vec/pow", {"a": list(map(float, a)), "t": float(t)})

N, ALPHA = 128, 0.15

def heat_generator():
    """The operator at t = 1, as a vector.

    omega_j = 2*pi*j/N are the grid's frequencies; the symbol e^{-alpha w^2}
    is real and positive, so this kernel only shrinks and never rotates.
    irfft turns the symbol into the time-domain vector the engine binds with.
    """
    w = 2 * np.pi * np.arange(N // 2 + 1) / N
    return np.fft.irfft(np.exp(-ALPHA * w**2), n=N)

G = heat_generator()          # stored once

u0 = np.zeros(N)
u0[48:80] = 1.0               # a box of heat on a ring

for t in (0.5, 4.0, 20.0):
    clock = vpow(G, t)        # this timestamp: one call
    u = bind(u0, clock)       # the solution at t: one call

    # Check against the closed-form spectral solution of the same PDE.
    w = 2 * np.pi * np.fft.fftfreq(N)
    exact = np.real(np.fft.ifft(np.fft.fft(u0) * np.exp(-ALPHA * w**2 * t)))
    print(f"t={t:5.1f}  max|err| = {np.abs(u - exact).max():.3e}")
```

Measured against the closed form, on the engine:

```
t=  0.5  max|err| = 5.033e-16
t=  4.0  max|err| = 6.106e-16
t= 20.0  max|err| = 2.554e-15
```

The error does not grow with `t`. There is no step to accumulate error in:
`t = 20` is one call, not four hundred calls of `t = 0.05`.

## Two dimensions, by separability

`e^{−α(ωx² + ωy²)t}` factors into `e^{−αωx²t} · e^{−αωy²t}`. A separable
operator applies along one axis and then the other — the same generator, the
same clock, `2N` binds instead of one.

```python
def diffuse(field, clock):
    """2-D heat: bind every row, then every column, with the same clock.

    Separability is what makes this cheap. A non-separable 2-D operator
    needs a genuine 2-D transform and does not decompose into row/column
    binds."""
    rows = np.array([bind(r, clock) for r in field])
    return np.array([bind(c, clock) for c in rows.T]).T

y, x = np.mgrid[0:N, 0:N]
img = np.zeros((N, N))
img[(x - 40)**2 + (y - 44)**2 < 22**2] = 1.0    # a disk
img[88:104, 24:104] = 0.8                       # and a bar

for t in (0.5, 2.0, 8.0):
    frame = diffuse(img, vpow(G, t))            # 1 pow + 256 binds
```

Against the closed-form 2-D spectral solution:

```
2-D t= 0.5  max|err| = 8.882e-16
2-D t= 2.0  max|err| = 8.882e-16
2-D t= 8.0  max|err| = 2.442e-15
```

Each frame costs `1 + 2N = 257` engine calls, whatever `t` is, and each frame
is computed from the *original* image — never from the previous frame.

## Running time backwards

The clock at `t = −2` is the same call with a negative exponent. Because the
symbol is `e^{−αω²t}`, a negative `t` inverts every frequency's decay exactly.

```python
# Diffuse forward, then undo it. No separate "deblur" code path exists:
# the inverse of a diagonal operator is the same operator at -t.
forward = diffuse(img, vpow(G, 2.0))
back    = diffuse(forward, vpow(G, -2.0))
print(f"{np.abs(back - img).max():.3e}")
```

On clean data this recovers the initial field to the float64 budget:

| Reverse from | Noise added at `t` | max &#124;err&#124; vs the original |
|---|---|---|
| `t = 2` | none | `2.010e-14` |
| `t = 8` | none | `2.073e-07` |
| `t = 0.5` | `1e-6` | `7.234e-06` |
| `t = 0.5` | `1e-3` | `7.234e-03` |
| `t = 8` | `1e-6` | `1.215e+03` |
| `t = 8` | `1e-3` | `1.215e+06` |

The blow-up is the physics, not a defect of the method. Backward heat
amplifies the frequency `ω` by `e^{+αω²t}`; the worst mode on this grid at
`α = 0.15, t = 8` carries a bound of `e^{2απ²t} ≈ 2·10¹⁰`, and the measured
gain from a `1e-6` perturbation to a `1.2e3` error is `1.2·10⁹`. Any method
that reports a clean answer here is regularising and not telling you.

Two rules follow, and they are usage rules rather than warnings:

- **The clean-data reversal is exact.** Error at `t = 8` sits at `2e-7` only
  because the forward decay `e^{−2απ²·8}` has already pushed the top
  frequencies below the double-precision floor; there is nothing left to
  invert. Shorter hops recover to `1e-14`.
- **The usable backward horizon is set by the noise floor.** A field known to
  `ε` is invertible over `t` while `ε · e^{2απ²t} ≪ ‖u‖`. At `α = 0.15` and
  `ε = 1e-6`, `t = 0.5` is comfortable and `t = 8` is meaningless.

## The same machinery, other kernels

Only the symbol changes. The calls do not.

**Burgers' equation**, `u_t + u·u_x = ν·u_xx`, is nonlinear, and the
Cole–Hopf substitution `u = −2ν·φ_x/φ` maps it *exactly* onto the heat
equation for `φ`. The nonlinearity moves entirely into the pointwise encode
and decode; the flow between them is one heat clock. Per queried timestamp:
one `pow`, one `bind` to evolve, one more `bind` with the `d/dx` operator to
decode. Against a 15,000-step pseudo-spectral RK4 reference at `ν = 0.1`,
`u₀ = sin x`, `N = 512`:

```
t=0.25: max|err| = 1.67e-10        t=1.50: max|err| = 2.94e-11
t=0.75: max|err| = 9.66e-11        t=3.00: max|err| = 2.26e-12
```

with `max |du/dx|` rising from `1.00` to `2.26` by `t = 0.75` — the shock
steepening, i.e. the nonlinear term doing its work, reproduced by a linear
clock in the transformed variable.

**Black–Scholes**, in log-price coordinates `x = ln(S/K)`, is a
constant-coefficient convection–diffusion–discount equation, so it is diagonal
too. Its generator has three parts and each is financial:

```python
w = 2 * np.pi * np.fft.rfftfreq(N, DX)
gen = (-0.5 * SIGMA**2 * w**2       # magnitude decay  = volatility
       + 1j * (R - 0.5 * SIGMA**2) * w   # phase rotation = risk-neutral drift
       - R)                          # uniform contraction = discounting
clock1 = np.fft.irfft(np.exp(gen), n=N)
```

Price is `bind(payoff, pow(clock1, tau))` — one call per maturity from one
stored generator — and delta is one more bind with the `d/dx` operator.
Measured across `T ∈ {0.25, 0.5, 1, 2}` and five spots, `N = 4096`,
`K = 100, r = 0.05, σ = 0.20`: worst price error `1.61e-4` on options worth
\$0.06–\$32, worst delta error `7.15e-6`. The payoff is tapered far from the
strike, which is the periodic-boundary constraint being paid in cash: a call
payoff does not wrap, so the wrap is put somewhere it costs nothing.

**An ODE** is the degenerate case where the field is one mode. The damped
oscillator `x'' + 2γx' + ω₀²x = 0` propagates by repeated binds with the clock
`e^{(−γ + iω_d)Δt}`: at `Δt = 0.4` over 25 steps, max error `8.29e-16`, where
explicit Euler at the same step reaches `59.2`. The semigroup identity
`ϕ(Δt)^⊗n = ϕ(nΔt)` collapses those 25 binds into one `pow`, to `6.31e-16`.

## Where this applies

The method needs all four: **linear**, **translation-invariant**,
**periodic**, and **diagonal in the frequency basis**. A variable-coefficient
operator (`α(x)∇²`) is not translation-invariant and does not diagonalise. A
nonlinear operator needs its nonlinear term evaluated pointwise outside the
basis — which is what Burgers does exactly, through Cole–Hopf, and what a
Navier–Stokes stepper does approximately, per step. See
[Physics in the frequency domain](../explanation/physics-in-the-frequency-domain.md)
for why the boundary sits exactly there.

## Related

- [Do calculus with vectors](do-calculus-with-vectors.md) — the `d/dx`
  operator vector used above, and its exact inverse.
- [Physics in the frequency domain](../explanation/physics-in-the-frequency-domain.md)
  — why one bind is a whole evolution.
- [Encode quantities and time](encode-quantities-and-time.md) — `pow` as a
  quantity encoder, the same operation in its other role.
- [Vector endpoints](../api/vectors.md) — `bind`, `unbind`, `pow` request
  shapes, and the `normalize` / `exact` flags.
- [Vector algebra](../explanation/vector-algebra.md) ·
  [bind](../terms/bind.md) ·
  [fractional power encoding](../terms/fractional-power-encoding.md).
