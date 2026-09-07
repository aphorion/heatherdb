# Physics in the frequency domain

A linear operator that is diagonal in the Fourier basis becomes a single
multiplication there — and multiplication in that basis is exactly what
[binding](../terms/bind.md) is. Evolving a field is therefore one database
operation, and time is a parameter of it rather than a loop.

## Why binding is a frequency-domain multiplication

Binding is circular convolution, computed through the FFT: transform both
operands, multiply pointwise, transform back. The convolution theorem says
convolution in the spatial domain *is* pointwise multiplication in the
frequency domain, so a bind is a multiplication of spectra wearing spatial
clothes.

This is normally described as an implementation detail of the algebra — how
[bind](../terms/bind.md) happens to be computed. Read the other way it is a
statement about what the store can do: any operation expressible as a pointwise
multiplication of spectra is available as one call.

## Which operators qualify

An operator is diagonal in the Fourier basis when its eigenfunctions are
complex exponentials — which is the case exactly when it is **linear** and
**translation-invariant**. Differentiation is such an operator, and so is every
polynomial in it, so a large part of classical physics qualifies:

| Operator | Spectrum | What it does |
|---|---|---|
| `∂/∂x` | `iω` | differentiate |
| `∫ dx` | `1/(iω)` | integrate |
| `∇²` | `−ω²` | Laplacian |
| heat, `e^{tα∇²}` | `e^{−αω²t}` | diffuse for time `t` |
| a Gaussian blur | `e^{−σ²ω²/2}` | smooth |
| a fractional derivative | `(iω)^s` | interpolate between the above |

Each row is a vector. Applying the operator is binding with it. Composing two
operators is binding their vectors — and because multiplication is
commutative, operators that commute in mathematics commute here for the same
reason.

## Time as an exponent

The heat kernel's spectrum is `e^{−αω²t}`, which is the `t`-th power of
`e^{−αω²}`. Raising a vector to a fractional power is a spectral operation the
store performs directly ([`/vec/pow`](../api/vectors.md)), so a kernel for one
unit of time is a kernel for **any** time:

```
clock(t) = pow(clock(1), t)
```

Two consequences follow, and they are the reason to do physics this way at all.

**There is no step error, because there are no steps.** A conventional solver
advances by `dt` and accumulates a local truncation error at every step; error
grows with the number of steps and therefore with `t`. Here `t = 20` is one
call. Measured against the closed form, the error at `t = 0.5`, `t = 4` and
`t = 20` sits at `5.0e-16`, `6.1e-16` and `2.6e-15` — the float64 floor,
flat in `t`.

**Any timestamp costs the same.** Evaluating a field at `t = 137.4` does not
require computing the 137 preceding states. This makes the solution
random-access in time, which is a different object from a trajectory: you can
query it the way you query a store, because it is one.

Higher dimensions come from separability. The 2-D heat spectrum
`e^{−α(ωx² + ωy²)t}` factors into `e^{−αωx²t} · e^{−αωy²t}`, so a separable
operator applies along one axis and then the other — the same clock, `2N` binds
instead of one.

## Negative exponents, and why the answer is allowed to explode

Nothing in `pow` requires `t > 0`. A negative exponent runs the operator
backwards, and for a diffusive kernel that inverts `e^{−αω²t}` into
`e^{+αω²t}` — a spectrum that amplifies rather than damps, worst at the highest
frequency on the grid.

On clean data this is exact: reversing from `t = 2` recovers the initial field
to `2.0e-14`. Under noise it diverges, and the divergence is quantitative
rather than qualitative. A field known to precision `ε` is invertible over `t`
only while `ε · e^{2απ²t} ≪ ‖u‖`. At `α = 0.15`, a `1e-6` perturbation
reversed from `t = 0.5` yields an error of `7.2e-06`; reversed from `t = 8` it
yields `1.2e+03`. The theoretical amplification bound at that horizon is
`e^{2απ²t} ≈ 2·10¹⁰` and the measured gain is `1.2·10⁹`.

This is the correct behaviour of backward heat, not a limitation of the method.
Backward diffusion is ill-posed: the forward operator destroys high-frequency
information, and no inverse can return what was not retained. The property
worth naming is that **this failure is legible**. The blow-up is visible in the
answer and predictable from the kernel, where a regularised solver returns a
plausible field and does not mention that it chose a prior.

## The boundary

Three conditions, each a real restriction:

**Periodic boundaries.** Binding is *circular* convolution, so the domain wraps.
A problem with fixed or absorbing edges needs the domain embedded in a larger
periodic one, or a different basis.

**Linearity.** The diagonalisation is a property of linear operators. A
nonlinear term has no spectrum to multiply by.

**Translation invariance.** A spatially varying coefficient — diffusivity that
depends on position — is not diagonal in this basis.

Nonlinear problems are still reachable, at the cost of the property that made
the linear case attractive. Burgers' equation and the Navier–Stokes equations
split into a linear part, which is one bind, and a nonlinear part evaluated in
the spatial domain; the split is applied over small time steps, so the
step-free, random-access-in-time property is spent. What remains is a
pseudo-spectral method, which is a good method and not a new one.

## What this says about the store

The interesting claim is not that a database can solve the heat equation. It is
that the operation the store already performs for symbolic reasons — binding a
role to a filler — is the same operation physics needs to advance a field, and
neither use was adapted to the other. Both are multiplication of spectra.

That coincidence is not a coincidence. It is why one substrate can hold a
[world model](world-models.md) and a diffusion kernel and a record with named
fields: all three are compositions in a space where composition is
multiplication, and the [cleanup](the-cleanup-loop.md) that makes symbolic
composition survive depth is the same read that makes a noisy field recoverable.

## Related

- [Solve a PDE by binding](../how-to/solve-a-pde-by-binding.md) — the procedure.
- [Do calculus with vectors](../how-to/do-calculus-with-vectors.md) —
  differentiation and integration as algebra.
- [Vector algebra](vector-algebra.md) — bind, unbind, bundle, permute.
- [Fractional power encoding](../terms/fractional-power-encoding.md) —
  continuous quantities as phases.
- [Vector endpoints](../api/vectors.md) — `/vec/bind`, `/vec/pow`,
  `/vec/rotate`.
