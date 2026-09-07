# Do calculus with vectors

Differentiate and integrate a function without a difference quotient, a
quadrature rule, or a step size — by binding it with one operator vector, and
unbinding it with the same one.

Under [fractional power encoding](../terms/fractional-power-encoding.md) a
quantity is a *phase*: `enc(x) = base^⊗x` gives every frequency bin the angle
`ω_j·x`. Phases add under multiplication, which is what
[bind](../terms/bind.md) does in the frequency domain, so the arithmetic of
quantities and the calculus of functions are the same two calls —
[`/vec/bind`](../api/vectors.md#post-vecbind) and
[`/vec/unbind`](../api/vectors.md#post-vecunbind).

## Setup

```python
import numpy as np
import requests

ENGINE = "http://localhost:6393"
S = requests.Session()

def call(path, body):
    r = S.post(ENGINE + path, json=body, timeout=120)
    r.raise_for_status()
    return np.asarray(r.json()["result"])

def bind(a, b):
    # normalize=False throughout: in this regime the spectrum's magnitude
    # is payload (a decay, a growth, an operator's gain), not an artefact
    # to be scaled away.
    return call("/vec/bind", {"a": list(map(float, a)),
                              "b": list(map(float, b)),
                              "normalize": False})

def unbind(c, key, exact=False, eps=1e-9):
    # exact=True switches from circular correlation to spectral division --
    # the true inverse when the key's spectrum is not unit magnitude, which
    # is the case for every operator vector below.
    return call("/vec/unbind", {"a": list(map(float, c)),
                                "b": list(map(float, key)),
                                "exact": exact, "eps": eps})

def vpow(a, t):
    return call("/vec/pow", {"a": list(map(float, a)), "t": float(t)})

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```

The base is built from an explicit set of frequencies, because the frequencies
are what the operator vector is later written against.

```python
D = 4096
rng = np.random.default_rng(7)

# One random frequency per interior bin. DC and Nyquist are pinned to 1:
# they are the bins with no phase to rotate, and leaving them free makes
# negative exponents blow up.
omega = rng.uniform(-np.pi, np.pi, D // 2 - 1)
spec = np.ones(D // 2 + 1, dtype=complex)
spec[1:D // 2] = np.exp(1j * omega)
BASE = np.fft.irfft(spec, n=D)

enc = lambda x: vpow(BASE, float(x))    # phi(x) = BASE^x
```

`|ω| ≤ π` is a hard cap, not a taste: `/vec/pow` exponentiates each bin on the
principal branch, so a frequency outside `(−π, π]` wraps and silently aliases.
That is the carrier's Nyquist limit expressed in generator space.

## Arithmetic is exact

Binding two encoded numbers adds them, because it adds their phases.

```python
# phi(2.3) (x) phi(1.4) should point exactly where phi(3.7) points.
print(1 - cos(bind(enc(2.3), enc(1.4)), enc(3.7)))     # addition
print(1 - cos(unbind(enc(3.7), enc(1.4)), enc(2.3)))   # subtraction

# Multiplication is addition in the log domain -- the same two calls,
# with ln(x) carried instead of x.
a, b = 6.0, 2.5
print(1 - cos(bind(enc(np.log(a)), enc(np.log(b))), enc(np.log(a * b))))
print(1 - cos(unbind(enc(np.log(a)), enc(np.log(b))), enc(np.log(a / b))))
```

Measured, on the engine:

| Operation | `1 − cos` against the encoded answer |
|---|---|
| `φ(2.3) ⊛ φ(1.4)` vs `φ(3.7)` | `0.0` |
| `φ(3.7) ⊘ φ(1.4)` vs `φ(2.3)` | `0.0` |
| `φ(ln 6) ⊛ φ(ln 2.5)` vs `φ(ln 15)` | `−2.2e-16` |
| `φ(ln 6) ⊘ φ(ln 2.5)` vs `φ(ln 2.4)` | `2.2e-16` |

This is **exact**, at float64. Nothing here is a nearest-neighbour match or a
cleanup: the identity `base^a ⊛ base^b = base^(a+b)` holds bin by bin. Note
that subtraction uses the *default* unbind — the base has unit spectrum, and
circular correlation is the exact inverse for unit-spectrum keys.

The encoding's similarity kernel is what makes it a *number line* rather than a
set of labels. On this base:

```
sim(φ(0), φ(0.0)) = 1.0000        sim(φ(0), φ(1.0)) = 0.0037
sim(φ(0), φ(0.1)) = 0.9838        sim(φ(0), φ(3.0)) = 0.0031
sim(φ(0), φ(0.5)) = 0.6396
```

Nearby values are near-identical, far values near-orthogonal, and the crossover
is set by the base's bandwidth — scale your inputs to put the falloff where you
want it, as in
[Encode quantities and time](encode-quantities-and-time.md).

## A function is a vector

Superpose the encodings of the sample points, weighted by the function's value
there. This is a Riemann sum in vector space: `F = Σ f(x_k) φ(x_k) Δx`.

```python
xs = np.linspace(-8, 8, 241)
dx = xs[1] - xs[0]
f = np.exp(-xs**2 / 8) * np.cos(1.2 * xs)   # band-limited: fits inside (-pi, pi)

phis = np.array([enc(x) for x in xs])       # one /vec/pow per grid point
F = (f[:, None] * phis).sum(axis=0) * dx    # the function, as one vector

# Reading a function vector back is an inner product against the same grid:
# <phi(x_k), F> peaks where f is large.
decode = lambda V: phis @ V
```

## Differentiation is a bind

Integration by parts gives `transform(f') = −iω · transform(f)`. So the
derivative operator has a symbol, and therefore a vector.

```python
# The d/dx operator vector: spectrum -i*omega on the interior bins.
# DC and Nyquist are ZERO, not one -- they are the operator's null space.
# The zeroed DC bin is precisely the constant of integration.
ospec = np.zeros(D // 2 + 1, dtype=complex)
ospec[1:D // 2] = -1j * omega
S_OP = np.fft.irfft(ospec, n=D)

dF = bind(F, S_OP)                          # differentiate: one call
iF = unbind(F, S_OP, exact=True, eps=1e-6)  # integrate: one call
```

Integration is the *same operator, inverted* — spectral division by `−iω` —
which is why it needs `exact: true`. `eps` is the threshold below which a bin
counts as null and is zeroed rather than divided by; without it, the near-null
bins divide by almost nothing and dominate the answer.

## Check it against the closed form

```python
# The exact derivative of f, differentiated by hand.
fp = np.exp(-xs**2 / 8) * (-xs / 4 * np.cos(1.2 * xs) - 1.2 * np.sin(1.2 * xs))
# A fine-grid cumulative integral, mean-removed (the constant is not recoverable).
Fi = np.cumsum(f) * dx
Fi -= Fi.mean()

def report(name, got, ref):
    # The decode carries an arbitrary overall scale (the kernel's gain), so
    # compare shape: correlation, and relative L2 after a least-squares scale.
    s = (got @ ref) / (got @ got)
    rel = np.linalg.norm(s * got - ref) / np.linalg.norm(ref)
    print(f"{name:10s} corr = {np.corrcoef(got, ref)[0, 1]:.4f}   "
          f"rel-L2 = {rel:.3e}")

report("f",        decode(F),  f)
report("df/dx",    decode(dF), fp)
integral = decode(iF); integral -= integral.mean()
report("integral", integral,   Fi)
```

Measured:

```
f          corr = 0.9994   rel-L2 = 3.471e-02
df/dx      corr = 0.9997   rel-L2 = 2.572e-02
integral   corr = 0.9987   rel-L2 = 5.115e-02
```

Note that `f` itself — the round trip through encode and decode with *no*
operator applied — carries `3.5e-2` of the error. The calculus is not what
loses precision here.

## Where it is exact, and where it is not

The two halves have completely different error behaviour, and separating them
is the whole story.

**The algebra is exact.** The engine's bind against a plain spectral multiply
of the same two vectors:

```python
ref = np.fft.irfft(np.fft.rfft(F) * np.fft.rfft(S_OP), n=D)
print(np.linalg.norm(dF - ref) / np.linalg.norm(ref))     # 8.05e-16
```

`8.05e-16` relative — float64 round-off. Differentiation as an operation on
function vectors introduces no error at all.

**The inverse is exact up to the null space.** Integrating and then
differentiating returns the original vector, minus the bins the operator
annihilates:

```python
round_trip = bind(iF, S_OP)
print(np.linalg.norm(round_trip - F) / np.linalg.norm(F))          # 4.612e-03

# ...and that residual is exactly F's DC + Nyquist content:
sp = np.fft.rfft(F)
null = np.zeros_like(sp); null[0] = sp[0]; null[D // 2] = sp[D // 2]
print(np.linalg.norm(np.fft.irfft(null, n=D)) / np.linalg.norm(F))  # 4.612e-03
```

The two numbers agree to every printed digit. `d/dx ∘ ∫` is the identity on
everything except the constant — which is the correct statement of the
fundamental theorem, recovered as a measurement rather than assumed.

**The encoding is approximate**, and it is the only approximate part:

| Source of error | Controlled by |
|---|---|
| Finite grid — `F` is a Riemann sum, not an integral | grid spacing `Δx` |
| Kernel width — the decode reads `f` through the similarity kernel, which smooths it | the base's bandwidth, and `D` |
| Bandwidth cap — content beyond `\|ω\| = π` aliases with no error reported | scale `x` so the function's spectrum fits |
| Superposition crosstalk — every sample interferes with every other | `D` (capacity grows with dimension) |

Raising `D` and refining the grid moves the `1e-2` figures down; nothing moves
the `8e-16` figures, because they are already at the floor.

## Dynamics: the same operator, exponentiated

An operator vector's spectral *power* is the flow it generates. Where
`bind(F, S_OP)` applies `d/dx` once, `pow` of an evolution generator applies its
flow for a continuous time — which is
[Solve a PDE by binding](solve-a-pde-by-binding.md). The damped oscillator
`x'' + 2γx' + ω₀²x = 0` propagated by 25 binds with the clock
`e^{(−γ+iω_d)Δt}` at `Δt = 0.4` holds `8.29e-16` max error where explicit Euler
at the same step reaches `59.2`; one `pow(clock, 25)` reproduces the whole
trajectory's endpoint to `6.31e-16`.

## Related

- [Solve a PDE by binding](solve-a-pde-by-binding.md) — the same operators,
  raised to a continuous power.
- [Physics in the frequency domain](../explanation/physics-in-the-frequency-domain.md)
  — why an operator has a vector at all.
- [Encode quantities and time](encode-quantities-and-time.md) — building and
  scaling the base.
- [Vector endpoints](../api/vectors.md) — `bind`, `unbind` (`exact`, `eps`),
  `pow`.
- [Vector algebra](../explanation/vector-algebra.md) ·
  [bind](../terms/bind.md) ·
  [fractional power encoding](../terms/fractional-power-encoding.md).
