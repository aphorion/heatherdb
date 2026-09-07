# Encode quantities and time

Give a continuous value a direction whose similarity falls off *smoothly* with
numeric distance, instead of the step function a bucket gives you.

Two stateless routes do the work:
[`/vec/pow`](../api/vectors.md#post-vecpow) for a quantity, and
[`/vec/rotate`](../api/vectors.md#post-vecrotate) for a position.

## Fractional power encoding

Fix one random base vector. Encode the number `x` as the base raised to the
spectral power `x`. Integer powers are repeated binds; fractional powers
interpolate between them, continuously.

```python
import numpy as np

def call(method, path, body=None):
    ...  # your HTTP client; every /vec route answers {"result": [...]}

# The base is the unit of the axis. Draw it once, persist it, and use the
# same one for every write and every query on this quantity -- two bases give
# two incomparable number lines.
BASE = np.random.default_rng(20260101).standard_normal(512)
BASE = (BASE / np.linalg.norm(BASE)).tolist()

def fpe(x: float) -> list[float]:
    """Encode a real number as a direction. sim(fpe(a), fpe(b)) falls off
    smoothly as |a-b| grows -- no bucket edges, no cliff at 59 vs 61."""
    return call("POST", "/vec/pow", {"a": BASE, "t": float(x)})["result"]
```

`/vec/pow` never normalises, so normalise the result yourself before writing
it into a collection.

**Scale the input to the falloff you want.** `fpe(x)` treats one unit as one
unit; if a "close" price is within \$50, pass `price / 50`. The scale factor
is the bandwidth of the similarity kernel, and it is the only tuning this
encoding has.

## Addition becomes a single bind

Spectral powers add, which is the property that makes this worth using:

```
pow(BASE, a) ⊛ pow(BASE, b) == pow(BASE, a + b)
```

So arithmetic on quantities is one call, with no decode step in between:

```python
# "Shift this reading by +3 units" -- bind, not re-encode. Useful when the
# offset is itself a stored vector you never converted back to a number.
shifted = call("POST", "/vec/bind",
               {"a": fpe(12.0), "b": fpe(3.0), "normalize": True})["result"]
# shifted now points where fpe(15.0) points.
```

Unbinding runs it backwards: `unbind(fpe(15), fpe(3))` recovers `fpe(12)`.
Recover the number itself by scoring the result against a ladder of candidate
encodings and taking the peak.

## Buckets versus FPE

| | One-hot buckets | Fractional power |
|---|---|---|
| Similarity in `x` | step function | smooth falloff |
| Values 59 and 61 | orthogonal | near-identical |
| Values 15 and 2000 | orthogonal | near-orthogonal |
| Interpolation | none | continuous |
| Arithmetic | none | bind = addition |
| Cost | a few dims, no calls | a `/vec/pow` call per value |
| Reading it back | read the hot index | score against a ladder |

Take buckets when the boundaries are real — a pricing tier, a severity band, a
regulatory threshold — and when the encoder must stay dependency-free and
inline. Take FPE when "nearby number" should mean "nearby vector", when you
want to interpolate, or when you want to do arithmetic in vector space.

## Position: `/vec/rotate`

`/vec/rotate` is the continuous form of
[`algebra/permute`](../api/algebra.md#post-dbdbalgebrapermute): `t=0` is the
identity, `t=1` reproduces the discrete permutation exactly, and fractional
`t` dials smoothly between them.

```python
def at(vec: list[float], t: float, axis: str = "sequence") -> list[float]:
    """Place a vector at continuous position t along a named axis.
    `name` is hashed to the permutation seed, so the axis name is the axis:
    "sequence" and "depth" are independent, with no seed bookkeeping."""
    return call("POST", "/vec/rotate",
                {"a": vec, "name": axis, "t": float(t)})["result"]

# Same event, 0.5 steps apart: similar but distinguishable. Same event four
# steps apart: effectively unrelated. That decay is the position metric.
near = at(event, 3.0)
also = at(event, 3.5)
```

Rotation is an exact isometry on odd-length permutation cycles, so it changes
where a vector points without changing how long it is.

Use rotation rather than FPE when the axis is ordinal position — token index,
window offset, hop count, tree depth — and you want a *bounded* wrap rather
than an unbounded number line.

## Time of day: two right answers

```python
# (a) sin/cos pair -- two components inside a feature block.
ang = 2 * np.pi * hour / 24
block = [W_HOUR * np.sin(ang), W_HOUR * np.cos(ang)]

# (b) phase on a clock base -- a whole vector, one full turn per day.
CLOCK = BASE                      # any fixed base vector
phase = call("POST", "/vec/pow",
             {"a": CLOCK, "t": hour / 24.0})["result"]
```

| Use | When |
|---|---|
| `(sin, cos)` | time is one column among many in a concatenated feature vector, and its share of the metric is a weight you set |
| `pow(CLOCK, t)` | time is a first-class dimension you bind other things to, or you need to add durations without decoding |

Both wrap correctly: 23:00 sits adjacent to 00:00, and 06:00 opposes 18:00.
The difference is structural, not numerical. `(sin, cos)` costs two dimensions
inside a block and is trivially auditable. `pow(CLOCK, t)` costs a whole
vector and a round trip, and earns it by composing:
`bind(pow(CLOCK, t), event)` is "this event, at this time" as one atom, which
a sin/cos pair cannot express.

For a calendar, use the same shape at each period — one base per period, and
bundle:

```python
def phase_of(base: list[float], frac: float) -> list[float]:
    """One full turn of `base` per period; `frac` is the fraction elapsed."""
    return call("POST", "/vec/pow", {"a": base, "t": frac})["result"]

# Hour, weekday and month as three independent phases, bundled into one
# time vector. Independent bases mean "3 p.m." and "March" cannot interfere.
t_vec = call("POST", "/vec/bundle", {"terms": [
    {"vector": phase_of(HOUR_BASE, hour / 24.0),      "weight": 1.0},
    {"vector": phase_of(DAY_BASE, weekday / 7.0),     "weight": 0.6},
    {"vector": phase_of(MONTH_BASE, month / 12.0),    "weight": 0.3},
]})["result"]
```

The weights are the metric here as everywhere: at 1.0 / 0.6 / 0.3, agreement
on the hour counts about `(1.0/0.3)² ≈ 11×` agreement on the month.

## Related

- [Encode tabular data](encode-tabular-data.md) — the bucket route, and where
  weights come from.
- [Vector endpoints](../api/vectors.md) — `pow`, `rotate`, `bind`, `bundle`.
- [Vector algebra](../explanation/vector-algebra.md) — why binding composes.
- [Choose a dimension](choose-a-dimension.md) — FPE needs headroom; buckets
  do not.
