# Forecast a time series

Write history as windows, then read the next window back by pattern completion:
put the part you know into the query, leave the part you want predicted blank,
and the memory fills it in. No training step, no forecaster object.

The encoding is the whole design. Use the **partition** scheme from
[Encoding reference](../reference/encodings.md#contexttarget-concatenation):
half the dimension holds the context window, half what followed it. That
sliding window, and the continuous encoders for the time axis, are in [Encode
quantities and time](../how-to/encode-quantities-and-time.md).

## The pair vector

```python
import numpy as np

W = 32                 # context length = target length; the database's d = 2W

def pair(series, t):
    """One example: the W samples before t, then the W after it. Both halves
    are deviations from series[t-1], the last known value -- that anchor holds
    the level, so the vector carries only SHAPE, which is what cosine sees.
    Unanchored, the level dominates the norm and every window at a similar
    level looks alike."""
    anchor = series[t - 1]
    return np.concatenate([series[t - W:t] - anchor,
                           series[t:t + W] - anchor]), anchor

def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)
```

Write one vector per origin. `bulk_load` stores windows as they are; `write`
merges similar ones competitively — both are measured below.

```python
V = [unit(pair(series, t)[0]) for t in train_origins]
call("POST", "/db/fc/collections/vol/bulk_load",
     {"addresses": [list(v) for v in V], "counters": [list(v) for v in V]})
```

## Forecast by completion

```python
def forecast(coll, series, t, strategy="fast"):
    """Submit the known half; the read returns a whole stored pair."""
    anchor = series[t - 1]
    q_raw = np.concatenate([series[t - W:t] - anchor, np.zeros(W)])
    scale = np.linalg.norm(q_raw)
    q = q_raw / scale                          # the engine sees unit vectors
    r = np.asarray(call("POST", f"/db/fc/collections/{coll}/read",
                        {"query": list(q), "strategy": strategy})["result"])

    fidelity = float(np.dot(unit(q), unit(r)))  # how far the read had to travel
    rc, rt = r[:W], r[W:]
    # Amplitude is not in a cosine. Recover it from the half already known --
    # least squares of the returned context onto the submitted one -- and apply
    # the same factor to the returned target half.
    s = float(np.dot(q[:W], rc) / (np.dot(rc, rc) + 1e-12))
    return anchor + s * rt * scale, fidelity
```

## The nulls that make a number mean something

Four baselines, all cheap; a forecast score is uninterpretable without them
([Find your metric's floor](find-your-metrics-floor.md), [null
model](../terms/null-model.md)).

| Baseline | What it is | What it kills |
|---|---|---|
| **Persistence** | the next window equals the last known value | the illusion that a smooth series is being predicted |
| **Training mean** | predict the unconditional mean at every horizon | the illusion that beating persistence is skill |
| **Shuffled pairing** | store each context with the future of a *different* origin | the association itself, keeping both marginals |
| **Phase randomisation** | resample the series with the same power spectrum, random phases | the specific dynamics, keeping autocorrelation |

```python
# Shuffled pairing: contexts and futures are both real; only the correspondence
# between them dies. Run it many times for a distribution, not a single value.
perm = rng.permutation(len(train_origins))
V_null = [unit(np.concatenate([series[t - W:t] - series[t - 1],
                               series[t2:t2 + W] - series[t2 - 1]]))
          for t, t2 in zip(train_origins, [train_origins[i] for i in perm])]
```

## Measured: real series, 2 310 days

A 396-asset universe reduced to two daily channels — market log return and
20-day log realised volatility, both z-scored. Windows of 32; 1 553 training
and 661 test origins, split chronologically so no test window's future is in
the store. Errors are mean absolute, in z-units.

**Volatility, one channel.** Persistence is extremely strong on a slow series:

| horizon | memory | persistence | mean | skill vs persistence | corr(pred, actual) |
|---|---|---|---|---|---|
| 1 | 0.090 | 0.087 | 0.538 | −0.025 | **+0.967** |
| 4 | 0.245 | 0.239 | 0.542 | −0.025 | +0.850 |
| 8 | 0.387 | 0.386 | 0.546 | −0.003 | +0.699 |
| 32 | 0.624 | 0.624 | 0.549 | −0.001 | +0.249 |

Correlation `+0.967` is not skill. The shuffled-pairing null over 20 draws puts
the same h = 4 error at `0.241 ± 0.002` against the observed `0.245`: the
memory does not beat a store whose futures were assigned at random, because
what both reproduce is "the level does not move much". A phase-randomised
surrogate with no dynamics at all still scores `corr = +0.647 ± 0.063` through
the identical pipeline — the floor of correlation on an autocorrelated series.

**Returns, one channel.** The reverse trap. Memory error runs 0.673–0.696
against persistence's 0.921–0.982 — a 24–30% improvement — with correlation
against the future at zero for every horizon (−0.048, +0.010, −0.042 at h = 1,
4, 16) and the unconditional mean beating the memory at 0.650. Persistence is
the wrong baseline for a noisy series: it carries yesterday's shock forward.

## Multi-channel windows

Channels are disjoint slices of one vector — both contexts, then both futures,
so one query zeroes both target slices. `d = 4W = 128`.

```python
def pair2(t):
    av, ar = vol[t - 1], ret[t - 1]         # one anchor per channel
    return np.concatenate([vol[t - W:t] - av, ret[t - W:t] - ar,     # context
                           vol[t:t + W] - av, ret[t:t + W] - ar])    # target
```

Forecasting volatility from the joint window, against its own shuffled-pairing
null (12 draws):

| horizon | joint | single channel | persistence | shuffled-pairing null |
|---|---|---|---|---|
| 4 | **0.224** | 0.245 | 0.239 | 0.245 ± 0.002 |
| 8 | **0.347** | 0.387 | 0.386 | 0.394 ± 0.004 |
| 16 | **0.497** | 0.547 | 0.573 | 0.582 ± 0.006 |

The joint window clears its null by 5–15 null standard deviations where the
single channel ties it: the return channel carries information about future
volatility that the volatility channel does not, and the slice layout lets one
read use both.

## Fidelity says when a forecast is out of scope

[Fidelity](../terms/fidelity.md) — the cosine between the query and the state
the memory settled into — is a coverage statistic, not an error estimate. Mean
fidelity across the runs above: **0.902** on volatility windows, **0.762** on
the phase-randomised surrogate, **0.656** on return windows.

Between series it separates cleanly and agrees with the skill tables. *Within*
a well-covered series it ranks nothing — `corr(fidelity, |h = 4 error|) =
+0.061`, mean error rising 0.205 → 0.271 from the lowest fidelity quartile to
the highest. Gate on it for "is this window like anything in history", not "how
wrong is this number"; [Calibrate a gate](calibrate-a-gate.md) sets the cut.

## Regime change is fidelity drifting down

A store written on one regime reports its own obsolescence. A series of period
40 for 900 steps and period 11 thereafter, with the memory written on the first
regime only:

| origins | regime | fidelity | \|h = 4 error\| |
|---|---|---|---|
| 700–859 | A (written) | 0.707 | 0.09 |
| 860–939 | crossing | 0.596 | 0.91 |
| 940–1499 | B (unseen) | 0.395 | 1.60 |

The shift is persistent rather than a spike, which is what distinguishes a
regime change from one odd window. Track fidelity as a rolling mean over a
block of origins, against its distribution on held-out data from the written
regime. On the real volatility series it holds between 0.876 and 0.941 across
six 120-day blocks — no regime change, correctly.

## Storage mode and read strategy

The same 1 553 anchored volatility windows: `bulk_load` + `fast` gives MAE
0.245 at h = 4 over 1 553 locations at fidelity 0.902; `write` + `fast` gives
0.238 over 515 locations at fidelity 0.782 — a third of the store at the same
error, reading lower fidelity by construction because a merged store answers
with a prototype. `iterative` costs error only in the merged store (0.270
against 0.238). Correlation is `+0.967` in all four combinations, so amplitude
error is the statistic that separates them.

## Boundary conditions

- **A horizon-averaged win is not a per-origin win.** A log-linear rate
  forecaster on epidemic case counts scores MAE 0.258 against persistence's
  0.435 at h = 7 (1.69×), 0.700 against 0.764 at h = 14, and loses at h = 3
  (0.228 against 0.205); on one mid-explosion origin its 14-day projection
  overshoots by 378% where persistence errs by 61%. Publish the distribution
  over origins and horizons, not the mean.
- **Chronological splits only.** Neighbouring windows overlap by `W − 1`
  samples, so a randomly split test origin is a near-duplicate of a training
  one ([held-out evaluation](../terms/held-out-evaluation.md)).
- **The anchor is part of the encoding.** Removing it drops h = 1 correlation
  from 0.967 to 0.851 and leaves the level wrong: cosine cannot see magnitude.
- **The store is only a [world model](../terms/world-model.md) of the regimes
  it holds.** Re-measure against the nulls after it grows or the series turns.

## Related

- [Encode quantities and time](../how-to/encode-quantities-and-time.md) ·
  [Encoding reference](../reference/encodings.md) — window/partition schemes.
- [Find your metric's floor](../how-to/find-your-metrics-floor.md) — the null
  discipline this page runs on · [Calibrate a gate](calibrate-a-gate.md).
- [Model a dynamical system](model-a-dynamical-system.md) ·
  [Build a world model](build-a-world-model.md) ·
  [Build an anomaly detector](build-an-anomaly-detector.md)
- [Fidelity](../terms/fidelity.md) · [Null model](../terms/null-model.md) ·
  [World model](../terms/world-model.md) ·
  [Held-out evaluation](../terms/held-out-evaluation.md)
