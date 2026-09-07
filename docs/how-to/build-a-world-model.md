# Build a world model

Store `(state, action) → next state` as bindings and the collection predicts.
This page builds one, rolls it forward, and measures how far it can be trusted.
The concepts are in [World models](../explanation/world-models.md).

## What you need

A database whose dimension you have already chosen, and a world you can sample
transitions from — a simulator, a log of state changes, an event stream. The
worked example below is a 9×9 grid with four actions, small enough to check
against ground truth.

```python
import json, urllib.request
import numpy as np

BASE, D = "http://localhost:6380/db/world", 1024

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"null")

def nrm(v):                      # every vector the engine sees is unit length
    return v / (np.linalg.norm(v) + 1e-12)

def bind(a, b):                  # circular convolution, matching heather_algebra
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

def unbind(c, a):                # convolution by the approximate inverse
    return np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(a)), n=len(c))
```

## 1. Encode states and actions

States are vectors; actions are roles. The two play different parts — a state is
a thing the model can return, a role is a key you unbind with — so keep them in
separate pools.

```python
rng = np.random.default_rng(0)
N_STATES, ACTS = 81, ["N", "S", "E", "W"]

# Independent random codes make every state maximally distinguishable and
# maximally unrelated. That is the right default when states have no metric
# you trust; it is also what forfeits generalization to unvisited states —
# see the encoder section at the bottom.
V = np.stack([nrm(rng.standard_normal(D)) for _ in range(N_STATES)])
role = {a: nrm(rng.standard_normal(D)) for a in ACTS}
```

## 2. Write transitions

Accumulate one binding per experienced transition into the acting state's map.
Nothing here is a training step: the map improves on every write, in place.

```python
M = np.zeros((N_STATES, D))

def observe(s, a, s_next):
    # The whole learning rule. Superposing the new binding leaves the ones
    # already in the counter intact, so repeated experience of the same
    # transition simply weights it more heavily.
    M[s] += bind(role[a], V[s_next])

s = int(rng.integers(N_STATES))
for _ in range(4000):                     # wander; write what you feel
    a = ACTS[int(rng.integers(4))]
    s2 = step(s, a)                       # your world
    observe(s, a, s2)
    s = s2
```

Load the maps as one collection whose address is the state and whose counter is
that state's transition function, and the state codes as a second collection
that serves as the cleanup codebook.

```python
# `dynamics`: address = state code, counter = the state's transition bundle.
call("POST", "/collections/dynamics/bulk_load",
     {"addresses": [v.tolist() for v in V],
      "counters":  [nrm(m).tolist() for m in M]})

# `states`: the codebook a prediction is snapped onto. Address and counter are
# the same vector, so a read against it returns a stored state and nothing else.
call("POST", "/collections/states/bulk_load",
     {"addresses": [v.tolist() for v in V],
      "counters":  [v.tolist() for v in V]})
```

## 3. Predict one step

Read the map, unbind the action, clean the residual up against the codebook.

```python
def predict(state_vec, a):
    # Read the acting state's transition function out of `dynamics`. The read
    # is itself associative, so a slightly-off state still lands on its map.
    m = call("POST", "/collections/dynamics/read",
             {"query": nrm(state_vec).tolist(), "strategy": "iterative"})["result"]
    # Unbind leaves the successor plus interference from the other actions.
    q = nrm(unbind(np.asarray(m), role[a]))
    # Cleanup: snap that residual onto a stored state. This is the step that
    # turns a point in space back into a symbol the next step can address.
    out = call("POST", "/collections/states/read",
               {"query": q.tolist(), "strategy": "iterative"})["result"]
    return nrm(np.asarray(out))
```

To read the *distribution* over successors instead of committing to one — useful
when a transition is stochastic — swap the second read for `analyze` with
`strategy: "fast"` and keep its `activated_locations` weights. See
[Read endpoints](../api/reads.md).

## 4. Roll forward n steps

A rollout is the same call in a loop, with the cleanup between iterations.

```python
def rollout(s0, actions):
    x = V[s0]
    path = []
    for a in actions:
        x = predict(x, a)        # predict -> cleanup -> predict
        path.append(int(np.argmax(V @ x)))   # which stored state it landed on
    return path
```

## 5. Measure decay against a baseline

A rollout number means nothing on its own. Score accuracy at each horizon
against a baseline that uses no model at all — **the state does not change** —
so you can see what the model contributes. This is the
[metric floor](find-your-metrics-floor.md) discipline applied to dynamics.

```python
def score(H, S=400, seed=1):
    r = np.random.default_rng(seed)
    starts = r.integers(0, N_STATES, size=S)
    seqs   = r.integers(0, 4, size=(S, H))
    true, pred = starts.copy(), starts.copy()
    for h in range(H):
        a = seqs[:, h]
        true = np.array([step(t, ACTS[i]) for t, i in zip(true, a)])
        # Local mirror of the engine read: unbind, then snap to the codebook.
        q = np.stack([nrm(unbind(nrm(M[p]), role[ACTS[i]]))
                      for p, i in zip(pred, a)])
        pred = np.argmax(q @ V.T, axis=1)
    # Two numbers, always reported together: the model, and doing nothing.
    return (pred == true).mean(), (starts == true).mean()
```

Measured on the 9×9 grid at `D = 1024`, 400 rollouts per point, for three
wandering budgets:

| Horizon | 135/324 written | 247/324 written | 324/324 written | Stay-put |
|---|---|---|---|---|
| 1 | 46% | 76% | 100% | 14% |
| 2 | 31% | 66% | 100% | 27% |
| 4 | 11% | 49% | 99% | 14% |
| 8 | 5% | 34% | 99% | 8% |
| 16 | 1% | 24% | 96% | 4% |
| 32 | 1% | 16% | 94% | 5% |

Read the table as a contract. Decay is multiplicative — a rollout survives only
if every step does — so the usable horizon is set by the per-step error rate and
falls fast when coverage is partial. At H = 2 with a third of the world written,
the model reads 31% against a 27% baseline: nearly nothing. Publish the horizon
at which your model crosses its baseline, and re-measure it when coverage
changes.

## Do not skip the cleanup

Feeding the raw unbind residual into the next step instead of a cleaned-up state
walls the rollout at depth two to four. Both arms below address identical maps
and differ only in what they carry forward:

| Horizon | Snap between steps | Raw residual, β = 5 | Raw residual, β = 15 |
|---|---|---|---|
| 1 | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% |
| 3 | 100% | 71% | 100% |
| 4 | 100% | 17% | 100% |
| 8 | 100% | 3% | 100% |
| 32 | 100% | 2% | 100% |

Residual error at step `h` becomes the input error of step `h+1`, which the next
unbind amplifies; the snap resets that floor every step. A sufficiently sharp
read is a cleanup — `beta = 15` on this codebook recovers the whole horizon,
`beta = 5` does not. [The cleanup loop](../explanation/the-cleanup-loop.md) is
the same wall in composition rather than in time.

## If you need unvisited states

Per-state maps memorise: a state never written has no map, and the model scores
at chance on it. Encoding states so that neighbours share structure makes each
action a single operator `s' = bind(T_a, s)` that holds everywhere — measured at
100% on unseen states from two training states, against 1% for random codes.
That path, and the boundary where actions do not commute, is in
[World models](../explanation/world-models.md).

## Related

- [World models](../explanation/world-models.md) — why a transition function is
  a bundle of bindings.
- [Find your metric's floor](find-your-metrics-floor.md) · [The cleanup
  loop](../explanation/the-cleanup-loop.md)
- [Encode quantities and time](encode-quantities-and-time.md) — structured codes
  for states with a metric.
- [Write endpoints](../api/writes.md) · [Read endpoints](../api/reads.md)
