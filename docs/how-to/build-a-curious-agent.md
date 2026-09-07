# Build a curious agent

Make an agent that explores an unknown space on its own — no reward signal, no
map, no gradient — using the store's own familiarity number as the compass.
Why that number points anywhere is in
[Curiosity](../explanation/curiosity.md); this page is the loop.

## The loop

**Sense** the current state as a vector · **score** familiarity for every
state one action can reach · **act** toward the least familiar of them ·
**write** what is observed, which raises that state's familiarity · repeat
until the stopping rule in step 4 fires. The write is what makes the score a
gradient rather than a fixed field: without it the agent walks to one peak and
stays.

## 1. Encode states so nearby states are similar

```python
import hashlib
import numpy as np

D, M = 1024, 6                          # an M x M world, never seen before

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def unitary(name):
    # Unit magnitude in every frequency bin, so binding by it is a clean
    # invertible shift and B**x walks a coordinate without losing energy.
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    ph = np.random.default_rng(seed % (2 ** 63)).uniform(0, 1, D // 2 + 1)
    f = np.exp(2j * np.pi * ph); f[0] = f[-1] = 1.0
    return np.fft.irfft(f, n=D)

def cpow(v, p):
    # Fractional power: the coordinate encoder. Adjacent coordinates stay
    # similar and distant ones go near-orthogonal, so novelty measures
    # distance from the explored region, not "was this cell visited".
    return np.fft.irfft(np.fft.rfft(v) ** p, n=D)

def bind(a, b):
    return normalize(np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D))

B, RX, RY = unitary("g:B"), unitary("g:RX"), unitary("g:RY")
STATE = {(x, y): normalize(bind(RX, cpow(B, x)) + bind(RY, cpow(B, y)))
         for x in range(M) for y in range(M)}
```

## 2. Score familiarity from the memory itself

```python
def familiarity(s, mem):
    # The first-contact read: how close this state sits to the nearest thing
    # already written. 1.0 = been here; novelty is 1 - this. No novelty model
    # and no visit counter -- the store's own read is the signal.
    if not mem:
        return 0.0                      # empty memory: everything is novel
    return float(np.max(np.stack(mem) @ s))
```

Against a live collection this is one [`read`](../api/reads.md) per candidate
with `"strategy": "fast"`, scored by cosine against the query — the
single-step read keeps the distance the iterative one converges away. Batch
them through `batch_analyze` when branching makes round trips matter.

## 3. Act toward the least familiar reachable state

```python
def neighbors(x, y):
    return [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= x + dx < M and 0 <= y + dy < M]

def step(pos, mem, rng):
    cands = neighbors(*pos)
    scores = [familiarity(STATE[c], mem) for c in cands]
    # Break ties at random: a deterministic argmin over a finite
    # neighbourhood admits limit cycles (see below).
    lo = min(scores)
    tied = [c for c, v in zip(cands, scores) if v <= lo + 1e-9]
    return tied[int(rng.integers(len(tied)))]

def walk(steps, seed, curious=True):
    rng, pos = np.random.default_rng(seed), (M // 2, M // 2)
    mem, seen, coverage = [STATE[pos]], {pos}, []
    for _ in range(steps):
        nb = neighbors(*pos)
        pos = step(pos, mem, rng) if curious else nb[int(rng.integers(len(nb)))]
        mem.append(STATE[pos])          # the write: this state is now familiar
        seen.add(pos); coverage.append(len(seen))
    return coverage
```

## The measured comparison

A 6 × 6 grid, 36 cells, agent starting at the centre with an empty memory,
20 seeds:

| cells covered | after 20 steps | after 80 steps | after 200 steps |
|---|---:|---:|---:|
| random walk | 11.4 | 24.8 | 32.7 |
| curious walk | **19.0** | **34.2** | **35.9** |

| steps to cover | 50% | 90% |
|---|---:|---:|
| random walk | 44 | 193 |
| curious walk | **19** | **48** |

Four times faster to 90% coverage, from a scalar the store returns anyway.
Neither walk covers all 36 cells within 200 steps: the last cell of a bounded
world is expensive for any local rule, curiosity included. The tie-break in
step 3 is load-bearing — with ties resolved deterministically the same loop
plateaus at 26 of 36 cells, a limit cycle indistinguishable from a saturated
memory in the coverage curve alone.

## 4. Stop, rather than keep exploring

Novelty says where to go, never when to stop. Storing a state as an exception
costs `b` bits once and removes its residual surprise from every later
encounter, so a state is worth visiting while

```
surprise(s) × recurrence(s) > b
```
```python
B_BITS = np.log2(240) + np.log2(32)     # address bits + value resolution

def worth_exploring(fam, expected_visits):
    # surprise = -log2(familiarity): a state the memory predicts costs nothing
    # to leave unstored. The recurrence factor is what a flat novelty
    # threshold cannot express -- it chases rare states that never pay back
    # their bits and skips frequent mild ones that bleed surprise forever.
    return -np.log2(max(fam, 2.0 ** -8)) * expected_visits > B_BITS

# Stop when no reachable state clears the bar -- abstention as the terminal
# condition, which a flat novelty threshold cannot produce.
frontier = [c for c in cands
            if worth_exploring(familiarity(STATE[c], mem), n_visits[c])]
```

Measured over a synthetic encounter stream, this rule reaches 484 bits of
total description length against 819 for the best of six fixed novelty
thresholds, and holds when the visitation distribution shifts (416 bits,
nothing retuned, against 819). The frontier shrinks as it is consumed: 21
states at the start, zero at the end.

## 5. Combine curiosity with a goal

An agent that only explores never does anything. Put both drives in one scalar
field and let the larger term win:

```python
def value(x, need, known_goals, visits):
    # PRAGMATIC: closeness to something known to satisfy the preference. Zero
    # until one is found -- an agent cannot want what it has not mapped.
    prox = max((1.0 / (1 + abs(x - g)) for g in known_goals), default=0.0)
    nov = 1.0 / (1 + visits[x])        # EPISTEMIC: visit-decayed novelty
    # need = how far the state sits below the preferred one. Full: novelty
    # dominates and it explores. Hungry: proximity dominates and it exploits.
    return need * prox + CURIOSITY * nov
```

Measured on a 30-position world, four hidden food sources, energy draining one
unit per step, preferred energy 12, over 12 seeds of 800 steps:

| | mean energy | min energy | starvations / 800 steps |
|---|---:|---:|---:|
| one-field agent | **12.3** | **1.9** | **3.5** |
| random walker | 10.0 | 1.0 | 35.3 |

It spends 64% of its steps exploring and the rest foraging, and is never told
to seek food — it holds a preferred state and acts to close the gap. Raising
`CURIOSITY` buys map coverage with starvations and lowering it buys the
reverse; that trade is this loop's one parameter, reported as a pair.

## Failure modes to watch

- **Flat field, agent stalls.** Either the world is mapped or the collection
  is [saturated](../explanation/capacity-and-interference.md); read the load,
  not the signal — they look identical from inside.
- **Agent orbits two states.** Deterministic tie-breaking: break ties at
  random, or add the recurrence term from step 4.
- **Novelty never drops.** The write is not landing, or every state is
  near-orthogonal to every other, so visiting one teaches nothing about its
  neighbours. Check that adjacent states read a raised familiarity.
- **Exploration never ends.** No stopping rule — a fixed novelty threshold
  cannot supply one; the bit-accounted rule can.

## Load the map, do not write it

Familiarity only works if each place is its own entry. The competitive write
path merges near-identical patterns on purpose — that is what makes it
generalise — and merging is exactly wrong here: four distinct-but-similar
squares written normally collapse into two entries, after which a visited
square scores 0.49 against an unvisited one at 0.48, and the agent is walking
blind. The same four loaded with `bulk_load` stay four, and score 1.00 against
0.50.

```python
# Each place must remain its own attractor, so place them directly rather than
# letting competitive learning decide they are the same place.
call("POST", C + "/bulk_load", {"addresses": seen, "counters": seen})
```

An agent built on `write` measures *worse* than a random walk, which is a
confusing result to debug from the outside.


## Related

- [Curiosity](../explanation/curiosity.md) — why the abstention signal
  explores · [Build a world model](build-a-world-model.md) — predicting where
  an action leads, so the agent can plan toward a novel state
- [Abstention](../explanation/abstention.md) ·
  [Calibrate a gate](calibrate-a-gate.md) ·
  [Build an agent memory](build-an-agent-memory.md) ·
  [Store a procedure](store-a-procedure.md) ·
  [fractional power encoding](../terms/fractional-power-encoding.md) ·
  [fidelity](../terms/fidelity.md)
