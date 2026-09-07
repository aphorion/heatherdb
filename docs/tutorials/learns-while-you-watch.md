# It learns while you watch

## Where you were

In [Your first memory](first-memory.md) you wrote 200 vectors, handed the
engine a query that was only 0.77 similar to the pattern it came from, and got
back one that was 0.99 similar. You also saw that 200 vectors became 51 hard
locations — the engine grew a codebook rather than filing your rows.

That leaves an obvious question hanging. If nobody trained anything, when did
the memory become good at this? Before your writes, it knew nothing. After
them, it reconstructs. There was no step in between.

## What this lesson shows

You are going to answer the same query three times — after 10 writes, after
100, and after 1000 — and watch the answer sharpen. There is no training
command in this transcript, and there is no moment at which the engine refuses
to answer because it is "not ready yet".

You need the engine you started in [lesson 1](first-memory.md#1-start-the-server)
still running, with the same `admin:tutorial-password` credentials.

## 1. A database and a shared toolkit

```bash
curl -s -u admin:tutorial-password \
  -H 'Content-Type: application/json' \
  -d '{"name": "growth", "dimension": 128}' \
  http://localhost:6380/db
```

Every lesson from here needs the same four helpers, so save them once as
`heather.py` and the later lessons will import them:

```python
import base64, json, math, random, urllib.request

BASE = "http://localhost:6380"
AUTH = base64.b64encode(b"admin:tutorial-password").decode()

def call(method, path, body=None):
    """One HTTP request against the engine. Returns the parsed JSON reply."""
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Basic " + AUTH})
    return json.load(urllib.request.urlopen(req))

def unit(v):
    """Scale to length 1 — direction is what the memory compares."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]

def cos(a, b):
    """Cosine similarity. 1.0 = same direction, 0.0 = unrelated."""
    return (sum(x * y for x, y in zip(a, b))
            / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))

def jitter(p, spread, d=128):
    """A noisy observation of pattern `p`. Bigger spread, worse copy."""
    return unit([x + random.gauss(0, spread / math.sqrt(d)) for x in p])
```

## 2. Ask the same question at three sizes

Save this as `lesson2.py`:

```python
from heather import call, unit, cos, jitter
import random, math

random.seed(7)
D = 128

# Five underlying patterns the data will be noisy copies of. The engine is
# never told they exist, how many there are, or which sample came from which.
prototypes = [unit([random.gauss(0, 1) for _ in range(D)]) for _ in range(5)]

# One fixed query, asked identically at every size, so the only thing changing
# between readings is how much the memory has seen. It is a bad copy of
# pattern 0 — spread 0.8 against the 0.35 the stored samples will use.
query = jitter(prototypes[0], 0.8)

C = "/db/growth/collections/signals"

def write_to(collection, vectors):
    """Write in pages of 100 — a single request is capped at 2 MB of JSON."""
    for i in range(0, len(vectors), 100):
        call("POST", collection + "/write", {"vectors": vectors[i:i + 100]})

def write(vectors):
    write_to(C, vectors)

written = 0
for target in (10, 100, 1000):
    # Top the collection up to the next size. Nothing is deleted or rebuilt
    # between rounds; this is the same memory, further along.
    write([jitter(prototypes[i % 5], 0.35) for i in range(written, target)])
    written = target

    result = call("POST", C + "/read", {"query": query})["result"]
    stats = call("GET", C + "/stats")

    # Two numbers, deliberately different:
    #   fidelity     — query vs its own reconstruction. You can always compute
    #                  this, on any data, without knowing the right answer.
    #   vs prototype — the reconstruction vs the pattern the query really came
    #                  from. Only computable here because we generated it.
    print("%5d writes   fidelity %.3f   vs prototype %.3f   locations %3d"
          % (written, cos(query, result), cos(result, prototypes[0]),
             stats["num_locations"]))
```

```bash
python3 lesson2.py
```
```
   10 writes   fidelity 0.232   vs prototype 0.277   locations   5
  100 writes   fidelity 0.760   vs prototype 0.962   locations  27
 1000 writes   fidelity 0.782   vs prototype 0.999   locations 254
```

## What just happened

Read the third column first. After 10 writes the engine's answer had almost
nothing to do with the pattern the query came from — 0.277 is barely better
than a random guess. After 100 it was 0.962. After 1000 it was 0.999, which is
the pattern, recovered from a query that was a poor copy of it.

Nothing in the transcript did any training. There was no fit, no epoch, no
checkpoint, no reload. The only calls were `write` and `read`. The improvement
is a side effect of writing: each write nudges the location it lands nearest,
and locations that keep receiving similar data converge on the pattern
underneath the noise. This is what [online
learning](../terms/online-learning.md) means when the store is doing it.

Three details worth stopping on.

**The engine was never unavailable.** At 10 writes it answered, badly. There is
no threshold below which the API refuses. A system that is 30% useful on day
one is a different proposition from one that is 0% useful until a training run
finishes.

**The codebook grew, but not one entry per vector.** 10 → 5, 100 → 27,
1000 → 254 — roughly a quarter of the writes, and not a number you configured.
Whether that ratio means anything is a fair question, so measure it: write the
same 1000 vectors again, drawn from 1000 unrelated patterns instead of 5.

```python
# Same volume, completely different structure: every vector its own pattern.
unrelated = [unit([random.gauss(0, 1) for _ in range(D)]) for _ in range(1000)]
write_to("/db/growth/collections/noise", unrelated)
print(call("GET", "/db/growth/collections/noise/stats")["num_locations"])
```

```
1121
```

Same 1000 writes, 258 locations when they came from five patterns and 1121 when
they came from a thousand. The codebook is sized by how much genuinely distinct
material arrived, not by how many requests you made — that decision belongs to
the [MDL gate](../how-to/tune-a-database.md#the-mdl-allocation-gate), which
allocates a new location only when an existing one cannot account for what
arrived.

**Fidelity moved with quality, but not to 1.0.** It went 0.232 → 0.760 → 0.782
and stopped. That ceiling is honest: your query is a *bad* copy, so the
reconstruction lands on the clean pattern and stays a fixed distance from the
query you actually sent. Fidelity measures agreement between question and
answer, not the engine's self-esteem — the next lessons lean on it hard.

One more thing to notice about what this replaces. There was no model to
select, no training run to schedule, no serving step, and no drift to monitor —
new data changes the store by being written to it. That is the whole operational
lifecycle of a learned system, collapsed into the write path.

## Where to go next

You have watched the memory improve. Every vector so far has been one we
generated for you; the next lesson has you encode data of your own, and shows
that the encoding decides what the memory can ever notice: [Encode something
real](encode-something-real.md).
