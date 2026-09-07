# It doesn't forget

## Where you were

Four lessons in, you have a memory that reconstructs corrupted queries, sharpens
as data arrives, holds structured records in single vectors, and tells you when
it does not recognise something.

Every one of those results came from a store that only ever learned one thing.
Which raises the question that decides whether any of it survives contact with
production: what happens when you write something *else*?

For a trained model the answer is well known and unpleasant. Train a network on
task A, then train it on task B, and performance on A collapses — the weights
that encoded A are the same weights B is now using. This is [catastrophic
forgetting](../terms/catastrophic-forgetting.md), and it is why most deployed
models are frozen at ship time and retrained wholesale later.

## What this lesson shows

You will write a body of data, record how well a query is answered, then write
two more unrelated bodies of data — five times as much as the first — and ask
the identical question again.

## 1. Learn one thing

```bash
curl -s -u admin:tutorial-password \
  -H 'Content-Type: application/json' \
  -d '{"name": "memory", "dimension": 128}' \
  http://localhost:6380/db
```

Save as `lesson5.py`:

```python
from heather import call, unit, cos, jitter
import random, math

random.seed(23)
D = 128

def symbol():
    return unit([random.gauss(0, 1) for _ in range(D)])

C = "/db/memory/collections/knowledge"

def write(vectors):
    for i in range(0, len(vectors), 100):
        call("POST", C + "/write", {"vectors": vectors[i:i + 100]})

def probe(label, query, pattern):
    """Ask the same question and report two things: how confident the memory
    is (fidelity), and whether it is actually right (vs the pattern we know
    the query came from)."""
    r = call("POST", C + "/read", {"query": query})["result"]
    stats = call("GET", C + "/stats")
    print("%-28s fidelity %.3f   vs pattern %.3f   locations %3d"
          % (label, cos(query, r), cos(r, pattern), stats["num_locations"]))

# The first body of knowledge: four patterns, 400 observations.
first = [symbol() for _ in range(4)]
write([jitter(first[i % 4], 0.3) for i in range(400)])

# The question we will keep asking, unchanged, for the rest of the lesson.
q = jitter(first[0], 0.7)
probe("after learning task A", q, first[0])
```

```
after learning task A        fidelity 0.793   vs pattern 0.999   locations  99
```

A near-perfect answer, as lesson 2 would predict.

## 2. Then learn something completely different

```python
# Four new patterns with no relationship to the first four — in 128 dimensions
# random directions are effectively perpendicular, so this is a different
# subject entirely, not a variation on the old one.
second = [symbol() for _ in range(4)]
write([jitter(second[i % 4], 0.3) for i in range(400)])

# Same query. Not re-issued from scratch, not re-encoded — the identical
# vector from before.
probe("after learning task B", q, first[0])

# And more still: six further patterns, 1200 observations. The collection now
# holds five times what it held when it answered the first probe.
third = [symbol() for _ in range(6)]
write([jitter(third[i % 6], 0.3) for i in range(1200)])
probe("after 1200 more, task C", q, first[0])
```

```bash
python3 lesson5.py
```
```
after learning task A        fidelity 0.793   vs pattern 0.999   locations  99
after learning task B        fidelity 0.795   vs pattern 0.998   locations 199
after 1200 more, task C      fidelity 0.791   vs pattern 0.992   locations 494
```

## What just happened

The answer to the first question went 0.999 → 0.998 → 0.992 while 1600 vectors
about unrelated subjects were written on top of it. That is the moment: a
fine-tune would have destroyed it, and here it moved by seven thousandths.

The mechanism is in the third column. Locations went 99 → 199 → 494. New
material did not overwrite the capacity holding the old material; it caused new
capacity to be allocated, alongside. Patterns that never came up again were
never disturbed, because nothing wrote near them — which is what "learning is
local" buys you, and what a network's shared weights cannot offer.

Two honest edges.

**Nothing is free.** The store grew — roughly proportionally with genuinely new
material. This trades disk for retention, which is a trade most systems would
take, but it is a trade. Watch `num_locations` on collections that ingest
forever, and see [Tune a database](../how-to/tune-a-database.md) for the
allocation gate that decides when a new location is justified.

**Interference is not zero.** 0.999 → 0.992 is a real decline, just a
negligible one at this scale. Data that genuinely *overlaps* older data will
move it, because that is learning, not forgetting. The distinction matters: the
memory updates where the world updated, and leaves alone what nobody
contradicted.

## What you know now

Five lessons, five results, and they compose into one claim:

1. A read is a **reconstruction that reports its own confidence**.
2. A write is a **learning step** — there is no training phase, and no moment
   of not-ready.
3. Structure **composes without a schema**, and comes back apart on request.
4. Fidelity is a **judgement**, so detection and abstention need no second model.
5. New knowledge **does not destroy old knowledge**, so the system can keep
   learning in production.

The sentence those add up to is the one worth carrying out of this track: this
is not a search index, it is a memory, and writing to it is the training.

## Clean up

You are done with the track. Stop the server with `Ctrl-C` — it shuts down
gracefully and flushes its audit buffer — then:

```bash
rm -rf /tmp/heather-tutorial
```

That removes all five databases: `tutorial`, `growth`, `records`, `watch` and
`memory`.

## Where to go next

The last lesson uses none of our data. You bring a problem, answer four
questions, and build: [Design your own](design-your-own.md).

- Put it somewhere real: [Run it in production](../how-to/production.md).
- Use the structure lesson in anger: [Store and query structured
  documents](../how-to/structured-documents.md).
- Understand the machinery: [How the memory works](../explanation/associative-memory.md).
- Look up any route: [HTTP API](../api/overview.md).
