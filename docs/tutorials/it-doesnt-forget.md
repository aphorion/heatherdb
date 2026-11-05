# Measure retention while learning

## Where you were

Four lessons in, you have a memory that reconstructs corrupted queries, sharpens
as data arrives, holds structured records in single vectors, and tells you when
it does not recognise something.

Every one of those results came from a store that only ever learned one thing.
Which raises the question that decides whether any of it survives contact with
production: what happens when you write something *else*?

This exercise measures retention under unrelated writes. It does not establish
that a memory never forgets. When familiar patterns change, gradual loss of old
pattern fidelity can instead be useful [graceful forgetting](../explanation/learning-and-graceful-forgetting.md).

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

The example output shows the original pattern's cosine changing from
0.999 → 0.998 → 0.992 as unrelated observations arrive and the location count
grows. These are illustrative outputs from this exercise, not a retention
bound; record your own results for your build and workload.

Local learning and allocation can preserve unrelated patterns while adding new
ones. Capacity, overlap, write settings and read settings still affect retention.
A changing pattern is a different test: competitive learning may shift responses
toward its new form, gradually reducing old-pattern fidelity. For an adaptive
knowledge policy, that is graceful forgetting. For an archive, preserve the
observations separately.

## What you know now

Measure the behaviour your application needs: held-out pattern learning,
retention of unrelated knowledge, adaptation to drift, or individual recall.
Query-to-response cosine alone is not correctness or calibrated confidence;
validate any application abstention threshold on independent examples.

[Choose a memory mode](../how-to/choose-memory-mode.md) before deciding whether
shared adaptation or preservation of individual observations is the objective.

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
