# Make the answer legible

## Where you were

[Encode something real](encode-something-real.md) left you with three
collections of card transactions and a problem: a read returns 128 numbers.
Not a transaction. Not an id. Not a row you can put in front of a human or an
alerting system.

Every lesson so far has hidden this, because the queries were vectors you
generated and could compare against patterns you still had lying around. On
real data you do not have the answer to compare against — that is the point of
asking.

## What this lesson shows

You will attach a second store — a plain dictionary — and use it to turn a
reconstruction back into a transaction. Then you will resolve the *same* read
two ways, and see that the two answers are different questions:

- the item nearest **your query** is the closest thing you already had;
- the item nearest **the reconstruction** is what the memory expected instead.

The gap between those two is the finding. It is also the thing a vector index
cannot give you, because it only ever answers the first question.

## 1. Teach it what normal looks like

Reuse the `designed` encoder and the 600 transactions from the last lesson.
This time load them with `bulk_load` rather than `write`:

```python
from heather import call, unit, cos

D = 128
C = "/db/txn_taught/collections/txn"
call("POST", "/db", {"name": "txn_taught", "dimension": D})

vectors = [designed(t) for t in train]

# bulk_load places the vectors as the codebook directly, so the transactions you
# taught are the only attractors. A `write`-built collection grows locations that
# generalise, which is what you want when learning online and not what you want
# when the question is "how far is this from the few things I was taught".
call("POST", C + "/bulk_load", {"addresses": vectors, "counters": vectors})
```
```json
{"n_loaded": 600, "dim": 128}
```

Keep the transactions themselves beside the vectors. This is the second store,
and it is not optional — the engine holds directions, not records.

```python
# The sidecar. A dict is enough at this size; a real system uses SQLite or
# whatever it already has. What matters is that every vector you wrote can be
# turned back into the thing it stood for.
catalogue = [(t, designed(t)) for t in train]

def nearest(vector):
    """The stored transaction whose vector points most nearly the same way."""
    return max(catalogue, key=lambda pair: cos(vector, pair[1]))[0]
```

## 2. Resolve one read two ways

```python
def inspect(t, label):
    query = designed(t)
    result = call("POST", C + "/read", {"query": query})["result"]

    # Two different questions asked of the same read:
    #   nearest to the query        — what is this most like, of things I stored?
    #   nearest to the reconstruction — what did the memory settle on instead?
    show = lambda x: "$%-8.2f %-9s %02d:00 %6.1fkm" % (
        x["amount"], x["cat"], x["hour"], x["distance"])

    print("%s  fidelity %.3f" % (label, cos(query, result)))
    print("    asked   %s" % show(t))
    print("    query→  %s" % show(nearest(query)))
    print("    recon→  %s" % show(nearest(result)))

inspect({"amount": 12.00, "hour": 8, "distance": 600.0, "online": False, "cat": "coffee"},
        "a coffee, 600 km from home")
inspect({"amount": 1800.00, "hour": 12, "distance": 2.0, "online": False, "cat": "pharmacy"},
        "an $1,800 pharmacy charge")
inspect({"amount": 52.00, "hour": 18, "distance": 3.0, "online": False, "cat": "coffee"},
        "an ordinary evening coffee")
```

```
a coffee, 600 km from home  fidelity 0.731
    asked   $12.00    coffee    08:00  600.0km
    query→  $13.12    coffee    08:00    4.3km
    recon→  $13.44    coffee    08:00   11.7km

an $1,800 pharmacy charge  fidelity 0.496
    asked   $1800.00  pharmacy  12:00    2.0km
    query→  $14.77    pharmacy  12:00    3.0km
    recon→  $33.35    pharmacy  08:00    1.3km

an ordinary evening coffee  fidelity 0.989
    asked   $52.00    coffee    18:00    3.0km
    query→  $33.22    coffee    18:00    3.1km
    recon→  $20.07    coffee    17:00    2.8km
```

## What just happened

Read the first case field by field. You asked about a $12 coffee at 8am, 600 km
from home. The memory returned a $13 coffee at 8am, 12 km from home. Every
field agrees except the one that is wrong, and the one that is wrong is named
with the value it should have had.

That is an explanation, and nothing produced it but a subtraction. There is no
second model, no rule about distance, and no feature-importance calculation —
the memory reconstructed the nearest thing to a coherent transaction, and what
it had to change to get there is the anomaly.

The $1,800 pharmacy charge behaves the same way: what came back is a $33
pharmacy purchase. The amount band is the discrepancy, and fidelity at 0.496
says the memory had to travel a long way to answer at all.

Now compare the two resolutions on the third case. For an ordinary transaction
they say almost the same thing — an evening coffee near home either way — which
is exactly right: when the memory recognises something, its expectation and
your query coincide. The two answers separate precisely when the input is
unusual, which is when you needed them to.

This is the difference between a memory and an index, in one line of code. An
index answers *what did I already have that is closest to this?* — for the
600 km coffee, that is a coffee 4 km from home, which tells you nothing about
why the charge is strange. A memory answers *what should this have been?*

**Resolve against the reconstruction.** It costs nothing and it changes the
question.

## Two practical notes

**The sidecar is infrastructure, not an optimisation.** Whatever you write
must remain reversible to the thing it stood for, or your answers are
uninterpretable. Metadata attached at write time is the other route — see
[Store and query structured documents](../how-to/structured-documents.md) —
but note that a `write` without `metadata` is not retrievable as a document at
all, even though it still shapes the memory.

**Held-out normals set the scale.** On this collection, 40 transactions the
memory never saw score between 0.874 and 0.986. Both unusual charges fall
below that band. Choosing where the line goes is the next lesson's subject, and
it is not done by eye.

## Where to go next

You have a signal that separates ordinary from unusual, and a way to say what
an answer refers to. Turning that into a decision — with a threshold you can
defend — is [Asking what isn't there](asking-what-isnt-there.md).
