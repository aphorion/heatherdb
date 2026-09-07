# Know it worked

## Where you were

You can encode data, store it, ask questions of it, and turn an answer back
into a thing. Every lesson so far ended with a number that looked good.

None of those numbers were checked.

## What this lesson shows

You will build a measurement that scores **1.000 on data containing nothing to
learn**, then repair it, then apply the repaired version to a real claim — and
find that the honest answer is good, but not as good as it first appeared, and
that "chance" is not where you assumed.

This is the shortest lesson in the track and the one that decides whether
anything else in it can be trusted.

## 1. A perfect score on noise

Two piles of random vectors, arbitrarily labelled A and B. There is no
structure: the labels mean nothing, by construction.

```python
from heather import call, unit, cos
import random

D = 128
random.seed(3)

def rnd():
    return unit([random.gauss(0, 1) for _ in range(D)])

# 150 random vectors called "A" and 150 more called "B". Nothing distinguishes
# them — the names are arbitrary and there is nothing here to discover.
A = [rnd() for _ in range(150)]
B = [rnd() for _ in range(150)]

def build(db, vectors):
    call("POST", "/db", {"name": db, "dimension": D})
    C = "/db/%s/collections/c" % db
    call("POST", C + "/bulk_load", {"addresses": vectors, "counters": vectors})
    return C

cA, cB = build("noise_a", A), build("noise_b", B)

def pick(v):
    """Whichever memory reconstructs this vector better claims it."""
    a = cos(v, call("POST", cA + "/read", {"query": v})["result"])
    b = cos(v, call("POST", cB + "/read", {"query": v})["result"])
    return "A" if a >= b else "B"

correct = sum(pick(v) == "A" for v in A) + sum(pick(v) == "B" for v in B)
print("scored on the vectors that were written : %.3f" % (correct / 300))
```

```
scored on the vectors that were written : 1.000
```

A perfect classifier of random noise. Nothing is broken, and the engine is not
at fault: every vector being scored is *in* the memory being asked, so each one
recalls itself. The measurement asked whether the store contains the item, and
the answer was yes.

Any procedure that scores an item against a memory holding that item measures
storage, not generalisation. It will read near-perfect on anything.

## 2. The repair

Score vectors the memory has never seen.

```python
held_A = [rnd() for _ in range(50)]
held_B = [rnd() for _ in range(50)]

correct = sum(pick(v) == "A" for v in held_A) + sum(pick(v) == "B" for v in held_B)
print("scored on vectors never written         : %.3f" % (correct / 100))
```

```
scored on vectors never written         : 0.490
```

0.490 — a coin. The same code, the same memories, the same metric, and the
result went from perfect to worthless because the answer stopped being in the
room. That gap, 1.000 against 0.490, is the size of the lie a self-scored
measurement tells.

**Hold out before you write, not after.** Decide which items the memory will
never see, keep them aside, and score only on those.

## 3. Now measure a real claim

The claim: this memory can tell a coffee purchase from a grocery run. Two
collections, 150 transactions each, using the `designed` encoder from
[Encode something real](encode-something-real.md).

```python
# 150 of each class written; 50 of each held back, never written.
colls = {"coffee": build("cls_coffee", [designed(t) for t in coffee[:150]]),
         "groceries": build("cls_groc", [designed(t) for t in groceries[:150]])}

def classify(t):
    q = designed(t)
    scores = {name: cos(q, call("POST", c + "/read", {"query": q})["result"])
              for name, c in colls.items()}
    return max(scores, key=scores.get)

seen   = [(t, "coffee") for t in coffee[:150]] + [(t, "groceries") for t in groceries[:150]]
unseen = [(t, "coffee") for t in coffee[150:200]] + [(t, "groceries") for t in groceries[150:200]]

accuracy = lambda data: sum(classify(t) == label for t, label in data) / len(data)
print("scored on the items that were written : %.3f" % accuracy(seen))
print("scored on items never written         : %.3f" % accuracy(unseen))
```

```
scored on the items that were written : 0.973
scored on items never written         : 0.940
```

0.940 on data the memory has never seen. This time the drop is small, which is
the difference between a real result and the noise experiment above — the
memory genuinely generalises here.

## 4. Ask what the number would be if the claim were false

0.940 sounds strong against a coin. But nothing has established that a coin is
the right comparison. Build the null: run the identical procedure with the
labels shuffled before writing, so the categories are destroyed and everything
else — the encoding, the transactions, the collection sizes, the scoring code —
is preserved.

```python
# Same items, same procedure, labels scrambled. Any structure the categories
# carried is now gone; anything else is untouched.
labels = [label for _, label in seen]
random.shuffle(labels)
shuffled = [(t, l) for (t, _), l in zip(seen, labels)]

null_colls = {
    "coffee":    build("null_a", [designed(t) for t, l in shuffled if l == "coffee"]),
    "groceries": build("null_b", [designed(t) for t, l in shuffled if l == "groceries"]),
}
# ...classify the same held-out items against these
```

```
null (labels shuffled), unseen items  : 0.610
```

**Chance is 0.610 here, not 0.5.** Two classes did not mean a coin flip: the
transactions are not evenly spread, and a memory built from any 150 of them
recalls some held-out items better than others regardless of labelling.

So the honest statement of the result is not "94% accurate". It is: *0.940
against a null of 0.610*, on 100 items never written. That is still a real
effect, and it is a smaller one than the first number suggested.

## What just happened

Three numbers, and only the third means anything:

| Measurement | Reads | What it measures |
|---|---|---|
| scored on written items | 0.973 | that the store contains what you wrote |
| scored on held-out items | 0.940 | whether it generalises |
| the same, against a shuffled-label null | 0.610 baseline | how much of that is the claim |

The properties worth carrying to any store, not just this one:

**A measurement that includes the answer measures storage.** It reads
near-perfect on anything, including noise.

**A null is built by destroying exactly what you claim and keeping the rest.**
Shuffle the labels, not the data. Rotate each series independently if the claim
is about coupling. The null runs through the identical pipeline.

**Chance is measured, not assumed.** Two classes are not a coin, and the gap
between the assumed baseline and the measured one is where confident wrong
conclusions live.

**A metric that cannot fail has not succeeded.** Before trusting a number, run
the procedure on input with no answer in it and check that it says so.

## Where to go next

One question is left, and it is the one that decides whether any of this is
deployable: what happens to what you taught it when you teach it something
else? [It doesn't forget](it-doesnt-forget.md).
