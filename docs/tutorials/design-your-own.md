# Design your own

## Where you were

Eight lessons, all on data chosen to make a point. You have watched a
reconstruction beat the row it came from, watched a memory sharpen without a
training step, held a record in one vector, made an answer legible, derived a
threshold instead of guessing one, measured a claim honestly, and written an
unrelated subject on top of a first without disturbing it.

This lesson uses none of our data. You bring a problem.

## What this lesson shows

By the end you will have a working system on your own data, built by answering
four questions in order. The questions are the transferable part — they are
what you will still have when the details of this engine have faded.

> 1. **What is an item?**
> 2. **What makes two items similar?**
> 3. **What is a pool?**
> 4. **What does low fidelity mean here?**

Answer them and the code is nearly mechanical. Skip one and you will build
something that runs and does not work.

## 1. What is an item?

The thing that becomes one vector, and the thing an answer will be about.

Choosing it is a decision about granularity, not a description of your data. A
support inbox can have a *message* as the item, or a *conversation*, or a
*customer*. All three are defensible and they build different systems: one
recalls similar messages, one recalls similar conversations, one profiles
people. Pick the one whose answers you actually want to read.

Write your answer down in this form: *an item is one ____, and there are
roughly ____ of them.*

The count matters, because it is the input to question 3.

## 2. What makes two items similar?

The design surface. The store compares direction, so this question is *what
should point the same way?* — and nothing about the engine answers it.

Go field by field and classify each one, exactly as [Encode something
real](encode-something-real.md) did with transactions:

| Kind of field | Encode as | Guide |
|---|---|---|
| identity, category, label | a hash-seeded random symbol | [tabular](../how-to/encode-tabular-data.md) |
| magnitude | a band, or a fractional power | [tabular](../how-to/encode-tabular-data.md) · [quantities](../how-to/encode-quantities-and-time.md) |
| cycle (hour, weekday, angle) | sin/cos pair | [tabular](../how-to/encode-tabular-data.md) |
| set of things | a bundle of their symbols | [tabular](../how-to/encode-tabular-data.md) |
| ordered sequence | permutation, not binding | [reference](../reference/encodings.md) |
| free text | embedding, hashed, or self-formed | [text](../how-to/encode-text.md) |
| structure with roles | bind, then bundle | [structure](structure-without-a-schema.md) |

Then decide the weights, because the weights *are* the similarity metric.
A category at 0.75 against an hour at 0.45 is a statement that two items of the
same kind at different times are more alike than two different kinds at the
same time. If that sentence is wrong for your problem, the weights are wrong.

Two failure modes to check before you write anything:

- **A field you left raw will dominate**, if its numbers are larger than
  everything else in the vector.
- **A field you squashed has vanished.** The most natural-looking normalisation
  in your file is the likeliest place to have deleted the thing you care about.

If your vectors come from an embedding model, subtract the corpus mean before
comparing anything: [Center your vectors](../how-to/center-your-vectors.md).

## 3. What is a pool?

One collection holds one pool, and capacity is per pool — roughly `d/32` items
before recall degrades. Nothing in the engine warns you when you cross it; the
answers simply get worse.

So decide what a collection *is* before you write:

- **One per entity** — a user, a device, a tenant, a conversation. This is the
  default for anything personalised, and a single global collection loses to
  plain cosine where per-entity collections win.
- **One per class** — for classification, so "which memory reconstructs this
  best" is the decision.
- **One per corpus** — for search, sharded by a natural key when it outgrows
  `d/32`.

Write down: *a collection is one ____, holding roughly ____ items, at dimension
____.* If items per collection exceeds `d/32`, either raise the dimension or —
usually better — [shard](../how-to/shard-a-large-pool.md).

## 4. What does low fidelity mean here?

Every read returns something. The question is what you do when it returns
something the memory did not recognise, and the answer must exist before you
deploy, because it is your abstention rule.

Pick the sentence that fits your problem:

- *Low fidelity means this is anomalous* — alert on it.
- *Low fidelity means we have not seen enough of this user yet* — fall back to
  a default.
- *Low fidelity means do not answer* — refuse, and route to a human.

Then derive the threshold from held-out data at a false-alarm budget you can
defend, as in [Asking what isn't there](asking-what-isnt-there.md). Do not
choose a number by eye, and do not copy one from these pages.

## 5. Build it

The shape is the same regardless of problem:

```python
from heather import call, unit, cos

D = 128                       # from question 3
POOL = "/db/mine/collections/items"

# --- question 2, made concrete -------------------------------------------
def encode(item):
    """Return one unit vector. Everything you decided above lives here, and
    nothing downstream can recover a distinction this function discards."""
    ...

# --- hold out first, before anything is written --------------------------
train, calibration, test = split(my_items)     # e.g. 80% / 10% / 10%

# --- write ---------------------------------------------------------------
call("POST", "/db", {"name": "mine", "dimension": D})
for i in range(0, len(train), 100):
    call("POST", POOL + "/write",
         {"vectors": [encode(x) for x in train[i:i + 100]],
          "metadata": [{"id": x["id"]} for x in train[i:i + 100]]})

# --- read, and resolve the answer back to a thing ------------------------
def ask(item):
    q = encode(item)
    result = call("POST", POOL + "/read", {"query": q})["result"]
    return cos(q, result), nearest_stored(result)   # see lesson 4
```

Two engine facts that bite at this point. A `write` without `metadata` is
unretrievable as a document later, though it still shapes the memory. And a
read auto-creates a collection on a typo, returning empty results rather than
an error — verify a collection exists with `GET .../stats`, which does 404.

## 6. Check it the way lesson 7 checked

Before you believe anything:

1. Score on `test`, never on `train`.
2. Run the null — destroy the structure you are claiming, keep everything else,
   and score again. Report your number beside it.
3. Feed it an input with no answer in it and confirm it says so.

If the null scores close to your result, you have measured your pipeline rather
than your data. That is a finding too, and a cheap one to have on day one
instead of after deployment.

## Grading your own design

A design is finished when you can answer these without hedging:

- [ ] An item is one ____, and there are about ____ of them.
- [ ] Two items are similar when ____ — and I can name a pair that *should*
      score high and a pair that should not.
- [ ] Each field is encoded as identity / magnitude / cycle / set / order /
      text, and I know which field dominates the vector.
- [ ] A collection is one ____, holding about ____ items, at dimension ____,
      which is under `d/32`.
- [ ] Low fidelity means ____, and the threshold comes from held-out data at a
      ____% false-alarm budget.
- [ ] My reported number is ____, measured on held-out data, against a null of
      ____.

## Where to go next

- Build one of the standard shapes rather than starting from nothing:
  [recommender](../how-to/build-a-recommender.md) ·
  [anomaly detector](../how-to/build-an-anomaly-detector.md) ·
  [semantic search](../how-to/build-a-semantic-search.md) ·
  [classifier](../how-to/build-a-classifier.md).
- Know where the technique stops:
  [Boundary conditions](../explanation/boundary-conditions.md).
- Put it somewhere real: [Run it in production](../how-to/production.md).
