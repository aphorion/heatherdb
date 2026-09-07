# Build a classifier

Assign a label from a handful of examples per class, and add a new class by
writing it.

The conventional stack for this is a trained head over an encoder, a labelled
training set large enough to fit it, a serving step, and a retrain every time
a class is added or an example is corrected. The memory replaces the training
and the retraining: the labelled examples *are* the model, a class is a
collection, and adding one is a load. You still supply an encoder, you still
supply labelled examples, and you still derive the abstention threshold.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| An item is | one labelled example — a ticket, a task, a row | the unit you classify |
| Similar means | encoder cosine, centred | see [Encode text](encode-text.md) |
| A pool is | one collection per class, `cls_<label>` | each class's examples must not interfere with another's |
| Loading | `bulk_load` | the labelled examples must be the only attractors |
| Answer resolves to | the collection name that reconstructs the input best | the label is the collection, so no sidecar is needed |
| Abstain when | the top score, or the top-two margin, is below the derived cut | an unseen class must not be forced into a known one |

## Encode an example

Whatever the class is defined on, encode it so that two members of the same
class point the same way. For text, that is an embedding model; for rows, a
weighted feature vector.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")     # 384-d

def encode(text: str) -> list[float]:
    v = model.encode(text, normalize_embeddings=True)
    # Centre before comparison. Raw model output shares a large common
    # component across every vector, which inflates all scores equally and
    # separates nothing -- and separation is the entire job here.
    v = v - CORPUS_MEAN
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()
```

The database's dimension must equal the encoder's width, or every load fails
with a dimension mismatch — and dimension is immutable per database, so settle
it first ([Choose a dimension](choose-a-dimension.md), [Center your
vectors](center-your-vectors.md)).

**Plain text embeddings beat structured encodings as class keys at small
scale.** Measured on a corpus of 40 items across 8 classes, raw task text
embedded directly gave top-1 of 40/40, while binding attribute/value pairs
into a role-filler structure gave 35/40. Reach for [structured
documents](structured-documents.md) when class confusion actually appears.

## One collection per class

```python
def col(label: str) -> str:
    # The label is the collection name, so a correct answer needs no lookup
    # table -- the winning collection IS the class.
    return "/db/triage/collections/cls_%s" % label
```

Classes are independent memories: a class with forty examples cannot blur a
class with four, and dropping a class is `DELETE` on one collection rather
than a filtered delete across a shared index.

## Load with `bulk_load`

`write` grows a codebook that generalises — it splits and migrates locations
to cover the space the data lives in, so a collection built by `write` has
space-filling attractors that reconstruct almost anything well. That destroys
the comparison between classes.

`bulk_load` replaces the collection's locations with exactly the rows you
supply, one location per example, nothing else.

```python
def load_class(label: str, examples: list[str]) -> None:
    rows = [encode(t) for t in examples]

    # Create first so a typo in `label` fails loudly here rather than
    # silently minting a new empty class on the first read.
    call("POST", "/db/triage/collections", {"name": "cls_" + label})

    # Addresses and counters are the same rows: address is what a query is
    # matched against, counter is what comes back. Identical rows mean
    # "reconstruct this example verbatim when something looks like it".
    call("POST", col(label) + "/bulk_load",
         {"addresses": rows, "counters": rows})
    # -> {"n_loaded": 5, "dim": 384}
```

`bulk_load` replaces the whole collection, so correcting or adding an example
is a full reload of that one class from your own store of labelled data —
the price of keeping the attractor set exactly equal to the labelled set, and
free at a few dozen examples per class. It also orphans the document index, so
keep these collections document-free; `documents/query` is not needed here,
because the label is the collection name.

## Classify by which class reconstructs the input best

One read per class, scored by the cosine between the input and what came
back. There is no `fidelity` field on the wire; the caller computes it.

```python
def classify(text, labels):
    q = encode(text)

    scores = []
    for label in labels:
        r = call("POST", col(label) + "/read",
                 # "iterative" settles the query onto this class's attractor
                 # set. An input belonging to the class is pulled all the way
                 # onto a stored example; an input from elsewhere is dragged
                 # somewhere it never asked to go, and scores low.
                 {"query": q, "strategy": "iterative"})["result"]
        scores.append((cos(q, r), label))

    scores.sort(reverse=True)
    return scores
```

```
0.81  dependency-bump
0.44  flaky-test
0.39  ci-config
```

Two numbers come out of that list. **The top score** says whether any class
fits at all; **the margin** between the top two says whether the winner is the
right class rather than the nearest of several equally poor fits.

## Gate on the margin, not on a global cut

```python
def classify_or_abstain(text, labels):
    scores = classify(text, labels)
    top, second = scores[0], scores[1] if len(scores) > 1 else (0.0, None)

    # Both conditions, because they fail differently: a low top score is an
    # input no class covers, and a thin margin is an input two classes cover
    # equally. Forcing either one produces a confident wrong label.
    if top[0] < TAU or (top[0] - second[0]) < MARGIN:
        return {"label": None, "confidence": round(top[0], 3),
                "reason": "below gate"}
    return {"label": top[1], "confidence": round(top[0], 3)}
```

Derive both cuts, do not choose them — hold out labelled examples per class
*before* loading, plus a set of inputs from classes you deliberately did not
load, and put each cut in the empty band between the groups. Report coverage
and error rate together. [Calibrate a gate](calibrate-a-gate.md) is the
procedure; [Find your metric's floor](find-your-metrics-floor.md) is how you
check the separation is not an artefact of the split.

**A single global similarity cut may not exist for your data even when
per-query classification is perfect.** On the 40-item, 8-class corpus above,
a global pairwise threshold scored AUC 0.789 with no clean band anywhere,
while the per-query statistic — top-1 similarity, leave-one-class-out —
separated known from novel classes at AUC 1.0000 with an empty band across
`(0.234, 0.474)` and a minimum per-query margin of `+0.166`. Gate on the
per-query numbers, and take the threshold from the middle of the measured
band.

## Add a class without retraining

Adding a class is `load_class(label, examples)` and appending the label to the
list you sweep. No retraining, no rebuild of any other class, no downtime for
the classes already serving.

The cost of a new class is one extra read per classification, so a linear
sweep is fine to a few dozen classes. Beyond that, route first — score the
input against each class's `fingerprint` to pick a shortlist, then read only
the shortlisted classes ([Shard a large pool](shard-a-large-pool.md)).

Re-derive the gate afterwards: a cut calibrated against seven classes is not
valid against eight, because the margin distribution changes when a new
neighbour appears.

## The honest limit

With five examples per class, the read activates every location in the
collection and returns their weighted blend. That is a soft nearest-prototype
decision, and it behaves like one:

- **Classes that overlap in the encoder's space will not separate.** No
  amount of tuning fixes an encoding under which two classes point the same
  way. Fix the encoding.
- **There is no discrimination *inside* a class.** The score says "this looks
  like `dependency-bump`", not which of the five examples it matches. If you
  need that, keep the per-example vectors client-side and resolve within the
  winning class.
- **The blend is only as good as the examples.** One mislabelled example in a
  five-example class is a fifth of that class's attractor mass, and it will
  pull inputs toward the wrong label until it is removed.

What the design buys instead is that every one of those failures is
inspectable and one reload away from fixed, and that a class added at noon is
serving at noon.

## Verify

`GET .../collections/cls_{label}/stats` on every class before serving. It
404s on a collection that does not exist — the check matters, because a read
against a typo creates an empty collection and then scores every input against
nothing. `num_locations` must equal the number of examples you loaded;
anything higher means something called `write` against the collection, and its
attractors are no longer the labelled set.

To see which stored example carried a decision, use `analyze` instead of
`read`: it returns the same reconstruction with the activated locations and
their softmax weights attached, and `bulk_load` assigns location ids by row
index, so the id is the index into the list you loaded. See [Explain a
result](explain-a-result.md).

## Related

- [Calibrate a gate](calibrate-a-gate.md) — deriving `TAU` and `MARGIN` ·
  [Find your metric's floor](find-your-metrics-floor.md)
- [Explain a result](explain-a-result.md) — `analyze` and the activation trace
- [Encode text](encode-text.md) · [Center your vectors](center-your-vectors.md)
  · [Choose a dimension](choose-a-dimension.md)
- [Shard a large pool](shard-a-large-pool.md) — routing when classes multiply
- [Write endpoints](../api/writes.md) — `bulk_load` · [Read
  endpoints](../api/reads.md) · [Abstention](../explanation/abstention.md)

Derived from `reflex`, the procedural-memory classifier in `heather_apps`.
