# Encode something real

## Where you were

Every vector so far arrived ready-made. [Your first memory](first-memory.md)
and [It learns while you watch](learns-while-you-watch.md) drew random
directions and added noise to them, which was honest for showing what the
memory does but dodged the only question you will actually have to answer on
your own data: **what should the numbers be?**

Nothing in the engine answers it. The store compares directions; which
directions your data occupies is entirely your decision.

## What this lesson shows

You will encode the same 600 card transactions three ways, write each into its
own collection, and watch the memory's idea of "similar" change completely —
without touching a single engine setting.

The encoding is not preparation for the work. The encoding *is* the work.

## 1. The data

Transactions with five fields: amount, hour, distance from home, whether it was
online, and a category. Save as `lesson3.py`:

```python
from heather import call, unit, cos
import hashlib, math, random

D = 128
random.seed(11)
CATS = ["coffee", "groceries", "fuel", "online", "pharmacy"]

def txn(i):
    """A plausible transaction. Amount and category vary independently — a
    coffee can be a $5 espresso or a $60 round for the office — because that
    independence is what makes the encoding choice matter."""
    c = CATS[i % 5]
    return {"amount": round(random.lognormvariate(3.2, 1.0), 2),
            "hour": random.choice([7, 8, 9, 12, 17, 18, 19, 20, 21]),
            "distance": round(random.uniform(0, 12), 1),
            "online": c == "online",
            "cat": c}

train = [txn(i) for i in range(600)]
held  = [txn(i) for i in range(40)]     # never written; used to see what normal looks like

def symbol(label, d=16):
    """A stable random direction for a categorical value. Hashing the label
    seeds the generator, so the same string gives the same vector in every
    process, with no lookup table to persist."""
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    r = random.Random(seed)
    return unit([r.gauss(0, 1) for _ in range(d)])

def pad(v):
    """Zero-pad to the store's width. Padding preserves the norm; projecting
    would mix the fields you just carefully separated."""
    return unit(v + [0.0] * (D - len(v)))
```

## 2. Three encodings

```python
# (a) Raw: every field as a number, scaled by whatever felt reasonable.
def raw(t):
    return pad([t["amount"] / 100.0, t["hour"] / 24.0, t["distance"] / 50.0,
                1.0 if t["online"] else 0.0] + list(symbol(t["cat"], 4)))

# (b) Squashed: the same, but amount divided by the largest value it could take,
#     which is the obvious way to stop one field dominating.
def squashed(t):
    return pad([t["amount"] / 5000.0, t["hour"] / 24.0, t["distance"] / 50.0,
                1.0 if t["online"] else 0.0] + list(symbol(t["cat"], 4)))

# (c) Designed: each field encoded according to what kind of thing it is.
AMOUNT_BUCKETS = [15, 60, 150, 600, 2000]
DIST_BUCKETS   = [1, 5, 25, 100, 500]

def designed(t):
    # Magnitude becomes a band. Cosine cannot see size, only direction, so a
    # number has to become a *position* before the memory can notice it.
    amt = [0.0] * (len(AMOUNT_BUCKETS) + 1)
    amt[sum(1 for b in AMOUNT_BUCKETS if t["amount"] > b)] = 1.3

    dst = [0.0] * (len(DIST_BUCKETS) + 1)
    dst[sum(1 for b in DIST_BUCKETS if t["distance"] > b)] = 1.0

    # An hour is cyclic: 23:00 and 01:00 are two hours apart, not twenty-two.
    hour = [0.45 * math.cos(2 * math.pi * t["hour"] / 24),
            0.45 * math.sin(2 * math.pi * t["hour"] / 24)]

    # The weights are the similarity metric. Category at 0.75 against hour at
    # 0.45 says a same-category match matters more than a same-hour match.
    return pad(amt + dst + hour + [0.3 if t["online"] else 0.0]
               + [0.75 * x for x in symbol(t["cat"], 16)])
```

## 3. Ask what "similar" means

Take one probe — a $58 coffee at 6pm — and look at the stored transactions
nearest to it under each encoding.

```python
probe = {"amount": 58.00, "hour": 18, "distance": 3.0, "online": False, "cat": "coffee"}

for name, enc in (("raw", raw), ("squashed", squashed), ("designed", designed)):
    ranked = sorted(((cos(enc(probe), enc(t)), i) for i, t in enumerate(train)), reverse=True)
    nearest = [train[i] for _, i in ranked[:6]]
    print("%-9s amounts %s" % (name, [round(t["amount"]) for t in nearest]))
```

```
raw       amounts [52, 58, 53, 62, 64, 67]
squashed  amounts [33, 52, 12, 35, 25, 19]
designed  amounts [35, 19, 52, 33, 36, 15]
```

Same data, same probe, three different answers to "what is this like?"

Under **raw**, the amount is by far the largest number in the vector, so it
dominates the direction: the nearest transactions are the ones costing about
$58. The encoding has quietly decided that a purchase is defined by its price.

Under **squashed**, dividing by 5000 pressed every amount into a narrow sliver
of the vector, so amount stopped mattering at all: a $12 purchase sits as close
as a $52 one.

Under **designed**, amount matters as a *band*. Everything returned is in the
$15–60 bucket; within that bucket the exact figure is ignored, which is
generally what you meant by "a similar purchase".

None of these is the encoding being wrong. Each is the encoding being obeyed.

## 4. Ask what each encoding can notice

Now write all three into the memory and give each an unusual transaction.

```python
def collection(name):
    return "/db/enc_%s/collections/txn" % name

BIG = {"amount": 5000.0, "hour": 18, "distance": 3.0, "online": False, "cat": "coffee"}
FAR = {"amount": 45.00, "hour": 3, "distance": 900.0, "online": False, "cat": "groceries"}

for name, enc in (("raw", raw), ("squashed", squashed), ("designed", designed)):
    call("POST", "/db", {"name": "enc_" + name, "dimension": D})
    C = collection(name)
    for i in range(0, len(train), 100):            # 2 MB per request; page the writes
        call("POST", C + "/write", {"vectors": [enc(t) for t in train[i:i + 100]]})

    def fidelity(t):
        r = call("POST", C + "/read", {"query": enc(t)})["result"]
        return cos(enc(t), r)

    n = sorted(fidelity(t) for t in held)
    print("%-9s normal min %.3f median %.3f | $5000 coffee %.3f | $45 at 3am, 900km %.3f"
          % (name, n[0], n[20], fidelity(BIG), fidelity(FAR)))
```

```
raw       normal min 0.698 median 0.969 | $5000 coffee 0.359 | $45 at 3am, 900km 0.162
squashed  normal min 0.956 median 0.986 | $5000 coffee 0.777 | $45 at 3am, 900km 0.160
designed  normal min 0.118 median 0.889 | $5000 coffee 0.327 | $45 at 3am, 900km 0.671
```

## What just happened

Three readings of the same event, and the differences are all consequences of
decisions made before the engine saw anything.

**Squashed cannot see the $5,000 charge.** It scores 0.777, inside the range
ordinary transactions occupy. Dividing by the largest possible amount is the
most natural-looking line in the whole file, and it deleted the field: once
every real amount lands between 0.001 and 0.02, the memory has nothing to
notice. A fraud detector built on this encoding is blind to the case it exists
for.

**Raw sees magnitude and little else.** It catches both unusual transactions,
because both contain an extreme number, and it ranks everything by price. It
would miss anything strange that is not numerically large — a normal-sized
purchase in the wrong category at the wrong hour.

**Designed sees structure, and pays for it.** Its normal minimum is 0.118: some
perfectly legitimate transactions read as very novel. With six amount bands,
six distance bands, five categories and a cyclic hour, 600 transactions do not
cover the space, so rare-but-real combinations look unprecedented — which they
are. That is not a bug in the encoding; it is the encoding having more
resolution than the data supports. The levers are fewer bands, lower weights on
the fields you care less about, or more data.

The general property: **an encoding decides what the memory is capable of
noticing, before a single vector is written.** No knob recovers a distinction
you encoded away, and no amount of data teaches the store to care about
something your vectors do not express.

That is why the first question on any new problem is not which model to use. It
is: *what is an item here, and what should make two of them similar?*

## Where to go next

You have three collections holding transactions and no way to say what any
answer refers to — a read returns 128 numbers, not a purchase. The next lesson
fixes that: [Make the answer legible](make-the-answer-legible.md).
