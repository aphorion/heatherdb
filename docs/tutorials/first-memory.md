# Your first memory

Everything you know about making a computer do something intelligent probably
starts the same way: choose a model, collect data, train it, serve it, watch it
drift. This lesson is the first of five that take that sequence apart. There is
no model here and no training step — only a store you write to and read from.

By the end of this page you will have written 200 vectors into the engine,
watched a badly corrupted query reconstruct into the pattern it came from, and
measured how much the engine recognised the question you asked.

The five lessons run in order and share one server, which you start here and
leave running until the end of lesson 5. You need a running engine —
[Getting started](../getting-started.md) covers installing it — plus `curl`
and `python3`. If you followed that page, stop the quickstart server it left
running before you start this one; they use the same port.

## 1. Start the server

```bash
HEATHER_DATA_DIR=/tmp/heather-tutorial \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='tutorial-password' \
  heather
```

The log tells you three things:

```
INFO heather: Opened server path=/tmp/heather-tutorial databases=1 names=["default"]
INFO heather::auth: Auth: bootstrapped admin user from HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD
INFO heather: HeatherDB server listening addr=0.0.0.0:6380
```

A `default` database was created for you, an `admin` user was minted from the
two environment variables, and the engine is listening on port 6380.

> If you leave `HEATHER_ADMIN_PASSWORD` unset, the engine generates a random
> 24-character password instead, prints it once in a box on stderr, and writes
> a copy to `$HEATHER_DATA_DIR/initial-admin-password`. Setting it explicitly
> keeps this tutorial simple.

Leave that terminal running. Open a second one for everything below.

## 2. Check it is alive

```bash
curl http://localhost:6380/health
```
```json
{"status":"ok"}
```

`/health` is the only kind of route that needs no credentials. Everything else
does:

```bash
curl -s http://localhost:6380/collections
```
```json
{"error":"missing or malformed Authorization header"}
```

```bash
curl -s -u admin:tutorial-password http://localhost:6380/collections
```
```json
{"collections":[]}
```

No collections yet — you have not written anything.

## 3. Make a database for this tutorial

```bash
curl -s -u admin:tutorial-password -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"tutorial","dimension":128}'
```
```json
{"name":"tutorial","created_at":1788260246,"dimension":128,"map_size_mb":4096,"collections":0}
```

**Vector dimension belongs to a database, not to the server.** It is fixed at
creation and cannot be changed afterwards — to work at a different dimension
you create a different database.

## 4. Write some vectors

128-dimensional vectors are awkward to type, so from here we drive the engine
from Python. Save this as `tutorial.py`:

```python
import base64, json, math, random, urllib.request

# The engine speaks HTTP and Basic Auth. Nothing here is a client library —
# it is the standard library talking to the same routes curl was hitting.
BASE = "http://localhost:6380"
AUTH = base64.b64encode(b"admin:tutorial-password").decode()

def call(method, path, body=None):
    """One request. Sends JSON if there is a body, returns the parsed reply."""
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Basic " + AUTH})
    return json.load(urllib.request.urlopen(req))

def unit(v):
    """Scale a vector to length 1. Direction is what the memory compares, so
    keeping every vector on the unit sphere keeps the numbers comparable."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]

def cos(a, b):
    """Cosine similarity: 1.0 means same direction, 0.0 means unrelated.
    This is how we will measure whether a reconstruction worked."""
    return (sum(x * y for x, y in zip(a, b))
            / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))

random.seed(11)   # fixed seed so your numbers land near the ones printed here
D = 128           # must match the dimension the `tutorial` database was created with

# Five underlying patterns. Random directions in 128 dimensions are very nearly
# perpendicular to each other, so these five are effectively unrelated — that
# is the property the whole space rests on.
prototypes = [unit([random.gauss(0, 1) for _ in range(D)]) for _ in range(5)]

def jitter(p, spread):
    """A noisy observation of pattern `p`. Bigger spread, worse copy.
    Dividing by sqrt(D) keeps `spread` meaning the same thing at any dimension."""
    return unit([x + random.gauss(0, spread / math.sqrt(D)) for x in p])

# 200 observations, cycling through the five patterns: 40 noisy copies of each.
# The engine is never told there are five patterns, or which sample is which.
samples = [jitter(prototypes[i % 5], 0.35) for i in range(200)]

# One request writes all 200. The collection does not exist yet; writing creates it.
print(call("POST", "/db/tutorial/collections/signals/write", {"vectors": samples}))

# What the memory made of them — how many hard locations it grew, and how
# concentrated the writes were across them.
print(call("GET", "/db/tutorial/collections/signals/stats"))
```

```bash
python3 tutorial.py
```
```
{'count': 200}
{'num_locations': 51, 'total_writes': 1061.5, 'current_eta': 0.0098,
 'avg_write_count': 20.8, 'max_write_count': 45.9}
```

Your numbers will be close to these, not identical.

Two things happened that a vector database would not have done.

First, you never created the `signals` collection. Writing to a collection
creates it.

Second, and more interesting: **200 vectors became 51 hard locations.** The
engine did not store your vectors. It grew a codebook to accommodate them —
competitive learning, running online, in the same call that stored the data.
There is no training step, and `num_locations` is not a setting you chose. It
is a readout of how complex the data turned out to be.

## 5. Read a corrupted query back

Append to `tutorial.py`:

```python
# A badly corrupted observation of pattern 0 — spread 0.9 against the 0.35 the
# stored samples were made with, so this is far noisier than anything written.
query = jitter(prototypes[0], 0.9)

# How wrong the query is: its similarity to the pattern it came from.
print("query vs prototype: %.3f" % cos(query, prototypes[0]))

# Hand that mess to the engine. `result` is the vector it settled on.
result = call("POST", "/db/tutorial/collections/signals/read", {"query": query})["result"]

# How right the answer is: the same measurement, against the same pattern.
print("result vs prototype: %.3f" % cos(result, prototypes[0]))

# Now the comparison that matters. If this were a search index, the best it
# could hand back is a row you wrote. So: how good is the closest one?
ranked = sorted(((cos(query, s), i) for i, s in enumerate(samples)), reverse=True)
nearest = samples[ranked[0][1]]
print("nearest stored row vs prototype: %.3f" % cos(nearest, prototypes[0]))

# And the obvious next move a vector database would let you make by hand:
# average the ten nearest rows.
top10 = [samples[i] for _, i in ranked[:10]]
centroid = unit([sum(v[j] for v in top10) / 10 for j in range(D)])
print("mean of 10 nearest rows vs prototype: %.3f" % cos(centroid, prototypes[0]))
```

```
query vs prototype: 0.768
result vs prototype: 0.989
nearest stored row vs prototype: 0.946
mean of 10 nearest rows vs prototype: 0.994
```

You handed the engine a vector 0.77 similar to the pattern it came from, and it
handed back one 0.99 similar. It did not return a stored row: the closest row
you wrote only reaches 0.946, because every row is itself a noisy copy. The
answer is a pattern nobody ever wrote, assembled from the interference between
everything relevant that was.

Read the last line honestly, because it is the strongest objection to
everything that follows. **Averaging the ten nearest rows also lands at 0.994**
— slightly better than the engine. That is not an embarrassment, it is the
mechanism: reconstruction *is* a weighted average, and here you have done it by
hand with the rows still in front of you.

The difference is what each side had to keep. Your average needed all 200 rows,
a similarity search over them, and a choice of *k* that you happened to get
right. The engine held 51 locations, no rows at all, chose its own weights, and
returned a number saying how much it recognised the question. Hold that
comparison in mind — the next four lessons are about the things the by-hand
version cannot do: get better as data arrives without being rebuilt, answer
questions about a record's internals, judge its own answer, and absorb an
unrelated subject without disturbing this one.

## 6. Ask why

Reconstruction is not a black box. `analyze` runs the same read and shows its
work:

```python
# `analyze` runs the same read but reports its working instead of just the answer.
a = call("POST", "/db/tutorial/collections/signals/analyze", {"query": query})

# How many settling steps the read took, and whether it reached a stable state.
print("iterations=%s converged=%s" % (a["iterations"], a["converged"]))

# The three hard locations that contributed most: how close each was to the
# query, and how much of the final answer each one accounts for.
for c in a["activated_locations"][:3]:
    print("  location %-3d similarity=%.3f weight=%.3f" % (c["id"], c["similarity"], c["weight"]))
```

```
iterations=3 converged=True
  location 26  similarity=0.947 weight=0.093
  location 46  similarity=0.948 weight=0.094
  location 31  similarity=0.940 weight=0.090
```

Those are the hard locations that produced the answer and how much each one
contributed. The [Hopfield read](../terms/hopfield-read.md) settled in three
iterations.

## 7. The fidelity signal

Compare what you asked for with what came back:

```python
# A direction the memory has never seen — not a noisy copy of anything stored,
# just a fresh random vector.
novel = unit([random.gauss(0, 1) for _ in range(D)])

# Read both, and compare each query with its own reconstruction. Note what is
# being measured: not query vs prototype (we only know the prototypes because
# we generated them), but query vs what came back — which you can always compute.
for label, v in (("known", query), ("novel", novel)):
    r = call("POST", "/db/tutorial/collections/signals/read", {"query": v})["result"]
    print("fidelity(%-5s) = %.3f" % (label, cos(v, r)))
```

```
fidelity(known) = 0.745
fidelity(novel) = 0.141
```

That number — the cosine between a query and its reconstruction — is
**fidelity**. It is not a field the engine returns: it is one line of
arithmetic over the answer you already have, available on every read. High
fidelity means a
strong attractor exists for this pattern. Low fidelity means the query is
novel, underrepresented, or in conflict with what the memory already holds.

## Leave it running

Do not stop the server or delete the data directory — the next four lessons use
this same engine, and lesson 5 tears everything down at the end.

## Where to go next

The next lesson answers the question this one leaves open. Nothing was trained,
so when did the memory get good at this? [It learns while you
watch](learns-while-you-watch.md).
