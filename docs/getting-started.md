# Getting started

By the end of this page you will have built the engine, booted it, written 200
vectors into it, watched a badly corrupted query reconstruct into the pattern
it came from, and used the fidelity signal to tell a known pattern from an
unknown one.

Everything happens in a throwaway directory. Nothing here touches a real
deployment, and you can delete the whole thing at the end.

You need: a Rust toolchain (1.88 or newer), `curl`, and `python3` for the last
two steps.

## 1. Build the engine

```bash
git clone https://github.com/aphorion/heather-db
cd heather-db
cargo build --release -p heather_server
```

The first build takes a few minutes. It produces one binary,
`target/release/heather`.

## 2. Start the server

```bash
HEATHER_DATA_DIR=/tmp/heather-tutorial \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='tutorial-password' \
  ./target/release/heather
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

## 3. Check it is alive

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

## 4. Make a database for this tutorial

The `default` database exists already, but making your own is one call and
shows you where the dimension actually lives.

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

## 5. Write some vectors

128-dimensional vectors are awkward to type, so from here we drive the engine
from Python. Save this as `tutorial.py`:

```python
import base64, json, math, random, urllib.request

BASE = "http://localhost:6380"
AUTH = base64.b64encode(b"admin:tutorial-password").decode()

def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Basic " + AUTH})
    return json.load(urllib.request.urlopen(req))

def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]

def cos(a, b):
    return (sum(x * y for x, y in zip(a, b))
            / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))

random.seed(11)
D = 128

# Five underlying patterns, and 200 noisy observations of them.
prototypes = [unit([random.gauss(0, 1) for _ in range(D)]) for _ in range(5)]

def jitter(p, spread):
    return unit([x + random.gauss(0, spread / math.sqrt(D)) for x in p])

samples = [jitter(prototypes[i % 5], 0.35) for i in range(200)]

print(call("POST", "/db/tutorial/collections/signals/write", {"vectors": samples}))
print(call("GET", "/db/tutorial/collections/signals/stats"))
```

```bash
python3 tutorial.py
```
```
{'count': 200}
{'num_locations': 51, 'total_writes': 1062.2, 'current_eta': 0.0098,
 'avg_write_count': 20.8, 'max_write_count': 44.8}
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

## 6. Read a corrupted query back

Append to `tutorial.py`:

```python
# A badly corrupted observation of prototype 0.
query = jitter(prototypes[0], 0.9)
print("query vs prototype: %.3f" % cos(query, prototypes[0]))

result = call("POST", "/db/tutorial/collections/signals/read", {"query": query})["result"]
print("result vs prototype: %.3f" % cos(result, prototypes[0]))
```

```
query vs prototype: 0.768
result vs prototype: 0.991
```

That is the whole idea. You handed the engine a vector that was only 0.77
similar to the pattern it came from, and it handed back one that is 0.99
similar. It did not find the nearest stored row — no row you wrote is that
close to the prototype either. It reconstructed the pattern from the
interference between everything relevant that was ever written.

## 7. Ask why

Reconstruction is not a black box. `analyze` runs the same read and shows its
work:

```python
a = call("POST", "/db/tutorial/collections/signals/analyze", {"query": query})
print("iterations=%s converged=%s" % (a["iterations"], a["converged"]))
for c in a["activated_locations"][:3]:
    print("  location %-3d similarity=%.3f weight=%.3f" % (c["id"], c["similarity"], c["weight"]))
```

```
iterations=3 converged=True
  location 0   similarity=0.951 weight=0.094
  location 21  similarity=0.941 weight=0.089
  location 10  similarity=0.933 weight=0.086
```

Those are the hard locations that produced the answer and how much each one
contributed. The Hopfield read settled in three iterations.

## 8. The fidelity signal

Compare what you asked for with what came back:

```python
novel = unit([random.gauss(0, 1) for _ in range(D)])

for label, v in (("known", query), ("novel", novel)):
    r = call("POST", "/db/tutorial/collections/signals/read", {"query": v})["result"]
    print("fidelity(%-5s) = %.3f" % (label, cos(v, r)))
```

```
fidelity(known) = 0.747
fidelity(novel) = 0.142
```

That number — the cosine between a query and its reconstruction — is
**fidelity**, and you get it for free on every read. High fidelity means a
strong attractor exists for this pattern. Low fidelity means the query is
novel, underrepresented, or in conflict with what the memory already holds.

One metric, several jobs. Anomaly detection is fidelity below a threshold.
Cold-start detection is fidelity on a new user's first vector. Regime-change
detection is fidelity drifting down over time. There is no second model to
train for any of them.

## 9. The collection's self-summary

```bash
curl -s -u admin:tutorial-password \
  http://localhost:6380/db/tutorial/collections/signals/fingerprint
```

A single 128-dimensional vector summarising everything ever written to
`signals` — the write-count-weighted centroid of every hard location, refined
through a Hopfield read so it lands on a real attractor rather than being a
bare average.

## Clean up

Stop the server with `Ctrl-C` (it shuts down gracefully and flushes its audit
buffer), then:

```bash
rm -rf /tmp/heather-tutorial
```

## Where to go next

- Run it somewhere real: [Deploy HeatherDB](how-to/deploy.md).
- Give other people scoped access: [Manage users and authentication](how-to/manage-users.md).
- Attach metadata and search over it: [Store and query structured documents](how-to/structured-documents.md).
- Understand what you just watched: [How the memory works](explanation/associative-memory.md).
- Look up any route: [HTTP API reference](reference/http-api.md).
