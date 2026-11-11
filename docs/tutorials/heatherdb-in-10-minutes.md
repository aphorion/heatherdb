# HeatherDB in 10 minutes

HeatherDB learns from every row you write. In this tutorial you build three
things with it, with no model, no training step and no encoding:

1. A score: which customers will buy.
2. A recommendation: which song a listener plays next.
3. A flag: which customer doesn't look like the others.

You need a running server ([Getting started](../getting-started.md)) and
Python 3. The script uses only the standard library.

## The whole idea

There are three things to learn:

1. **Describe your rows.** Name each column and its type, and say what you
   want to know.
2. **Write rows.** Before storing a row, HeatherDB guesses it from what it
   already holds, compares, and learns from the miss.
3. **Ask.** `predict` fills in a column. `next` says what comes next.
   `novelty` says how unusual a row is.

## Set up

Save this as `heather.py`, a small helper for calling the server:

```python
import base64, json, random, urllib.request

BASE = "http://localhost:6380"
AUTH = "Basic " + base64.b64encode(b"admin:change-me-1234").decode()

def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": AUTH, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or "null")
```

## Part 1: who will buy?

**Describe the rows.** A customer has a plan, a country, a number of visits,
an amount spent, and whether they bought. You want to know `bought`:

```python
call("POST", "/db", {
    "name": "shop",
    "columns": {"plan": "category", "country": "category",
                "visits": "number", "spend": "number", "bought": "number"},
    "predict": ["bought"]})
```

- `category` is any name: a plan, a country, a product id.
- `number` is a quantity.
- A yes/no outcome is a number, `1` or `0`.

There's no dimension to choose and no encoding to design. HeatherDB works
that out from the columns.

**Write rows.** Make 2,000 customers. Pro customers usually buy, free ones
rarely do, and frequent visitors buy more. HeatherDB is never told this
rule:

```python
random.seed(7)
rows = []
for _ in range(2000):
    plan = random.choice(["free", "starter", "pro"])
    visits = random.randint(0, 40)
    likely = {"free": 0.05, "starter": 0.3, "pro": 0.7}[plan] + (0.2 if visits > 20 else 0)
    rows.append({"plan": plan, "country": random.choice(["KE", "NG", "US", "DE"]),
                 "visits": visits, "spend": round(random.lognormvariate(3, 1.5), 2),
                 "bought": int(random.random() < likely)})

for i in range(0, len(rows), 500):
    call("POST", "/db/shop/collections/customers/rows", {"rows": rows[i:i + 500]})
```

Notice what you didn't do:
- You didn't scale `spend`, even though it ranges from cents to thousands.
- You didn't one-hot encode `plan`.

HeatherDB scales numbers itself and learns what each plan means.

**Ask.** Give what you know and leave out the rest:

```python
for who in [{"plan": "pro", "visits": 30}, {"plan": "free", "visits": 2}]:
    out = call("POST", "/db/shop/collections/customers/predict",
               {"row": who, "column": "bought", "contributors": 2})
    print(who, out["predicted"])
```

```
{'plan': 'pro', 'visits': 30} {'bought': 0.885}
{'plan': 'free', 'visits': 2} {'bought': 0.052}
```

The true rates in the data are 0.9 and 0.05. It recovered the rule from
2,000 rows written once.

`contributors` lists the stored customers each answer drew on, so you can
always see why.

## Part 2: what plays next?

**Describe the rows.** A listening history is a sequence of songs, one
collection per listener:

```python
call("POST", "/db", {"name": "music", "columns": {"song": "category"}, "sequence": True})
```

**Write rows.** 300 listeners each play bits of albums in track order:

```python
albums = [[f"a{a}-t{t}" for t in range(1, 6)] for a in range(30)]
random.seed(3)
for u in range(300):
    history = []
    for _ in range(4):
        album, start = random.choice(albums), random.randint(0, 2)
        history += album[start:start + 3]
    call("POST", f"/db/music/collections/user_{u}/rows",
         {"rows": [{"song": s} for s in history]})
```

**Ask.** A new listener plays one song. What's next?

```python
call("POST", "/db/music/collections/new_listener/rows", {"rows": [{"song": "a7-t1"}]})
print(call("GET", "/db/music/collections/new_listener/next?column=song&n=3")["values"])
```

```
[{'value': 'a7-t2', 'score': 4.96}, {'value': 'a12-t1', 'score': 1.27}, {'value': 'a6-t4', 'score': 0.72}]
```

The next track on the same album comes first, by a wide margin. Nobody told
HeatherDB about albums or track order. It learned which songs follow which
from the order of writes, and it learned how much recent plays matter more
than old ones.

## Part 3: is this customer unusual?

The same shop can say how much a customer stands out from the ones it has
seen. Give every column except the predicted one:

```python
out = call("POST", "/db/shop/collections/customers/novelty", {"rows": [
    {"plan": "pro", "country": "KE", "visits": 25, "spend": 40},
    {"plan": "enterprise", "country": "KE", "visits": 400, "spend": 250000}]})
print(out["results"])
```

```
[{'novelty': 1.02}, {'novelty': 4.38}]
```

About `1` means the customer sits the way their neighbours do. The second
one, on a plan nobody has and spending far beyond anyone, scores more than
four times higher. This is the fraud-flag question, with no extra setup.

## What just happened

Every row you wrote was a small lesson. Before storing it, HeatherDB guessed
it and learned from the miss:

- **In the shop**, it learned which columns matter for `bought` and what
  each plan means.
- **In the music library**, it learned a code for each song, so songs that
  follow the same songs end up alike, and it learned how fast the past fades.

The rows themselves are stored exactly as written. What's learned is saved in
the same transaction, so a restart continues where it left off.

## Where next

- [Describe your data with columns](../how-to/describe-your-data-with-columns.md):
  every column type, the full `predict` and `next` options, and the limits.
- [Build an anomaly detector](../how-to/build-an-anomaly-detector.md):
  "how unusual is this?"
- [Your first memory](first-memory.md): the vector layer underneath, and the
  ideas behind it (associative memory, vector algebra).
