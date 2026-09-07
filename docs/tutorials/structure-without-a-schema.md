# Structure without a schema

## Where you were

Two lessons in, a vector has been one indivisible thing: you write it, you get
a cleaned-up version of it back. [It learns while you
watch](learns-while-you-watch.md) showed the memory sharpening as data arrives,
but everything it stored was a single blob with no parts.

Real records have parts. A film has a genre *and* a director. An event has an
actor, an action, and a target. The usual way to hold that is a schema — columns
declared up front, or fields in a document, or a graph of edges you maintain.

## What this lesson shows

You are going to build records out of parts, store each record as **one
vector**, search by shape, and then pull an individual field back out of a
record — without declaring a single column, and without the engine ever being
told what a "genre" is.

The tools are three operations you may know from the glossary:
[bind](../terms/bind.md) attaches a role to a value, [bundle](../terms/bundle.md)
superposes several of those into one vector, and [unbind](../terms/unbind.md)
reverses a binding.

## 1. A database and a vocabulary

```bash
curl -s -u admin:tutorial-password \
  -H 'Content-Type: application/json' \
  -d '{"name": "records", "dimension": 128}' \
  http://localhost:6380/db
```

Save as `lesson3.py`:

```python
from heather import call, unit, cos
import random, math

random.seed(3)
D = 128

def symbol():
    """A fresh random direction. In 128 dimensions two of these are almost
    exactly perpendicular, so every symbol is distinguishable from every other
    without anyone assigning ids. This is the whole reason the trick works."""
    return unit([random.gauss(0, 1) for _ in range(D)])

# Roles — the questions a record can answer.
GENRE, DIRECTOR = symbol(), symbol()

# Fillers — the answers. Nothing marks these as different in kind from roles;
# the distinction is entirely in how you use them.
scifi, drama = symbol(), symbol()
nolan, tarkovsky, kubrick = symbol(), symbol(), symbol()

# The three algebra operations, over the stateless /vec routes. These touch no
# collection — they are arithmetic the engine performs on vectors you send.
def bind(a, b):   return call("POST", "/vec/bind",   {"a": a, "b": b})["result"]
def unbind(c, k): return call("POST", "/vec/unbind", {"a": c, "b": k})["result"]
def bundle(vs):   return call("POST", "/vec/bundle",
                              {"terms": [{"vector": v} for v in vs]})["result"]
```

This vocabulary is yours, held client-side. The engine never sees the word
"genre"; it sees a 128-number direction that you have decided means genre.

## 2. Three records, one vector each

```python
films = [("Interstellar", scifi, nolan),
         ("Solaris",      scifi, tarkovsky),
         ("Barry Lyndon", drama, kubrick)]

# "genre is X" bound, "director is Y" bound, the two superposed. The result is
# 128 numbers wide — exactly as wide as a single symbol. Composition does not
# grow the representation, which is why the family is called *reduced*.
records = [bundle([bind(GENRE, g), bind(DIRECTOR, d)]) for _, g, d in films]

C = "/db/records/collections/films"

# Metadata rides along for your benefit; it is not what the memory searches on.
print(call("POST", C + "/write",
           {"vectors": records, "metadata": [{"title": t} for t, _, _ in films]}))
```

```
{'count': 3, 'ids': [0, 1, 2]}
```

Three films are stored. No schema was declared, no field was named, and the
engine cannot tell you how many "columns" this collection has, because it does
not have any.

## 3. Search by shape

```python
# Build the query the same way you built the records. It is not a filter
# expression — it is a vector of the shape you want, and the engine ranks
# stored records by how much they resemble it.
q = bundle([bind(GENRE, scifi), bind(DIRECTOR, nolan)])

for r in call("POST", C + "/documents/query", {"query": q, "n": 3})["results"]:
    print("  %-13s %.3f" % (r["metadata"]["title"], r["similarity"]))
```

```
  Interstellar  1.000
  Solaris       0.460
  Barry Lyndon  -0.171
```

Interstellar matches exactly — both roles agree. Solaris scores about half:
same genre, different director, so one of the two bundled terms lines up.
Barry Lyndon shares neither and lands near zero. That ordering is not a rule
anyone wrote. It falls out of the arithmetic.

## 4. Ask a record for one of its fields

Here is the part that has no equivalent in a vector database.

```python
# Take a record and divide out the role. What comes back is the filler that was
# bound to it — approximately, because the other bundled term interferes.
raw = unbind(records[0], DIRECTOR)

# "Approximately" is not good enough to use directly, so clean it up: compare
# against the vocabulary and take the best match. This is cleanup memory done
# by hand; the engine does the same job on data it holds.
for name, v in (("nolan", nolan), ("tarkovsky", tarkovsky),
                ("kubrick", kubrick), ("scifi", scifi)):
    print("  vs %-10s %.3f" % (name, cos(raw, v)))
```

```
  vs nolan      0.558
  vs tarkovsky  -0.030
  vs kubrick    -0.154
  vs scifi      -0.141
```

Ask the same record a different question:

```python
raw_genre = unbind(records[0], GENRE)
for name, v in (("scifi", scifi), ("drama", drama), ("nolan", nolan)):
    print("  vs %-10s %.3f" % (name, cos(raw_genre, v)))
```

```
  vs scifi      0.610
  vs drama     -0.062
  vs nolan     -0.213
```

## 5. Let the engine do the cleanup

Comparing against a hand-held list works because this vocabulary has five
entries. With fifty thousand it is a similarity search you now have to build
and maintain — which is exactly the job the memory already does. So write the
vocabulary into a collection and read the noisy result back out of it.

```python
V = "/db/records/collections/vocabulary"
vocabulary = [("scifi", scifi), ("drama", drama), ("nolan", nolan),
              ("tarkovsky", tarkovsky), ("kubrick", kubrick)]

# Write each symbol several times so each one forms a solid attractor. This is
# lesson 2 at small scale: repetition is what sharpens a pattern.
for _ in range(20):
    call("POST", V + "/write", {"vectors": [v for _, v in vocabulary]})

# `raw` is the smudged 0.558 vector from the step above. Read it back.
clean = call("POST", V + "/read", {"query": raw})["result"]
for name, v in vocabulary:
    print("  vs %-10s %.3f" % (name, cos(clean, v)))
```

```
  vs scifi      -0.236
  vs drama       0.375
  vs nolan       0.978
  vs tarkovsky  -0.012
  vs kubrick    -0.080
```

0.558 went in and 0.978 came out. You did not compare anything; the memory
settled the noisy vector onto the symbol it was reaching for. That is a
[cleanup memory](../terms/cleanup-memory.md), and it is the same `read` you
have used since lesson 1 — pointed at a vocabulary instead of at data.

## What just happened

You asked a vector "who directed you?" and it answered Nolan. You asked the
same vector "what genre are you?" and it answered science fiction. One 128-number
record, two different questions, two correct answers, and at no point did
anything have a field called `director`.

Notice the shape of the raw answer: **0.558, not 1.0**. Unbinding does not
return the filler, it returns the filler plus the interference of everything
else in the bundle. That is the tax every [vector symbolic
architecture](../explanation/vector-symbolic-architectures.md) pays, and it is
why the field had to be cleaned up before it was usable — first by hand, then
by the engine.

That two-step is the thing to take away, because it is what the usual
alternative cannot do in one system. The algebra composes and decomposes
meaning; the memory removes the noise that composing leaves behind. Symbolic
reasoning over stored data, no model in the loop, and both halves are the
routes you have already been calling. Bundles do have a capacity, and the
vocabulary is yours to keep — [vector symbolic
architectures](../explanation/vector-symbolic-architectures.md) covers both.

## Where to go next

You can compose meaning and take it apart. The next lesson turns the fidelity
number into a decision: [Asking what isn't there](asking-what-isnt-there.md).
