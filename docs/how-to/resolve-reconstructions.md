# Resolve a reconstruction to a thing

A read returns a vector, not a row. To turn that 128-float array back into a
title, an id or a sentence, keep a sidecar store of item → vector and resolve
the **reconstruction** against it.

## The two-store pattern

The engine owns the associative half: it absorbs vectors into hard locations,
merges what overlaps, and hands back the state it settles into. It does not
own your nouns. Keep those in a second store — a dict, a SQLite table, the
row you already have in Postgres — keyed the same way you keyed the vectors
you wrote.

```python
# The sidecar. Any key-value store works; the only requirement is that the
# same key names both the vector you wrote and the thing it stood for.
import sqlite3, json

side = sqlite3.connect("sidecar.db")
side.execute("CREATE TABLE IF NOT EXISTS items (key TEXT PRIMARY KEY, "
             "label TEXT, vec TEXT)")

def remember(key, label, vec):
    # Store the vector verbatim. Resolution is a cosine against these, so
    # they must be the exact vectors written to the engine, not re-encoded
    # ones — a re-encode with a different model resolves to noise.
    side.execute("INSERT OR REPLACE INTO items VALUES (?,?,?)",
                 (key, label, json.dumps(vec)))

def cache():
    return [(k, l, json.loads(v))
            for k, l, v in side.execute("SELECT key,label,vec FROM items")]
```

## Resolve against the reconstruction, not the query

This is the whole difference between an associative read and a similarity
search. Both end in a nearest-neighbour scan over the same cache; they differ
in what they scan *from*.

- The **query's** neighbours are the items closest to what you already had.
  That is a lookup, and the engine contributed nothing to it.
- The **reconstruction's** neighbours are the items closest to the state the
  memory settled into after superposition, competition and the Hopfield loop.
  That is what the memory concluded.

```python
def nearest(v, items, n=5):
    """Rank cached items by cosine against one vector."""
    return sorted(((cos(v, iv), k, l) for k, l, iv in items), reverse=True)[:n]

q = build_query(...)                       # whatever you are asking with
r = call("POST", "/db/movies/collections/taste/read", {"query": q})["result"]

items = cache()
print("query resolves to        :", nearest(q, items, 3))
print("reconstruction resolves to:", nearest(r, items, 3))
```

Run both on the same read and print them together. On a well-populated
collection they diverge, and the divergence is the point:

```
query resolves to         : [(0.94, 'f17', 'Solaris'),
                             (0.61, 'f02', 'Stalker')]
reconstruction resolves to: [(0.72, 'f02', 'Stalker'),
                             (0.68, 'f88', 'Andrei Rublev')]
```

The query's top hit is the item you were nearest to on the way in. The
reconstruction's ranking is flatter and wider because the settled state is a
blend of every engram that competed for the query — the neighbours of that
blend are the memory's answer.

Compare the two before trusting either:

| Read | Resolve against | Answers |
|---|---|---|
| any | the query | "what did I already have?" |
| `read` / `attention` | the result | "what does the memory make of it?" |
| `analyze` | the result, plus `activated_locations` | the answer and its causes |

## Keep the payload in metadata

A sidecar is only needed when the vector must resolve to something the engine
never saw. If the thing itself is small enough to store, write it as
[document metadata](structured-documents.md) and let the vector be nothing but
an address:

```python
# The verbatim payload rides along with the vector. Nothing is recomputed at
# read time, and there is no second store to keep in sync with the first.
call("POST", "/db/movies/collections/films/write",
     {"vectors": vectors,
      "metadata": [{"title": t, "text": full_text[t]} for t in titles]})
```

Then resolve by handing the reconstruction back to the engine as a document
query, which does the nearest-neighbour scan for you over the posting list:

```python
# `query` here is the RECONSTRUCTION, not the original query — the same
# substitution as above, executed inside the engine instead of your process.
hits = call("POST", "/db/movies/collections/films/documents/query",
            {"query": r, "n": 5})["results"]
for h in hits:
    print(h["similarity"], h["metadata"]["title"])
```

Choose between them on one axis:

| | Sidecar scan | `documents/query` |
|---|---|---|
| Candidates considered | every cached item | only items on the activated locations' posting lists |
| Cost | linear in cache size, in your process | index lookup, in the engine |
| Sees items far from the probe | yes | no — they are never candidates |
| Needs the vectors kept client-side | yes | no |

Use the sidecar when you need a full ranking including distant items, or when
the items are not documents in this collection. Use `documents/query` for
everything else.

## The trap: a write without metadata is not a document

A plain write is retrievable only as influence. It shapes the hard locations,
it changes what every future read reconstructs — and there is nothing to fetch
afterwards.

```python
call("POST", "/db/movies/collections/films/write", {"vectors": vectors})
# → {"count": 200}     … no "ids", and documents/query will never return these
```

Metadata is what mints an id:

```python
call("POST", "/db/movies/collections/films/write",
     {"vectors": vectors, "metadata": [{"title": t} for t in titles]})
# → {"count": 200, "ids": [0, 1, …]}
```

There is no backfill. A collection written without metadata cannot be made
document-retrievable later — the vectors have already been merged into
engrams, and superposition does not come apart. Decide before the first write,
or keep a sidecar so the decision is reversible.

## Related

- [Store and query structured documents](structured-documents.md) — metadata,
  ids, `role_pairs`
- [Explain a result](explain-a-result.md) — attributing a reconstruction to the
  items that produced it
- [Read endpoints](../api/reads.md) · [Document
  endpoints](../api/documents.md)
- [Attractor](../terms/attractor.md) · [Fidelity](../terms/fidelity.md)
