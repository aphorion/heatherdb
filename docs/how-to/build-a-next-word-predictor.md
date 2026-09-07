# Build a next-word predictor

Suggest the word that follows what someone just typed, learned from their own
text, with no model in the process.

The conventional stack for this is a small language model — a distilled
transformer, or a KenLM-style n-gram table with a smoothing scheme — behind an
inference endpoint. This replaces the table and the serving: the memory holds
the associations, a read completes them, and the only code you own is an
encoder and a vocabulary. It does not replace a language model. What you get
is an n-gram predictor with graceful degradation and no smoothing table.

## The design

**An item is one observed transition** — a context and the word that followed
it. Not a sentence, not a document: transitions are what a predictor is asked
about, so they are what gets written.

**Two items are similar when their contexts are similar.** Word identity is
arbitrary, so use hash-seeded random symbols: unrelated words are near
orthogonal, and two contexts are similar only when they share words. See
[Encode text](encode-text.md), route 2 — the hashed route. An embedding model
would make *synonyms* share predictions, which is a different product.

**A pool is a corpus with one voice** — one author, one document class, one
language. Capacity is about `d/32` patterns per pool
([Choose a dimension](choose-a-dimension.md)), and mixing two writing styles
into one collection buys interference, not transfer.

**The answer resolves through the vocabulary**, not through the engine: the
returned vector's target half is scored against every known word symbol. See
[Resolve a reconstruction to a thing](resolve-reconstructions.md).

**The abstention rule is the norm of the target half.** An unseen context
activates nothing that carries a target, so the second half of the
reconstruction comes back short and directionless — return no suggestion.

## The partition

Lay each vector out as two disjoint halves — `[context | target]` — and write
both halves together, as one vector. Superposition binds them: every location
that absorbs the pattern carries the context in its first 64 components and
the word that followed in its last 64.

```python
import hashlib, json, urllib.request
import numpy as np

D, H = 128, 64                 # 128-dim database, split 64 | 64
DB, COLL = "text", "author"

def call(method, path, body=None):
    req = urllib.request.Request(
        "http://localhost:6380" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

_sym = {}
def symbol(word):
    """A word's symbol: a unit vector of width H, seeded by its hash.

    Deterministic, so the vocabulary is reproducible rather than persisted
    state. Width is H, not D: a symbol occupies one half."""
    w = word.lower().strip(".,!?;:\"'()[]{}—–-…")
    if w not in _sym:
        seed = int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big")
        v = np.random.RandomState(seed).randn(H)
        _sym[w] = v / np.linalg.norm(v)
    return _sym[w]
```

Create the database at 128; the collection appears on first write. Reads
auto-create a collection on a typo, so a misspelled name gives an empty memory
rather than an error — `GET .../stats` is the only route that does not create
one, so a `404` there is how you test that a name exists.

```python
call("POST", "/db", {"name": DB, "dimension": D})
```

## Write the transitions

Sum two context symbols for a trigram. A sum of near-orthogonal symbols is
similar to each of them, so the trigram's context is also partially matched by
either word alone — that is where the graceful degradation comes from, free.

```python
def transition(ctx_words, target):
    """One [context | target] vector, unit length.

    Both halves are normalised before concatenation so neither half can
    dominate the cosine: a long context must not outvote the target."""
    c = sum(symbol(w) for w in ctx_words)
    c = c / np.linalg.norm(c)
    t = symbol(target)
    v = np.concatenate([c, t])
    return (v / np.linalg.norm(v)).tolist()

def learn(words):
    """Write every bigram and trigram in a token stream.

    No metadata: a write without metadata is unretrievable as a document
    but still shapes the memory, which is exactly what transitions are
    for, and it makes the write one batched transaction."""
    vecs = []
    for i in range(len(words) - 1):
        vecs.append(transition([words[i]], words[i + 1]))
    for i in range(len(words) - 2):
        vecs.append(transition(words[i:i + 2], words[i + 2]))
    call("POST", f"/db/{DB}/collections/{COLL}/write", {"vectors": vecs})
    return len(vecs)
```

```json
{ "count": 1994 }
```

Keep the vocabulary as you go — the sidecar here is just the set of words
seen, since the symbols are recomputable from the hash.

## Read the prediction out of the target half

Query with the context half filled and the target half **zeroed**. The engine
activates the locations whose addresses match on the components you supplied;
what comes back in the zeroed half is the superposition of every target those
locations absorbed, weighted by how often each was written.

```python
def predict(ctx_words, vocab, n=5, tau=0.35):
    """Predict the next word. Returns [] when the memory has nothing.

    `tau` is a placeholder — derive it on held-out contexts."""
    c = sum(symbol(w) for w in ctx_words[-2:])
    c = c / np.linalg.norm(c)
    q = np.concatenate([c, np.zeros(H)])       # target half deliberately empty
    q = q / np.linalg.norm(q)
    r = np.array(call("POST", f"/db/{DB}/collections/{COLL}/read",
                      {"query": q.tolist(), "strategy": "iterative"})["result"])
    # The prediction lives in the half the query left blank. Everything the
    # engine put there came from stored transitions, not from the query.
    tgt = r[H:]
    if np.linalg.norm(tgt) < tau:
        return []                              # abstain: nothing completed it
    tgt = tgt / np.linalg.norm(tgt)
    # Resolve against the vocabulary — the reconstruction's neighbours, not
    # the query's. The query has no target half to have neighbours with.
    scored = sorted(((float(tgt @ symbol(w)), w) for w in vocab), reverse=True)
    return [(w, round(s, 3)) for s, w in scored[:n] if s > 0]
```

Backing off from trigram to bigram costs one extra read and recovers contexts
whose two-word form was never seen. There is no smoothing table: the fallback
*is* a second read, merged at half weight because the shorter context is the
less specific evidence.

```python
def predict_backoff(ctx_words, vocab, n=5):
    out = dict(predict(ctx_words, vocab, n=n * 4))
    for w, s in predict(ctx_words[-1:], vocab, n=n * 4):
        out[w] = out.get(w, 0.0) + 0.5 * s
    seen = {w.lower() for w in ctx_words}       # never suggest the input back
    ranked = sorted(out.items(), key=lambda kv: -kv[1])
    return [(w, s) for w, s in ranked if w not in seen][:n]
```

## Sample instead of ranking, for variety

Top-1 from an n-gram memory is deterministic and repetitive. Perturbing the
context before the read moves the query off the attractor it would otherwise
fall straight into, and the memory settles somewhere adjacent.

```python
def sample(ctx_words, vocab, temperature=0.15):
    """Noise on the context half only. The target half must stay exactly
    zero — noise there is a fake prediction the engine completes around."""
    c = sum(symbol(w) for w in ctx_words[-2:])
    c = c / np.linalg.norm(c) + temperature * np.random.randn(H)
    q = np.concatenate([c / np.linalg.norm(c), np.zeros(H)])
    r = np.array(call("POST", f"/db/{DB}/collections/{COLL}/read",
                      {"query": (q / np.linalg.norm(q)).tolist()})["result"])
    return max(vocab, key=lambda w: float(r[H:] @ symbol(w)))
```

## Check it works

1. **Round-trip a memorised transition.** Write a phrase, then predict from
   its first word. The word that followed should rank first.
2. **Check abstention on nonsense.** Predict from two words the corpus never
   contained; the target half should fall below `tau`. If it does not, `tau`
   is too low for this pool — derive it from held-out contexts rather than
   adjusting it by eye ([Calibrate a gate](calibrate-a-gate.md)).
3. **Watch `num_locations`.** `GET /db/text/collections/author/stats` returns
   `num_locations`, `total_writes`, `avg_write_count`. Locations equal to
   writes means nothing is merging: the context encoding is too sparse.

## Tune

| Symptom | Change |
|---|---|
| Predictions blend unrelated words | `d` too low, or too many transitions in one pool — split by author |
| Every context predicts one word | that transition was written far more often; correct for a frequency-weighted memory |
| Abstains on a seen context | `tau` too high, or the trigram sum lost it — try `predict(ctx[-1:])` |
| Suggestions never vary | raise `temperature`, or use `strategy: "fast"` |

Capacity is per pool, about `d/32`. Past that, shard by document class or
raise the dimension — the halves scale with it, `H = d/2`.

## What this is not

It has no syntax, no context beyond the two words you sum in, and no
probability normalisation — the scores are cosines against vocabulary
symbols, not a distribution. It degrades on unseen contexts rather than
failing, as a smoothed n-gram table does, but by superposition instead of a
backoff schedule you fit. For anything that must *compose* meaning rather
than recall a continuation, use a language model and give it a memory as
recall — see [Pair the memory with a language
model](pair-with-an-llm.md).

## Related

- [Encode text](encode-text.md) — the hashed-symbol route used here
- [Choose a dimension](choose-a-dimension.md) — `d/32`, and `H = d/2`
- [Resolve a reconstruction to a thing](resolve-reconstructions.md)
- [Calibrate a gate](calibrate-a-gate.md) — deriving `tau`
- [Write endpoints](../api/writes.md) · [Read endpoints](../api/reads.md)
- [Superposition](../terms/superposition.md) ·
  [Attractor](../terms/attractor.md)

Derived from the `ghost` sample app (with the chain behaviour of `drift` as
the contrast) in
[`aphorion/heatherdb-samples`](https://github.com/aphorion/heatherdb-samples).
