# Pair the memory with a language model

Split the work along its natural seam: the memory does storage, recall and
scoring; the model does language. Anything you can do with a dict, keep in
ordinary code.

| Job | Owner |
|---|---|
| Store, merge, recall | the memory |
| Score confidence, abstain | the memory |
| Rank, filter, exact lookup | ordinary code |
| Read, write, summarise, phrase | the model |

## Resolve first, prompt second

A model cannot read a 128-float array. Resolve the reconstruction to text
before it goes anywhere near a prompt — see [Resolve a reconstruction to a
thing](resolve-reconstructions.md).

```python
r = call("POST", "/db/movies/collections/taste/read", {"query": q})["result"]

# Resolve the RECONSTRUCTION, not the query. The query's neighbours are what
# the caller already had; the reconstruction's neighbours are what the memory
# concluded, and only the second is worth spending prompt tokens on.
hits = call("POST", "/db/movies/collections/films/documents/query",
            {"query": r, "n": 8})["results"]

context = "\n".join("- %s" % h["metadata"]["text"] for h in hits)
prompt = "Recalled context:\n%s\n\nQuestion: %s" % (context, question)
```

## Let fidelity size the context

Fidelity — the cosine between the query and its own reconstruction — is
available on every read at no cost, and it is the right input to two decisions
the model should not be making for itself: how much recalled context to
include, and how much to tell the model to trust it.

```python
fidelity = cos(q, r)

# High fidelity means the query landed on a strong attractor: recall is
# specific, so a few items carry the answer and the model can lean on them.
# Low fidelity means the state travelled a long way: recall is diffuse, so
# widen the context and downgrade the instruction that frames it.
if fidelity >= 0.70:
    n, stance = 3,  "Answer from the recalled context."
elif fidelity >= 0.45:
    n, stance = 8,  "Use the recalled context as evidence; say when it is thin."
else:
    n, stance = 0,  "Nothing relevant was recalled. Say so."
```

Do not inherit those cut points. Derive them on your own collection — see
[Calibrate a gate](calibrate-a-gate.md) — and recalibrate as the pool grows.

## Label recalled items by confidence band

A flat bulleted list tells the model that every line is equally true. Bands
let it discount:

```python
# Bands, not raw numbers. A model handles "uncertain" more reliably than it
# handles 0.42, and banding keeps the prompt stable when scores drift.
def band(s):
    return "certain" if s >= 0.85 else "likely" if s >= 0.60 else "uncertain"

context = "\n".join("- [%s] %s" % (band(h["similarity"]), h["metadata"]["text"])
                    for h in hits)
```

```
- [certain] Solaris (1972), Tarkovsky — slow science fiction
- [likely] Stalker (1979), Tarkovsky
- [uncertain] Interstellar (2014), Nolan
```

## Let the memory abstain

An abstention is an answer. Route it as one rather than handing the model an
empty context and a question it will answer anyway.

```python
if fidelity < TAU:                       # TAU derived, not chosen
    # The memory has no attractor for this. Returning that fact is cheaper
    # and more truthful than a model turn spent guessing from nothing.
    return {"answer": None, "reason": "no confident recall",
            "fidelity": round(fidelity, 3)}
```

The same applies per item: drop hits below the band floor before building the
prompt instead of asking the model to ignore them.

## Chaining reconstructions is not multi-hop reasoning

Feeding a reconstruction back in as the next query does not walk a chain of
inferences. An iterative read has already run the Hopfield loop to
convergence — the state it returns *is* a fixed point, so reading it again
returns approximately itself.

```python
r1 = read(q)
r2 = read(r1)
print(cos(r1, r2))        # ~1.0 — the second read confirms the attractor
```

Check `analyze` if you want to see it directly: `converged: true` with a low
`iterations` on the second call means there was nowhere to go.

For genuine multi-hop, change the query between hops with something outside
the memory:

1. Read, resolve to text, and let the model **name the next question**.
2. Encode that question as a fresh vector.
3. Read again.

The model supplies the move between attractors; the memory supplies each
attractor. Alternatively, use the algebra to construct a query the memory has
not settled on — see [Compose memories](compose-memories.md) — or route across
several collections with `compose/read`.

## Keep the boring parts boring

An associative read is a reconstruction under competition. Ask it for a fact
you can look up and you get a blend of neighbours where you wanted a value.

| Task | Use |
|---|---|
| "What is user 4192's plan tier?" | your database |
| "Which of these 40 results is newest?" | `sorted()` |
| "Is this string in the allow-list?" | a `set` |
| "What has this user liked that is like this?" | a read |
| "Is this pattern one we have seen before?" | a read, plus fidelity |

Forcing a plain table lookup through an associative read adds latency, adds a
blend where you wanted an exact value, and removes the one thing exact lookup
gives you: the guarantee that the answer was stored.

## Related

- [Resolve a reconstruction to a thing](resolve-reconstructions.md)
- [Calibrate a gate](calibrate-a-gate.md) — deriving the fidelity cut points
- [Explain a result](explain-a-result.md) — what to hand back with the answer
- [Compose memories](compose-memories.md) — `compose/read` and routing
- [Fidelity](../terms/fidelity.md) · [Attractor](../terms/attractor.md) ·
  [Hopfield read](../terms/hopfield-read.md)
