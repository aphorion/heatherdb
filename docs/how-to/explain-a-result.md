# Explain a result

Attribute an answer to what produced it, using the read path itself. No second
model, no surrogate, no post-hoc explainer.

Three techniques, in increasing cost: the query/result diff, location-overlap
attribution through `analyze`, and per-criterion scores from `role_pairs`.

## 1. Diff what you sent against what came back

A read returns a reconstruction, not your query. Wherever the two differ, the
memory added something — and that addition is the finding.

```python
q = build_query(...)
r = call("POST", "/db/movies/collections/taste/read", {"query": q})["result"]

# Component-wise difference on unit-normalised vectors. Large positive
# entries are dimensions the reconstruction asserts and the query did not:
# features the memory supplied from what it holds, not from what you asked.
q, r = unit(q), unit(r)
delta = [ri - qi for qi, ri in zip(q, r)]

# Rank the added features. With a named basis (one dimension per attribute,
# or a role/filler vocabulary) these are directly readable.
top = sorted(range(len(delta)), key=lambda i: -delta[i])[:8]
print([(FEATURE_NAMES[i], round(delta[i], 3)) for i in top])
```

```
[('subtitled', 0.21), ('slow_pacing', 0.19), ('1970s', 0.17), …]
```

With an opaque embedding basis the raw components mean nothing; resolve the
difference vector instead — hand `delta` to your sidecar cache or to
`documents/query` and read off which stored items it points at. See
[Resolve a reconstruction to a thing](resolve-reconstructions.md).

The diff answers "what did the memory add?". Two useful degenerate cases:
a near-zero diff means the query was already on an attractor and the read told
you nothing new, and a diff larger than the query means the reconstruction has
travelled — check
[fidelity](../terms/fidelity.md) before reading it as a finding.

## 2. Attribute through activated locations

`analyze` is `read` with its activation trace attached:

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/analyze \
  -H 'Content-Type: application/json' -d '{"query":[…]}'
```
```json
{"iterations": 4, "converged": true,
 "activated_locations": [{"id": 7, "similarity": 0.91, "weight": 0.63},
                         {"id": 22, "similarity": 0.78, "weight": 0.21}],
 "result": [ … ]}
```

`iterations` and `converged` describe the settling; `activated_locations`
names the hard locations that carried the answer, each with its cosine to the
query state and its softmax weight in the blend.

To attribute an answer to the stored items that produced it, analyze the
answer *and* each candidate cause, then intersect their activated sets:

```python
def trace(v):
    a = call("POST", "/db/movies/collections/taste/analyze", {"query": v})
    # id -> weight. Weight is the softmax share, so it already accounts for
    # how much each location contributed rather than merely being present.
    return {loc["id"]: loc["weight"] for loc in a["activated_locations"]}

answer = trace(r)

def explanation_weight(candidate_vec):
    cand = trace(candidate_vec)
    # Shared locations only. The score is how much of the ANSWER's mass sits
    # on regions this candidate also drives — an overlap of causes, not of
    # appearances.
    shared = set(answer) & set(cand)
    return sum(answer[i] * cand[i] for i in shared)

ranked = sorted(((explanation_weight(v), k) for k, v in candidates.items()),
                reverse=True)
```

```
0.41  Stalker
0.33  Andrei Rublev
0.04  Interstellar
```

**Location overlap is not embedding similarity.** The activated set is where
competitive learning placed each item, so two items with only moderate cosine
between them can carry high explanation weight — they were absorbed into the
same engrams and therefore genuinely produce each other's answers. The
converse also holds: a high-cosine item that landed in a different region
contributes nothing, and the attribution correctly says so.

Use `batch_analyze` when there are many candidates; it runs them in parallel
in one request, in request order, and skips the audit log.

## 3. Per-criterion scores from `role_pairs`

When documents are built from bound role/filler pairs, ask for each criterion
to be scored on its own and render the answer as a sentence:

```python
hits = call("POST", "/db/movies/collections/films/documents/query", {
    "query": q,
    # Over-fetch. `role_pairs` adds numbers to the hits; it does NOT steer
    # recall — retrieval is always full-bundle cosine on `query` — so the
    # document you want must survive ranking before you can re-rank it.
    "n": 50,
    "role_pairs": [{"role": GENRE,    "filler": scifi},
                   {"role": DIRECTOR, "filler": nolan}]})["results"]

ROLES = ["genre", "director"]           # request order; scores come back in it

for h in hits[:5]:
    s = h["role_scores"]
    # Scores are unweighted and unaggregated by design. Split them at zero to
    # get a plain matched/not-matched reading per criterion.
    on  = [ROLES[i] for i, v in enumerate(s) if v > 0]
    off = [ROLES[i] for i, v in enumerate(s) if v <= 0]
    print("%s: matched on %s, not on %s"
          % (h["metadata"]["title"], ", ".join(on) or "nothing",
             ", ".join(off) or "nothing"))
```

```
Interstellar: matched on genre, not on director
Solaris: matched on genre, not on director
```

Role scores are small numbers — unbinding recovers the filler plus crosstalk
from every other role in the bundle. Read them as relative signal between
candidates, and as sign rather than magnitude when rendering them as prose.

## Which technique for which question

| Question | Technique |
|---|---|
| What did the memory add to my query? | the diff |
| Which stored items produced this answer? | `analyze` + location overlap |
| Which criteria did this hit satisfy? | `role_pairs` |
| Should I trust the answer at all? | fidelity — see [Calibrate a gate](calibrate-a-gate.md) |

## Related

- [Resolve a reconstruction to a thing](resolve-reconstructions.md)
- [Store and query structured documents](structured-documents.md)
- [Read endpoints](../api/reads.md) — `analyze`, `batch_analyze`
- [Associative memory](../explanation/associative-memory.md) · [What similarity
  means](../explanation/what-similarity-means.md)
