# Build a memory for an agent

Give an agent recall of everything it has said and been told, across sessions,
without carrying it all in the context window.

The conventional stack for this is retrieval-augmented generation: chunk the
transcript, embed it, put it in a vector index, retrieve top-k by cosine, and
paste the chunks into the prompt. This replaces the index and the retrieval
step. The memory superimposes turns rather than filing them, so a partial cue
completes into a pattern, repeated topics recall more strongly than one-off
mentions, and the read hands you a confidence number that the retrieval step
never had. The model still does the language. That division of labour is
[Pair the memory with a language model](pair-with-an-llm.md); this page is the
build.

## The design

**An item is one durable statement** — a user turn, a stated fact, a decision
and its reason. Not a whole session, not a raw assistant paragraph. Write a
compact form: what the user said plus what was concluded, in one or two
sentences.

**Two items are similar when they are about the same thing** — an embedding
model ([Encode text](encode-text.md), route 1) with the corpus mean subtracted
([Center your vectors](center-your-vectors.md)). Uncentred output scores every
turn near 1.0 against every other, and the confidence number this design rests
on stops carrying signal.

**A pool is one conversation, or one agent.** Capacity is about `d/32` per
pool, so a shared collection both leaks and interferes — one user's turns
become another's recall. One collection per conversation for a chat
assistant, one per agent for a persistent role; route by path,
`/db/agents/collections/{conversation_id}/...`.

**The answer resolves to text before it reaches a prompt.** A read returns a
vector; a model cannot read one. Keep a sidecar of turn → vector and resolve
the reconstruction against it — see
[Resolve a reconstruction to a thing](resolve-reconstructions.md).

**The abstention rule is a fidelity floor.** When the memory has no attractor
for the current message, it says so, and the agent proceeds with recent
context only rather than being handed weak recall it will treat as fact.

## What to write, and what not to

| Write | Do not write |
|---|---|
| User statements of fact and preference | Every assistant paragraph verbatim |
| Decisions, with the reason attached | Tool output and raw payloads |
| Corrections ("actually, it's the other X") | Boilerplate turns — greetings, acks |
| A one-line summary of each exchange | Anything you can look up exactly |

The last row matters most. An associative read returns a blend under
competition; a fact you can key on belongs in a table. Writing everything also
spends capacity: at `d = 384`, a pool holds on the order of a dozen
well-separated patterns before superposition starts blending them, so what you
choose *not* to write is the tuning knob you use first.

Every write here carries metadata, so each turn is also a retrievable
document. A write without metadata still shapes the memory but is
unretrievable as a document, and there is no backfill — decide before the
first write.

```python
import json, sqlite3, struct, urllib.request
import numpy as np

DB = "agents"

def call(method, path, body=None):
    req = urllib.request.Request(
        "http://localhost:6380" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0

side = sqlite3.connect("memory.db")
side.execute("CREATE TABLE IF NOT EXISTS turns (conv TEXT, ts REAL, "
             "text TEXT, vec BLOB)")

def remember(conv, text, ts):
    """Write one turn to its conversation's collection and to the sidecar.

    The metadata rides along so the turn is retrievable as a document; the
    sidecar copy holds the exact vector, which is what resolution scores
    against. Storing a re-encoded vector later would resolve to noise."""
    v = embed(text)                                   # centred, unit-length
    call("POST", f"/db/{DB}/collections/{conv}/write",
         {"vectors": [list(v)], "metadata": [{"text": text, "ts": ts}]})
    side.execute("INSERT INTO turns VALUES (?,?,?,?)",
                 (conv, ts, text, struct.pack(f"{len(v)}f", *v)))
    side.commit()
```

```json
{ "count": 1, "ids": [43] }
```

## Recall, and the number that comes with it

One read per user message. Compute the fidelity yourself — there is no such
field on the wire — then resolve the reconstruction, not the query.

```python
def recall(conv, message, n=5):
    """Returns (fidelity, [(text, confidence), ...]).

    Fidelity is cos(query, reconstruction): how far the state had to travel
    from what was asked to what the memory settled on. High means the topic
    already has a basin here; low means this is new ground."""
    q = embed(message)
    r = call("POST", f"/db/{DB}/collections/{conv}/read",
             {"query": list(q), "strategy": "iterative"})["result"]
    f = cos(q, r)

    # Resolve the RECONSTRUCTION. The query's nearest turns are the ones
    # this message already resembles — which the agent has in front of it.
    # The reconstruction's nearest turns are what the memory concluded,
    # including turns that are not top-1 similar to the message itself.
    rows = side.execute("SELECT text, vec FROM turns WHERE conv=?", (conv,))
    scored = []
    for text, blob in rows:
        cv = np.array(struct.unpack(f"{len(blob)//4}f", blob))
        scored.append((cos(r, cv), text))
    scored.sort(reverse=True)
    return f, [(t, s) for s, t in scored[:n]]
```

Because every turn was written with metadata, `documents/query` can run that
scan inside the engine instead — cheaper, but it only considers turns on the
activated locations. The trade is tabulated in
[Resolve a reconstruction to a thing](resolve-reconstructions.md).

## Let fidelity size and label the recall

Fidelity decides two things the model should not decide for itself: how much
recalled context to include, and how much to trust it. Bands, not raw numbers
— a model handles "uncertain" more reliably than it handles `0.42`, and bands
keep the prompt stable as scores drift.

```python
def band(s):
    """Confidence label for one recalled turn. Cut points are illustrative:
    derive them on your own collection — see calibrate-a-gate.md."""
    return "vivid" if s >= 0.85 else "clear" if s >= 0.70 else "vague"

def build_context(conv, message):
    f, mems = recall(conv, message)

    if f < TAU:                      # TAU derived from held-out messages
        # Abstain. Handing the model an empty context and a question it
        # will answer anyway is how a memory system produces confident
        # fabrication; returning "nothing recalled" is an answer.
        return None, f

    # Strong recall is specific: a few turns carry it. Weak recall is
    # diffuse: widen the window and downgrade the framing around it.
    k = 3 if f >= 0.70 else 8
    lines = "\n".join("- [%s] %s" % (band(s), t) for t, s in mems[:k]
                      if s >= 0.50)
    return lines, f
```

```
- [vivid] User is migrating the billing service off Stripe; deadline is Q3.
- [clear] Decided against a queue in front of the webhook — retries suffice.
- [vague] User mentioned a staging environment in eu-west.
```

Hand that to the model as clearly-delimited recall, with the fidelity
attached, and let it write the reply. The prompt shape is in
[Pair the memory with a language model](pair-with-an-llm.md).

## Chaining recalls does not give you multi-hop

Feeding a reconstruction back in as the next query does not walk a chain of
inferences. An iterative read runs the Hopfield loop to convergence, so the
state it returns is already a fixed point: reading it again returns
approximately itself and confirms the attractor rather than moving to a new
one.

```python
r2 = read(r1)            # r1 was itself a reconstruction
print(cos(r1, r2))       # ~1.0 — no new ground was covered
```

For an agent this is the difference between a memory and a reasoner. Genuine
multi-hop needs something outside the memory to change the query between hops:
recall, resolve to text, let the model name the next question, encode that,
and read again. The model supplies the move between attractors; the memory
supplies each attractor. `strategy: "fast"` leaves a state that has not
settled, which is useful for exploration, but it is drift, not inference.

## Check it works

1. **Round-trip a turn.** Write a statement, then recall with a paraphrase of
   it. It should come back in the `vivid` or `clear` band. If it does not,
   check the encoder is centred before suspecting the memory.
2. **Check abstention.** Recall on a topic the conversation never touched:
   fidelity should fall below `TAU`. Derive `TAU` on held-out messages —
   [Calibrate a gate](calibrate-a-gate.md) — and re-derive it as the pool
   grows; a gate is valid only at the load it was calibrated at.
3. **Verify the collection is the one you think.** Reads and writes
   auto-create a collection, so a typo in a conversation id silently gives a
   fresh empty memory that answers everything with low fidelity.
   `GET /db/agents/collections/{conv}/stats` is the only route that does not
   create, and returns `num_locations`, `total_writes`, `current_eta`,
   `avg_write_count`, `max_write_count`. A `404` means it does not exist.
4. **Watch `avg_write_count`.** Repeatedly discussed topics drive the same
   locations repeatedly, which is what makes them recall strongly. A flat
   value near 1.0 across a long conversation means nothing is accumulating —
   usually turns written verbatim rather than summarised.

## Tune

| Symptom | Change |
|---|---|
| Recall blends turns into a plausible non-turn | pool over capacity (`≈ d/32`) — write less, or split per session |
| Everything recalls at ~0.95 | `d` too high for the turn count, or vectors uncentred |
| Never abstains | `TAU` derived on a smaller pool than the current one — recalibrate |
| Cites a superseded decision | write the correction; superposition does not come apart, so nothing is removable |
| Cross-user leakage | a shared collection — one per conversation, and check the id |

## Related

- [Pair the memory with a language model](pair-with-an-llm.md) — the division
  of labour and the prompt shape
- [Resolve a reconstruction to a thing](resolve-reconstructions.md)
- [Calibrate a gate](calibrate-a-gate.md) — deriving `TAU` and the bands
- [Encode text](encode-text.md) · [Center your vectors](center-your-vectors.md)
  · [Choose a dimension](choose-a-dimension.md)
- [Structured documents](structured-documents.md) ·
  [Multiple databases](multi-database.md) · [Fidelity](../terms/fidelity.md) ·
  [Attractor](../terms/attractor.md)

Derived from the `recall`, `memoria` and `swarm` sample apps in
[`aphorion/heatherdb-samples`](https://github.com/aphorion/heatherdb-samples).
