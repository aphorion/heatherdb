# Dreaming

When the engine has been idle for a while, it can reprocess its own memory: no
external input, no second collection, nothing to ingest. A collection simply
wakes up better organised.

The biological name for this is hippocampal replay, and the mechanism is the
same one HeatherDB uses to learn — the write rule, run against what the memory
already holds.

## Why it helps

A collection's structure depends on the order it was written in. Early writes
established locations against a sparse index; later writes arrived when the
neighbourhood was already crowded. Nothing rebalances that during normal
operation, because a write only touches the *k* locations it activated.

Replay re-presents every stored attractor to the write rule as if it were new.
Three things follow:

- **Attractors sharpen.** Addresses migrate toward the centre of what they
  actually represent.
- **Splits reorganise.** Novelty and overload splits that would have fired
  under the present topology get to fire.
- **Redundancy folds away.** The subsequent merge pass dedups locations that
  have drifted together and enforces capacity.

Consolidation costs nothing at query time and cannot happen usefully during
load, so it belongs in the idle window.

## The idle loop

Off by default. Opt in per database, in `db.toml`:

```toml
[dream]
enabled = true
idle_secs = 60
passes = 1
collections = []       # empty means every collection
```

The loop wakes every 15 seconds. It runs a pass over a database only when:

- that database has `dream.enabled`, **and**
- the **whole server** has been idle for `idle_secs` — every request stamps a
  last-activity clock, so this is server-wide inactivity, not per database, and
- the collection's location count has changed since its last dream.

That last condition is what stops a quiescent collection from being dreamed
every tick: after a pass, the settled count is recorded, and an unchanged
collection is skipped until something writes to it again.

Consequence worth knowing: on a server that is never fully idle for
`idle_secs`, dreaming never runs. Trigger it manually instead.

## Triggering it by hand

```bash
curl -u admin:pw -X POST 'http://localhost:6380/db/movies/dream'
```

This bypasses both the idle gate and the `enabled` opt-in — a manual trigger
should not require configuration. Query parameters override the persisted
config for that call:

```bash
curl -u admin:pw -X POST \
  'http://localhost:6380/db/movies/dream?collections=taste,history&levels=2'
```

```json
{"database":"movies",
 "collections":[{"collection":"taste","passes":1,
                 "locations_before":812,"locations_after":806,"merged":6,
                 "tau_split":0.0,"tau_cohere":0.0}]}
```

Useful in a maintenance window, or from a scheduler that knows your traffic
pattern better than an idle timer does.

## Replay mode

The default. `passes = 1` is pass 0 alone, and it is the proven, safe
behaviour: every location's address is written back through the write rule,
then `merge()` dedups and enforces capacity. In place; no new collections.

`passes > 1` extends this to coarser granularity. On pass `g ≥ 1` the operands
are no longer individual locations but **cluster centroids**: locations are
greedily clustered by cosine at a threshold that loosens with depth
(0.7, 0.5, 0.3, …, floored at 0.1), and the centroids are replayed. This grows
coarse super-attractors alongside the fine ones — the substrate stays flat, and
organisation emerges from replaying at increasing operand granularity.

That is the experimental frontier. Leave `passes` at 1 unless you are
deliberately exploring it.

## Ladder mode

`mode = "ladder"` is a different operation with the same trigger, and it does
not reorganise anything in place.

Read a collection as a set of `(situation, procedure)` episodes. Ladder mode
consolidates them — through the gated two-field write — into a `<name>__L1`
collection of `(context prototype, law)` pairs, then `__L1` into `__L2`, and so
on for `passes` levels. Collections whose names contain `__L` are skipped as
inputs, so the ladder never climbs its own output.

The gate is what makes this work. In an ordinary write the winner is the
closest address; under the gate, among candidates close enough in *address* to
share a context (`sim ≥ tau_split`), the winner is the one whose accumulated
*counter* best coheres with the incoming counter. If nothing coheres above
`tau_cohere`, the write spawns a new family instead of joining a wrong one.

Two same-context-different-law episodes therefore land in separate families
rather than averaging into a law that is true of neither. A gated join is also
winner-take-all and accumulates at full weight, keeping each family's law pure,
and its address converges to a running *mean* of its members' routing vectors
so the prototype does not stay stuck near whichever member arrived first.

When `tau_split` and `tau_cohere` are left at 0.0, each rung auto-calibrates its
own boundaries from its own inputs: address candidacy from the largest gap in
the upper region of the pairwise similarities, and the coherence gate from
Otsu's threshold — the valley between the within-family and cross-family modes.
Self-scaling, so the ladder works at any depth without a constant to tune.

This is the machinery that derives schemas, and laws about laws, from raw
episodes. It is the most experimental thing in the engine.

## What to watch

`locations_after` versus `locations_before` in the report tells you whether a
pass did anything. A collection that consistently reports `merged: 0` and no
change in count has nothing left to consolidate.

Dreaming mutates a collection. It is a write, and it is not free — replay costs
one write per stored location, plus a merge pass. On a large collection, that
is real work, which is precisely why it is gated on idleness.

If you want compaction without replay, `POST …/collections/{name}/compress` is
the narrower tool: it merges locations while doing so lowers description
length, and stops at the MDL minimum. It does not re-present anything to the
write rule.

## Related

- [How the memory works](associative-memory.md) — the write rule being replayed.
- [Configuration: `[dream]`](../reference/configuration.md#dream) — every knob.
- [HTTP API: `POST /db/{db}/dream`](../api/dream.md#post-dbdbdream).
