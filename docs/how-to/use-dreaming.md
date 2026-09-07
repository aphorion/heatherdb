# Use dreaming

Dreaming is the engine reprocessing its own memory. It is a **write**: it
changes what a collection holds, and in replay mode it changes it in place. This
page is the operational procedure — when to run it, which mode, what moves
afterwards, and the four ways it surprises people.

The mechanism is in [Dreaming](../explanation/dreaming.md) and why it helps at
all is in [Consolidation](../explanation/consolidation.md); the route and its
parameters are in [`POST /db/{db}/dream`](../api/dream.md#post-dbdbdream).

## When to run it

Consolidation is worth running when a collection's **shape** has changed, not
merely when it has grown a little:

- After a bulk ingest, where write order left locations unbalanced.
- After a load whose distribution differs from what came before — a new tenant,
  a new region, a new content type.
- On a schedule matched to your traffic, in a window you choose, rather than
  hoping the idle loop finds a gap.
- Never mid-load. It competes for the same writes and its result reflects a
  half-written collection.

The background loop only fires when the **whole server** has been idle for
`idle_secs`, so a service that is never quiet never dreams. On such a server the
manual trigger is the mechanism, not the fallback.

## Choose a mode

| | Replay | Ladder |
|---|---|---|
| `mode` | `replay` (default) | `ladder` |
| Operates on | the collection itself | a derived stack |
| Effect on the input | rewritten in place | left untouched |
| Produces | fewer or equal locations | `<name>__L1`, `__L2`, … |
| What it needs from your data | nothing | `(address, counter)` read as `(situation, procedure)` |
| Maturity | proven | experimental |

**Replay** re-presents every stored attractor to the write rule, then merges.
Attractors sharpen, splits that the present topology would justify get to fire,
and drifted duplicates fold away.

**Ladder** strips the context off each location — `unbind(counter, address)`
leaves the relation the location carries — and routes those relations by context
through the gated two-field write. Out come `(context prototype, accumulated
law)` locations: the families of the input, one rung up.

## Run it

```bash
# Replay one named collection. Always name the collections.
curl -u admin:pw -X POST \
  'http://localhost:6380/db/movies/dream?mode=replay&collections=taste'
```

```bash
# One ladder rung over an episodic collection.
curl -u admin:pw -X POST \
  'http://localhost:6380/db/movies/dream?mode=ladder&levels=1&collections=episodes'
```

The call is synchronous and answers with what it did, per collection.

## What changes afterwards

**Location count.** The number in `locations_after` is the headline. Replay can
only hold or lower it; a collection that reports no change and `merged: 0` has
nothing left to consolidate. Ladder reports the *rung's* count — how many
families the level found — against `locations_before`, the number of items that
entered it.

Measured on ten episodes carrying two distinct laws over one shared context,
at `D = 256` (`bulk_load`ed, then dreamed on a live engine):

| Call | `locations_before` | `locations_after` | `merged` |
|---|---|---|---|
| Replay | 10 | 9 | 2 |
| Ladder, on the untouched episodes | 10 | **2** | 0 |
| Ladder, after one replay pass | 10 | 3 | 0 |
| Ladder, after two replay passes | 10 | 4 | 0 |

Two is the right answer: two laws were written. Note that `merged` and the drop
in count are different quantities — a merge can be offset by a split in the same
pass.

**Fidelity.** Replay moves addresses toward the centre of what they represent,
so reads of in-distribution queries land on sharper attractors. Sample
[fidelity](../terms/fidelity.md) over a fixed query set before and after, and
keep the pair; it is the only direct evidence that a pass helped rather than
merely shrank the index.

**Schema collections.** Ladder mode creates `<name>__L1`, and with `levels > 1`
climbs `__L1` into `__L2` and so on, stopping when a level has fewer than two
locations. Each rung **clears its destination first**, so a rung is
reproducible and never accumulates across runs. Collections whose names contain
`__L` are skipped as inputs, so the ladder does not climb its own output. The
per-rung `tau_split` and `tau_cohere` in the report are the boundaries the rung
actually used — auto-calibrated from that rung's own inputs when you leave them
unset, so they will differ level to level and run to run.

## Four traps

### A manual trigger ignores the opt-in

`POST /db/{db}/dream` sets `enabled = true` for the duration of the call.
Confirmed against a database whose `db.toml` carries `[dream] enabled = false`:
the trigger ran and the collection went from 10 locations to 9. The `[dream]`
block gates the **background loop**, not the route. Treat the route as a write
endpoint that any caller with database scope can fire.

### `?mode=` fails open to replay

Mode parsing accepts `ladder` (case-insensitively) and reads **anything else**
as replay — including a typo. Confirmed live: `?mode=laddr` ran a replay pass
and mutated the collection in place, reporting `merged: 4`, with no error and no
warning. There is no way to tell from the response that the mode you asked for
was not the mode you got, except that a ladder run names a `__L` collection in
its report and a replay run names the input. Check that.

### Omitting `?collections=` selects everything

An empty collection list means every collection in the database. On a manual
trigger with no parameters that is a replay pass over the whole database, which
on a large one is one write per stored location plus a merge, inside a
synchronous request bounded by `--request-timeout`. Always pass
`?collections=`.

### Replay before ladder changes the answer

Replay rewrites the episodes the ladder would have read. In the table above,
laddering untouched episodes recovers the two true families; the same episodes
after one replay pass yield three, after two, four. Replay is not order-neutral
with respect to a ladder rung, and the input it consumed is gone.

So, before the first ladder run on real data, take a snapshot. It is safe with
the engine running — a live LMDB copy:

```bash
heather --data-dir /var/lib/heatherdb snapshot create --db movies
```

Then run the rung on a named collection and compare the family count against
what you believe the data contains. A rung that returns as many families as it
was given items has found no structure; one that returns a single family has
merged everything.

## Verify a pass

```bash
# The report is the primary evidence. Keep it.
curl -u admin:pw -X POST \
  'http://localhost:6380/db/movies/dream?mode=replay&collections=taste' | tee dream.json
```

Three checks, in order:

1. **Did it act?** `locations_after` differs, or `merged > 0`.
2. **Is recall intact?** Re-read a fixed sample of queries and compare fidelity
   to the pre-dream sample. Replay must not lose attractors; a drop means the
   pass merged things you needed apart, and the snapshot is how you get them
   back.
3. **For a ladder rung, is the family count plausible?** Against your own
   estimate of how many distinct laws the input carries, not against a default.

## Related

- [Dreaming](../explanation/dreaming.md) — the write rule being replayed, and
  the idle loop's conditions.
- [Consolidation](../explanation/consolidation.md) — why replay changes what a
  memory holds.
- [Dream endpoint](../api/dream.md) — parameters, status codes, timeout.
- [Backup and restore](backup-and-restore.md) — snapshots before an
  experimental pass.
- [Tune a database](tune-a-database.md) · [Configuration:
  `[dream]`](../reference/configuration.md#dream)
