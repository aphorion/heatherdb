# The audit log and privacy

Every database keeps a persisted record of who read what, when, and what came
back. This page is about the three design decisions behind it, each of which
trades something away deliberately.

## What it is, and what it is not

The audit log is a generic access-log primitive, comparable in scope to
Postgres's `pgaudit` or MongoDB's audit log. It records raw events — who, which
route, against which object, when, with what outcome — and stays agnostic about
what those events mean.

It does not decide what counts as "trending" or "relevant", and it builds no
rollup aimed at a particular UI panel. A consumer that wants that queries the
raw log through `GET /db/{db}/audit`, with its user, collection, route and time
filters, and aggregates it for its own purposes.

Keeping that logic out of the engine is what makes the primitive usable by any
application built on HeatherDB, rather than only the one that motivated it.

Records are keyed `[timestamp_ms: 8B BE | seq: 8B BE]`, so LMDB's own key order
*is* time order. Newest-first paging is a reverse cursor walk from the end of
the tree, and a time window is a prefix-bounded scan rather than a full table
read. `seq` breaks ties within a millisecond and keeps keys unique.

## Decision 1: the query vector is hashed

By default a record carries a 64-bit hash of the query vector, never the vector
itself. Two reasons, both load-bearing.

**Size.** At d=4096 a query is about 32 KB of `f64`. A repository doing a
million reads would spend 32 GB auditing reads against data that is itself
smaller than the log. The log would dwarf what it audits.

**Sensitivity.** A query vector is a reconstructable statement of what someone
was looking for. Run it back through `attention` and it names the documents.
Storing it turns the audit log into a second, unguarded copy of every search
anyone has ever made over confidential personnel, legal and financial material.

The hash keeps the property the log actually needs — repeated identical queries
correlate, so you can see that someone ran the same search twice — while being
one-way.

This is a default, not a mandate. `store_raw_query = true` lets an operator who
wants raw vectors for their own analysis opt in per database. The engine picks
the safe default and gets out of the way.

What is *always* recorded is the returned identifiers: `location_ids` and
`document_ids`. That is what makes the log answer **who has seen this
document**, not merely **who ran a search** — which is the question an
investigation actually asks, and it survives the query vector being hashed.

## Decision 2: writes buffer, and a hard kill loses some

Recording a read means writing on the read path, and LMDB has exactly one
writer. Opening a write transaction per read would funnel every concurrent
reader through that single lock, converting a lock-free MVCC read path into a
serialised one.

So `record` takes a mutex, pushes onto a `Vec`, and returns. No transaction, no
I/O, no `fsync`. A separate `flush` drains the buffer *outside* the lock and
writes the whole batch in one transaction, on two triggers: a size threshold
(`flush_threshold`, default 256, reported back by `record`'s return value) and a
periodic tick the server owns (`flush_interval_secs`, default 2).

Because LMDB readers never block on a writer, that batch write is invisible to
concurrent reads. It contends only with other *writes*, and one transaction per
few hundred events instead of one per event makes that contention negligible.

**The trade-off, stated plainly:** an unclean shutdown — SIGKILL, power loss —
loses whatever is still buffered, at most `flush_threshold` records or
`flush_interval_secs` of activity.

That is accepted deliberately. Losing the last second of the access log
degrades an audit trail slightly; serialising every read behind the writer lock
degrades the product. A graceful shutdown flushes every database's staged
records, so the window only opens on a hard kill.

A hard `max_buffer` cap (default 16 384) bounds memory if the flusher ever
stalls: records past the cap are dropped and counted rather than growing the
heap without limit.

## Decision 3: root-only by default

`GET /db/{db}/audit` requires root scope. A `Database("movies")` credential gets
`403` on `/db/movies/audit`, even though it can read every collection in that
database.

The reasoning: the log is a record of what *every* user of a database searched
for, and `Scope::Database` is a data scope shared by every user of that
database. In a knowledge-repository deployment, that is the ordinary employee
scope. Granting it by default would let any employee read every colleague's
search history over confidential material — and searches over a bid repository
leak commercial intent on their own, without any document being opened.

An operator who wants a database's own users to see their own database's log —
and nothing about other databases — opts in per database:

```toml
[audit]
visibility = "DbUsers"
```

The authorisation function takes that decision as a parameter rather than
hard-coding it, because it has no way to look up per-database config itself.
The auth middleware is the only place that sees both the user's scope and the
database's config, so that is where the resolution happens.

**This is one binary policy switch, not a role system.** The `Scope` enum has
two levels and no per-user or role dimension. It cannot express "read your own
entries but not your colleagues'", and there is no dedicated auditor role. If
you need that, it belongs in a layer above the engine.

## Coverage, honestly

Recorded: `read`, `attention`, `analyze`, `documents/query` — the four routes
that return content.

Not recorded: writes, `bulk_load`, all algebra, `compress`, database creation
and deletion, user management, token minting, and `batch_analyze`. If you need
an audit trail of *mutations*, this is not it; the log answers who saw what,
not who changed what.

When auth is disabled the user and scope are recorded as `-`. A username can
never be `-` — the validator demands a leading letter — so the placeholder
cannot collide with a real one, and it is a value you can filter on.

`enabled = false` silently stops recording. To make that harder to do by
accident, `GET /db/{db}/audit` returns `409` rather than an empty list when a
database has logging off, so a disabled log cannot be mistaken for a quiet one.

## Related

- [Read the audit log](../how-to/read-the-audit-log.md) — querying, paging,
  retention, visibility.
- [HTTP API: `GET /db/{db}/audit`](../api/audit.md#get-dbdbaudit).
