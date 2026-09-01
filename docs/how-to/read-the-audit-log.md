# Read the audit log

Every database keeps a persisted access log of who read what, when, and what
came back. It is **on by default** — a security control, not a feature you have
to discover.

## What is recorded

Reads only, on four routes: `read`, `attention`, `analyze` and
`documents/query`. Writes, algebra, database management and `batch_analyze` are
not recorded.

Each entry carries the username and scope at request time, the database and
collection, the logical route name, the HTTP status, how many results came
back, the ids that were returned, and a 64-bit hash of the query vector.

The returned ids are the point. They are what makes the log answer *who has
seen this document*, not merely *who ran a search*.

## Query it

```bash
curl -u admin:pw 'http://localhost:6380/db/movies/audit?limit=50'
```
```json
{"database":"movies","count":2,"limit":50,
 "entries":[
   {"seq":41,"timestamp_ms":1788260246123,"user":"alice","scope":"db:movies",
    "database":"movies","collection":"taste","route":"read","status":200,
    "result_count":1,"location_ids":[],"document_ids":[],
    "query_hash":"9f2c1ab30de4f781","query_raw":null}]}
```

Newest first. Filters:

| Param | Meaning |
|---|---|
| `user` | exact username |
| `collection` | exact collection name |
| `route` | `read`, `attention`, `analyze`, `documents/query` |
| `since` | inclusive lower bound, unix **milliseconds** |
| `until` | inclusive upper bound, unix milliseconds |
| `limit` | default 100, clamped to `1..=1000` |

```bash
# Everything alice read from the `taste` collection in the last hour.
SINCE=$(( ($(date +%s) - 3600) * 1000 ))
curl -u admin:pw "http://localhost:6380/db/movies/audit?user=alice&collection=taste&since=$SINCE"
```

```bash
# Who has seen document 4211?
curl -u admin:pw 'http://localhost:6380/db/movies/audit?route=documents/query&limit=1000' \
  | python3 -c 'import json,sys
for e in json.load(sys.stdin)["entries"]:
    if 4211 in e["document_ids"]: print(e["timestamp_ms"], e["user"])'
```

There is no filter on returned ids — that one is client-side.

## Page backwards through it

There is no cursor. Page with `until`: take the oldest `timestamp_ms` you got,
subtract one, ask again.

```bash
curl -u admin:pw 'http://localhost:6380/db/movies/audit?limit=1000&until=1788260246122'
```

A filtered scan also gives up after 200 000 examined rows, so narrow the time
window rather than filtering across the whole history.

## Who is allowed to read it

Root scope only, by default. A `Database("movies")` credential gets `403` on
`/db/movies/audit`.

That is deliberate. The log is a record of what *every* user of a database
searched for, and `Scope::Database` is a shared data scope — in a
knowledge-repository deployment it is the ordinary employee scope. Granting it
by default would let any employee read every colleague's search history.

### Let a database's own users read it

Per database, in `db.toml`:

```toml
[audit]
visibility = "DbUsers"
```

Restart. A `Database("movies")` user can now read `/db/movies/audit` — and
still never another database's.

This is one binary switch, not a role system. There is no auditor role, and no
way to express "read your own entries but not your colleagues'".

## Configure retention and buffering

All per database, in `db.toml`:

```toml
[audit]
enabled = true              # off silently stops recording
max_entries = 1000000       # oldest pruned past this
retention_days = 90         # 0 disables the age bound
flush_threshold = 256       # buffered records that force a flush
flush_interval_secs = 2     # periodic flush cadence
max_buffer = 16384          # hard memory cap; records past it are dropped and counted
store_raw_query = false     # see below
visibility = "Root"         # or "DbUsers"
```

`GET /db/{db}/audit` returns `409` when `enabled = false` for that database,
rather than an empty list — so a disabled log cannot be mistaken for a quiet one.

## Storing raw query vectors

By default a record carries only a hash of the query vector. Turn that off with
care:

```toml
[audit]
store_raw_query = true
```

Two costs. **Size**: at d=4096 a query is about 32 KB of `f64`, so a million
reads is 32 GB of log auditing data that may be smaller than the log. And
**sensitivity**: a query vector is a reconstructable statement of what someone
was looking for — run it back through `attention` and it names the documents.
Storing it turns the audit log into a second, unguarded copy of every search
anyone has made.

The hash keeps the property the log actually needs — repeated identical queries
correlate — while being one-way. See
[The audit log and privacy](../explanation/audit-and-privacy.md).

## Know the loss window

Records are buffered in memory and flushed in batches, because LMDB has one
writer and a transaction per read would serialise the read path. An unclean
shutdown (SIGKILL, power loss) loses whatever is still buffered — at most
`flush_threshold` records or `flush_interval_secs` of activity.

A graceful shutdown flushes, so the window only opens on a hard kill. Stop the
engine with SIGTERM or Ctrl-C.
