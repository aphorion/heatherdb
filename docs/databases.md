# Databases (multi-tenancy)

A HeatherDB **server** hosts many **databases**. Each database has its
own vector dimension, its own EAM configuration, its own LMDB
environment on disk, and its own collection namespace. Cross-database
isolation is provided by the OS — separate envs, separate file
handles, separate locks.

```
heather_server (one process, one root data dir)
├── default                (dim=128, what auto-creates on first boot)
│   ├── collections
│   │   ├── users
│   │   ├── events
│   │   └── …
│   └── algebra ops
├── memoria                (dim=384, your conversational app)
│   └── collections
└── navigator              (dim=64, your robot danger memory)
    └── collections
```

## Why split

A new database is the right answer when…

- **You need a different vector dimension.** Dimension is per-database
  and immutable after first write — sentence-transformers (384) and
  the MovieLens encoder (128) and an FFT mic (128) all want their
  own.
- **You want hard isolation between projects.** Different EAM tuning
  knobs, different backup cadences, different snapshot diffs.
- **You want per-project credentials.** Scope a user to
  `Database("foo")` and they can never touch `bar`. See [auth](./auth.md).
- **You want to drop a project cleanly.** `DELETE /db/{name}` moves
  the directory to `_trash/<name>-<ts>/`; no leaked collections, no
  leaked attractors, no orphan locks.

A new database is the **wrong** answer when…

- You want a per-user split. Make a collection per user inside one
  shared database — collections are cheap. Databases are heavier
  (separate LMDB env per).
- You think you'll save memory. LMDB envs are cheap to keep open
  but each is its own mmap region; ~30 databases is fine, 300 is
  pushing it on a small VPS.

## Names

Database names follow the same rules as Postgres identifiers,
slightly tighter:

- 1–63 ASCII chars
- starts with a lowercase letter
- rest: `[a-z0-9_-]`
- must not start with `_` (reserved for system / `_trash`)

Reserved name: `default` — this is the auto-created DB and the target
of every legacy non-`/db/` route. You cannot drop it via the HTTP API
(refused with 403); to wipe it, stop the engine and delete
`$HEATHER_DATA_DIR/db/default/`.

## On disk

```
$HEATHER_DATA_DIR/
├── server.toml             # marker: "this is a v0.2 root"
├── users.json              # auth credentials
├── db/
│   ├── default/
│   │   ├── db.toml         # per-DB config (dimension, EAM knobs)
│   │   └── data/
│   │       ├── data.mdb    # LMDB env (5 named sub-DBs inside)
│   │       └── lock.mdb
│   ├── memoria/
│   │   ├── db.toml
│   │   └── data/
│   └── navigator/…
└── _trash/                 # dropped DBs land here, recoverable by
    └── memoria-1730000000/ #   moving back into db/<name>/
```

`db.toml` is a small TOML file — `cat` it for a quick read:

```toml
name = "memoria"
created_at = 1730000000
map_size_mb = 4096

[eam]
d = 384
l_0 = 1000
k = 20
beta = 5.0
# … all EAM knobs, with defaults from the engine
```

The dimension lives in `eam.d`. **It is immutable after first write**
— Heather refuses to load a collection at a dimension that doesn't
match its persisted config. To change the dimension, create a new DB
and re-write your data.

## Lifecycle

### Create

```bash
curl -u admin:'…' -X POST http://127.0.0.1:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"memoria","dimension":384,"map_size_mb":4096}'
```

Internally:

1. Validate the name (rules above).
2. `mkdir -p $ROOT/db/memoria/data`.
3. Write `$ROOT/db/memoria/db.toml` (idempotency anchor).
4. Open the LMDB env at `$ROOT/db/memoria/data/`.
5. Insert into the in-memory `databases` map.

A crash anywhere mid-create leaves a recoverable state: re-run the
same `POST /db` call.

### List

```bash
curl -u admin:'…' http://127.0.0.1:6380/db
# {"databases":[
#   {"name":"default","dimension":128,"collections":3,...},
#   {"name":"memoria","dimension":384,"collections":12,...}
# ]}
```

### Inspect

```bash
curl -u admin:'…' http://127.0.0.1:6380/db/memoria
# {"name":"memoria","created_at":...,"dimension":384,
#  "map_size_mb":4096,"collections":12}
```

### Drop

```bash
curl -u admin:'…' -X DELETE http://127.0.0.1:6380/db/memoria
# {"dropped":true}
```

The directory is **moved** to `$ROOT/_trash/memoria-<ts>/`, not
deleted outright — you have a window to recover by stopping the
engine, moving `_trash/memoria-…` back to `db/memoria`, and
restarting.

`DELETE /db/default` is refused with 403 (would silently break every
legacy route).

### Restart safety

The server scans `$ROOT/db/` on boot, opens every directory that
contains a `db.toml`, and skips anything else. Half-built databases
(directory present, `db.toml` missing — the moment after `mkdir`,
before `db.toml` write) are silently ignored at boot, so a re-run of
`POST /db` recovers them.

## Multi-dimension example

One engine, three dimensions:

```bash
# 64-d default (set at boot via HEATHER_DIMENSION=64)
heather_server --data-dir /var/lib/heatherdb --dimension 64 \
  --admin-user admin --admin-password '…' &

# Add a 128-d navigator + 384-d memoria
curl -u admin:… -X POST http://localhost:6380/db -d '{"name":"navigator","dimension":128}' \
  -H 'Content-Type: application/json'
curl -u admin:… -X POST http://localhost:6380/db -d '{"name":"memoria","dimension":384}' \
  -H 'Content-Type: application/json'

# 64-d vector → default ✓
curl -u admin:… -X POST http://localhost:6380/db/default/collections/x/write \
  -H 'Content-Type: application/json' -d "{\"vectors\":[[$(printf '0.1,%.0s' {1..64} | sed 's/,$//')]]}"
# {"count":1}

# 64-d vector → memoria (expects 384) ✗
curl -u admin:… -X POST http://localhost:6380/db/memoria/collections/x/write \
  -H 'Content-Type: application/json' -d "{\"vectors\":[[$(printf '0.1,%.0s' {1..64} | sed 's/,$//')]]}"
# {"error":"dimension mismatch: expected 384, got 64"}
```

The engine enforces the dimension contract per database. There's no
way to write a wrong-shaped vector silently — the EAM internals
refuse it at the bottom of the write path.

## Algebra and database boundaries

`/db/{db}/algebra/{add,sub,scale,intersect}` operates **within one
database only**. Both source collections must live in the named DB.
Cross-database algebra is rejected (mathematically meaningless across
different attractor spaces — different dimensions, different random
projections, different EAM configs).

If you need to combine populations from two databases, you have to
re-encode and re-write into a single shared database first.

## Backup + restore

Per-database backup is a directory tar:

```bash
# Stop the engine first to guarantee a consistent snapshot.
systemctl stop heatherdb
tar -czf memoria-2026-05-11.tar.gz -C /var/lib/heatherdb db/memoria
systemctl start heatherdb

# Restore on the same or a different host.
tar -xzf memoria-2026-05-11.tar.gz -C /var/lib/heatherdb
# Engine picks it up on next boot.
```

For a live backup (no engine downtime), use LMDB's `mdb_copy` against
`$ROOT/db/memoria/data/`. The resulting copy is a consistent snapshot
even while writers are active.

## Limits

- **No quotas.** Any authenticated user with the right scope can
  fill up `map_size_mb`. Provision generously and watch disk on the
  host.
- **Soft cap on database count.** No enforced limit, but each open DB
  costs one LMDB env (a few file descriptors + an mmap region). 32–64
  is comfortable on a small VPS; 256 starts being interesting.
- **No replication.** Databases are single-writer per env (LMDB) and
  the engine is single-process. For HA, snapshot + restore is the
  story today.
