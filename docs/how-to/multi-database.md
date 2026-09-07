# Work with multiple databases

One HeatherDB server hosts N databases. Each has its own LMDB environment, its
own `db.toml`, its own directory under `$HEATHER_DATA_DIR/db/`, and — the part
that usually drives the decision — **its own vector dimension**.

Isolation is provided by the operating system: separate environments, separate
file handles. There is no key-prefix multiplexing.

## When to create a new database

- **You need a different vector dimension.** This is the common reason.
  Dimension is fixed at creation and cannot be changed.
- **You want independent EAM tuning** — a different `k`, `beta`, or the MDL
  allocation gate.
- **You want a separate map-size ceiling**, backup unit, or audit policy.
- **You want to hand someone a credential that reaches one dataset only.**

Do *not* create a database per tenant if you have thousands of tenants — each
one is a mapped LMDB environment. Use collections within a database for that.

## Create one

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"movies","dimension":384,"map_size_mb":32768}'
```

Names are 1–63 characters, `[a-z][a-z0-9_-]*`, and must not start with `_`
(reserved for system directories). Root scope only.

You can also set EAM knobs at creation — see
[Tune a database](tune-a-database.md).

## Inspect

```bash
curl -u admin:pw http://localhost:6380/db
curl -u admin:pw http://localhost:6380/db/movies
```
```json
{"name":"movies","created_at":1788260246,"dimension":384,"map_size_mb":32768,"collections":0}
```

`collections` is a live count, not a stored field.

## Address a database

Everything you can do to the default database, you can do to a named one by
prefixing the path:

```bash
# default database, legacy shape
curl -u admin:pw -X POST http://localhost:6380/collections/taste/write     -d '…'

# named database, scoped shape
curl -u admin:pw -X POST http://localhost:6380/db/movies/collections/taste/write -d '…'
```

The two shapes run the same handler. The legacy shape exists for backwards
compatibility and is permanently pinned to the database named `default`.

Prefer the scoped shape in new code. It is explicit, it works for every
database, and a `Database("movies")`-scoped credential can use it.

Not mirrored under `/db/{db}/`: the stateless `/vec/*` routes, which touch no
database at all.

## The default database

`default` is created automatically on first boot if no databases exist. Its
dimension comes from `--dimension` / `HEATHER_DIMENSION` (128 unless you say
otherwise) and its map ceiling from `--map-size-mb`, both only on the boot that
creates it. Afterwards the persisted `db.toml` wins and those flags are ignored
for it.

`DELETE /db/default` returns `403`. The legacy routes target it, so dropping it
would make them start silently 404-ing. To be rid of it, start over with an
empty data directory.

## Drop one

```bash
curl -u admin:pw -X DELETE http://localhost:6380/db/movies
```
```json
{"dropped": true}
```

This does **not** delete anything. The directory is moved to
`$HEATHER_DATA_DIR/_trash/movies-<unix-seconds>/`. There is no automatic
cleanup of the trash — reclaiming that disk is your call.

`{"dropped": false}` means the database was not mounted.

## Recover a dropped database

```bash
systemctl stop heatherdb
mv /var/lib/heatherdb/_trash/movies-1788260246 /var/lib/heatherdb/db/movies
systemctl start heatherdb
```

The directory carries its own `db.toml`, so it comes back with its original
dimension, map size, dream and audit settings.

## Move a database between servers

Snapshot it live, copy the directory, restore it on the target:

```bash
# source
heather --data-dir /var/lib/heatherdb snapshot create --db movies
scp -r /var/lib/heatherdb/snapshots/movies-… target:/tmp/

# target (engine stopped)
heather --data-dir /var/lib/heatherdb restore restore --input /tmp/movies-…
```

See [Back up and restore](backup-and-restore.md).
