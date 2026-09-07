# Run it in production

[Deploy HeatherDB](deploy.md) gets a process running. This page is what to
settle before real traffic reaches it, and what to keep doing afterwards. Work
top to bottom the first time; the [checklist](#the-checklist) at the end is
the version you copy into a runbook.

## Decide what cannot be changed later

Two choices outlive the deployment, because neither can be edited in place.

**Dimension is fixed per database.** The engine refuses to open a collection at
a width that disagrees with `db.toml`, and there is no re-dimension command —
changing it means a new database and a re-ingest. Pick it from the encoder you
are actually going to use, not from the default.

**Pre-1.0 means no migrations.** There is no migration machinery in the engine,
so a format change between versions is handled by re-ingesting, not by an
upgrade path. Read the release notes before moving a version, and keep the
source data you ingested from for as long as that is true.

Everything else — map size, EAM knobs, users, limits — is adjustable later.
[Tune a database](tune-a-database.md) covers each.

## Put the data somewhere real

The data directory is an LMDB environment, memory-mapped and fsynced.

- **Local block storage.** Not NFS, not SMB, not a shared network mount. LMDB's
  locking assumes a local filesystem and will corrupt across one that lies
  about fsync.
- **One process per data directory, ever.** The engine takes an exclusive
  advisory lock (`engine.lock`) and refuses to double-mount. Two engines on one
  directory is data loss, and the lock is the only thing standing between you
  and it — do not defeat it by copying the directory while a process is live.
- **Size the map ceiling per database.** `--map-size-mb` sizes only the
  auto-created `default` database, and only on the boot that creates it. Raise
  a ceiling with the [map size](tune-a-database.md#map-size) procedure rather
  than by editing files under the data directory.
- **The volume is the deployment.** In Docker or Kubernetes, a container
  without a named volume or PVC loses everything on restart. Check this before
  you write anything you care about.

## Close the front door

- **Set `HEATHER_ADMIN_PASSWORD` explicitly.** If you do not, the engine
  generates one, prints it once, and writes it to
  `$HEATHER_DATA_DIR/initial-admin-password` (mode 0600). Copy it, then delete
  the file.
- **Give every application its own scoped user.** A user scoped to
  `Database(name)` cannot touch another database or the `/db` management
  routes. The admin account is for operators, not for services — see
  [Manage users and authentication](manage-users.md).
- **Never set `HEATHER_AUTH_DISABLED` on anything a network can reach.**
  Anyone who opens the port gets full read and write. The engine warns loudly
  at boot in that mode; do not train yourself to ignore it.
- **Bind to loopback when a proxy fronts it.** `HEATHER_HOST=127.0.0.1` keeps
  the engine off every other interface. (Inside a container, keep `0.0.0.0` and
  publish the port deliberately instead.)
- **Terminate TLS in front.** The engine speaks plain HTTP by design; Caddy,
  nginx, Traefik, or a cloud load balancer all work. See
  [Terminate TLS in front](deploy.md#terminate-tls-in-front).

## Set the limits to match your workload

The defaults are sized for a small deployment and fail closed, which shows up
as a rejected request rather than a slow one.

| Setting | Default | Raise it when |
|---|---|---|
| `HEATHER_REQUEST_TIMEOUT` | `30` (seconds) | Large algebra or bulk operations time out. The bundled compose file uses `600`. |
| `HEATHER_MAX_BODY_SIZE` | `2097152` (2 MB) | A single write batch or bulk load exceeds 2 MB of JSON. |
| `HEATHER_MAX_BULK_ITEMS` | see [configuration](../reference/configuration.md) | You bulk-load in bigger pages. |
| `HEATHER_MAX_BATCH_QUERIES` | see configuration | You batch more queries per request. |
| `HEATHER_MAX_ALGEBRA_LOCATIONS` | see configuration | Algebra over large collections is refused. |

Two of these have to agree with the proxy in front: its request timeout must
exceed `HEATHER_REQUEST_TIMEOUT`, and its body-size limit must exceed
`HEATHER_MAX_BODY_SIZE`. A mismatch produces an error from the proxy that looks
like an engine bug.

## Make restores boring

Backups are only worth what a rehearsed restore is worth.

- **Snapshots are safe while the engine runs** — `heather snapshot create --db
  NAME` uses a live LMDB copy. This is the one to schedule.
- **Cold backups need the engine stopped**, and so does a restore: it takes the
  LMDB lock. Plan the window.
- **Rehearse the restore somewhere else, on a schedule.** A backup that has
  never been restored is a hypothesis. [Back up and
  restore](backup-and-restore.md#test-the-restore) has the procedure.
- **Snapshot before every upgrade**, before every knob change with a re-ingest
  behind it, and before any operation you have not done before in production.

## Let the orchestrator see it

Four unauthenticated routes exist for exactly this:

- `GET /health` (and `/healthz`) — the process is up. Use it as a liveness
  probe.
- `GET /ready` (and `/readyz`) — the databases are open and serving. Use it as
  a readiness probe, and as the gate on a load balancer.

Everything else requires credentials, so do not point a probe at it.

**Stop gracefully.** SIGTERM and Ctrl-C flush staged audit records; SIGKILL
loses whatever is still buffered. Give the engine a real termination grace
period — 30 seconds is plenty — rather than the default impatience of some
orchestrators.

## Watch it

There is **no Prometheus endpoint yet**. What you have instead:

- **Logs.** `RUST_LOG=info` by default, `debug` when you are diagnosing. Every
  write and read logs its collection and dimension; failures log the reason.
- **`GET /collections/{name}/stats`** — location count, total writes, current
  eta, write-count distribution. This is the signal that a collection is
  filling up or that writes are concentrating; poll it into whatever you
  already run.
- **The audit log.** Per database, on by default, queryable at
  `GET /db/{db}/audit`, with retention and buffering configured in `db.toml`.
  Records are buffered and flushed, so there is a small loss window on an
  ungraceful stop — [read the audit log](read-the-audit-log.md#know-the-loss-window)
  is explicit about it. Set `visibility` deliberately: `Root` keeps it to
  operators, `DbUsers` lets a database's own users read their own trail.

## Upgrade deliberately

1. Read the release notes for anything that changes the on-disk format.
2. `heather snapshot create --db NAME` for every database.
3. Stop the engine (SIGTERM, and wait for it).
4. Swap the binary, or the image tag, or `apt install` the new `.deb`.
5. Start it, and watch the boot log — it names every database it opened.
6. Check `/ready`, then one real query per database.

Rolling back means restoring the snapshot you took at step 2 with the previous
binary. That is the whole plan, and it is why step 2 is not optional.

## The checklist

| | Before first traffic |
|---|---|
| ☐ | Dimension chosen from the real encoder, not the default |
| ☐ | Data directory on local block storage, on a persistent volume |
| ☐ | Exactly one engine process per data directory |
| ☐ | `HEATHER_ADMIN_PASSWORD` set; `initial-admin-password` deleted |
| ☐ | A scoped user per application; admin reserved for operators |
| ☐ | `HEATHER_AUTH_DISABLED` unset everywhere |
| ☐ | TLS terminated in front; engine on loopback or a deliberate port |
| ☐ | Proxy timeout and body limit both above the engine's |
| ☐ | Liveness on `/health`, readiness on `/ready` |
| ☐ | Termination grace period long enough for a clean flush |
| ☐ | Snapshot schedule running, and one restore rehearsed |
| ☐ | Logs shipped somewhere you will actually look |

## Related

- [Deploy HeatherDB](deploy.md) — the install paths themselves.
- [Tune a database](tune-a-database.md) — dimension, map size, EAM knobs.
- [Back up and restore](backup-and-restore.md) — snapshots, cold backups, restores.
- [Manage users and authentication](manage-users.md) — scopes and rotation.
- [Configuration](../reference/configuration.md) — every flag and variable.
