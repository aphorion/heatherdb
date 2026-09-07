# Back up and restore

Three operations, each for a different situation.

| | `snapshot` | `backup` | `restore` |
|---|---|---|---|
| Engine may be running | **yes** | no | no |
| Covers | one database | everything, or one database | whatever is in the input |
| Includes users | no | yes (full backup) | yes (full backup) |
| Output | a directory | one `.tar.gz` | — |

All three are subcommands of the same `heather` binary and operate on a data
directory directly.

## Take a live snapshot

This is the one you want for routine backups of a serving engine. It uses
LMDB's `Env::copy_to_file` — the same primitive `mdb_copy` uses — to produce a
transactionally consistent copy without stopping anything.

```bash
heather --data-dir /var/lib/heatherdb snapshot create --db movies
```
```
✓ snapshot: /var/lib/heatherdb/snapshots/movies-1788260246
```

The output directory contains `db.toml` and `data/`. Compaction is on by
default (smaller files, slightly slower); pass `--compact false` to skip it.
Send it elsewhere with `--output`:

```bash
heather --data-dir /var/lib/heatherdb snapshot create --db movies \
  --output /backups/movies-$(date +%F)
```

The command refuses if the output path already exists.

Manage the conventional location:

```bash
heather --data-dir /var/lib/heatherdb snapshot list
heather --data-dir /var/lib/heatherdb snapshot delete movies-1788260246
```

A snapshot covers **one database**. It does not include the user store, so it
is not a complete disaster-recovery artefact on its own.

## Take a cold backup

```bash
systemctl stop heatherdb
heather --data-dir /var/lib/heatherdb backup create \
  --output /backups/heatherdb-$(date +%F).tar.gz
systemctl start heatherdb
```

A full backup packs `server.toml`, `system/` (the user and token store) and
`db/` — everything needed to stand the deployment back up. Restrict it to one
database with `--db movies`, which packs only `db/movies/`.

`backup create` **refuses to run while the engine is serving**: it takes the
same exclusive advisory lock the engine holds, and an inconsistent copy of a
live LMDB environment is worse than no copy.

```bash
heather --data-dir /var/lib/heatherdb backup list
```

lists what is in `$DATA_DIR/backups/` — informational only. Keep archives
wherever you like.

## Restore

The engine must be stopped; restore takes the same lock.

```bash
systemctl stop heatherdb
heather --data-dir /var/lib/heatherdb restore restore \
  --input /backups/heatherdb-2026-05-11.tar.gz
systemctl start heatherdb
```

`restore restore` accepts both input shapes:

**A `.tar.gz`** is unpacked into the data directory, landing each file back
where it came from. Add `--db movies` to extract only that database from a full
archive. Existing files are overwritten.

**A snapshot directory** is copied into `db/<name>/`, where the name comes from
the snapshot's own `db.toml`. Override it with `--db` to restore under a
different name. This path **refuses if the target directory already exists** —
move the current one aside first:

```bash
mv /var/lib/heatherdb/db/movies /var/lib/heatherdb/db/movies.old
heather --data-dir /var/lib/heatherdb restore restore --input /backups/movies-2026-05-11
```

Archive entries are validated on the way in: absolute paths, `..` components,
unknown top-level directories and non-regular entries (symlinks, devices) are
all refused, and a database name read out of an untrusted `db.toml` is
validated before it becomes a path.

## A workable routine

```bash
# Hourly, no downtime — per database.
heather --data-dir /var/lib/heatherdb snapshot create --db movies --output /backups/hourly/movies-$(date +%FT%H)

# Weekly, during a maintenance window — everything, users included.
systemctl stop heatherdb
heather --data-dir /var/lib/heatherdb backup create --output /backups/weekly/full-$(date +%F).tar.gz
systemctl start heatherdb
```

Snapshots give you low-RPO coverage of the data. The weekly cold backup is what
lets you rebuild the whole deployment, credentials and all.

## Test the restore

An untested backup is a rumour. Restore into a scratch directory and boot
against it:

```bash
mkdir /tmp/restore-test
heather --data-dir /tmp/restore-test restore restore --input /backups/weekly/full-2026-05-11.tar.gz
HEATHER_DATA_DIR=/tmp/restore-test HEATHER_PORT=6390 heather
curl -u admin:pw http://localhost:6390/db
```

Nothing here touches the live data directory, and the engine lock means you
cannot accidentally point two processes at the same one.
