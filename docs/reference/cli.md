# Operator CLI reference

The `heather` binary is both the server and the admin tool. With no
subcommand it starts the HTTP server; with a subcommand it runs synchronously
against the data directory and exits.

```
heather [OPTIONS]                       # run the server
heather [OPTIONS] <SUBCOMMAND>          # run one operation and exit
```

`--data-dir` (or `HEATHER_DATA_DIR`) is required by everything. It is a global
argument, so it may appear before or after the subcommand.

> The crate is named `heather_server`; the installed binary is `heather`.

## `user`

Manages the LMDB user store at `$DATA_DIR/system/data/`. Mutations are visible
to a running engine immediately — no restart, because every request opens a
fresh read transaction.

### `user create <NAME> --password <PW> [--scope <SCOPE>]`

```bash
heather --data-dir /var/lib/heatherdb user create alice \
  --password 'a-real-password' --scope movies
```

| Argument | Default | Notes |
|---|---|---|
| `<NAME>` | *(required)* | `[a-zA-Z][a-zA-Z0-9_.-]{0,63}` |
| `--password` | *(required)* | At least 8 characters |
| `--scope` | `root` | `root`, `*`, or a database name |

Fails if the user already exists.

### `user list`

Prints name, scope and creation time. Never prints hashes.

### `user passwd <NAME> --password <PW>`

Rotates a password. At least 8 characters.

### `user delete <NAME>`

Exits non-zero if the user does not exist.

## `backup`

### `backup create --output <FILE.tar.gz> [--db <NAME>]`

Cold `tar.gz` of the data directory. **Refuses to run while the engine is
serving** — it takes the same exclusive lock.

A full backup packs `server.toml`, `system/` (users and tokens) and `db/`. With
`--db` it packs only `db/<name>/`, laid out so the archive untars at the
data-directory root and lands back where it came from.

### `backup list`

Inventories `$DATA_DIR/backups/`. Informational only — archives can live
anywhere.

## `restore`

### `restore restore --input <PATH> [--db <NAME>]`

Note the doubled word: `restore` is both the subcommand group and its only
command. Refuses to run while the engine is serving.

Two input shapes:

- **`.tar.gz`** — unpacked into the data directory. `--db NAME` extracts only
  that database from a full archive. Existing files are overwritten.
- **A snapshot directory** — copied to `db/<name>/`, where the name comes from
  the snapshot's own `db.toml`. `--db` overrides the target name. **Refuses if
  the target directory already exists**; move it aside first.

Archive entries are validated on the way in: absolute paths, `..` components,
unknown top-level directories and non-regular entries (symlinks, devices) are
all rejected, and a database name read from an untrusted `db.toml` is validated
before it becomes a path.

## `snapshot`

### `snapshot create --db <NAME> [--output <DIR>] [--compact <BOOL>]`

**Safe with the engine running.** Uses LMDB's `Env::copy_to_file` — the same
primitive `mdb_copy` uses — for a transactionally consistent live copy.

| Argument | Default |
|---|---|
| `--db` | *(required)* |
| `--output` | `$DATA_DIR/snapshots/<db>-<unix-seconds>` |
| `--compact` | `true` — smaller files, slightly slower |

Output is a directory containing `db.toml` and `data/`. Refuses if the output
path already exists. Covers one database and does **not** include the user
store.

### `snapshot list`

Lists directories under `$DATA_DIR/snapshots/`.

### `snapshot delete <NAME>`

Deletes one snapshot directory under `$DATA_DIR/snapshots/`.

## Exit codes

`0` on success, `1` on any error, with the message on stderr.

## Server options

Every flag in [Configuration](configuration.md#process-flags) is also a
`heather --help` option. The ones you are most likely to type:

```bash
heather --data-dir /var/lib/heatherdb --port 6380 --host 127.0.0.1
heather --data-dir /tmp/scratch --auth-disabled          # dev only
RUST_LOG=debug heather --data-dir /var/lib/heatherdb
```
