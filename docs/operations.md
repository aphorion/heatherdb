# Operations

Production deployment, monitoring, backups, upgrades. Anything you need
to keep HeatherDB alive and authenticated.

## Three deployment shapes

Pick one — they're not mutually exclusive but mixing is rare.

> **For Coolify deploys, jump straight to [coolify.md](./coolify.md).**
> The compose file in this repo is already Coolify-friendly; that page
> is the focused walkthrough (env vars, domain assignment, day-2 ops
> via the container terminal, backups against the persistent volume).

### A. Docker (recommended)

`docker compose up` brings up the **engine + Fovea** in two services
that share an internal network. Fovea's Caddy front-end terminates the
SPA + reverse-proxies `/api/*` to the engine over the compose network.
Auth on Fovea's edge is HTTP Basic (configured via `fovea/.env`); the
engine itself is auth-on-by-default with the bootstrap admin.

```bash
git clone https://github.com/aphorion/heather-db
cd heather-db

# Optionally tighten Fovea: copy fovea/.env.example → fovea/.env and
# fill in the bcrypt-hashed Caddy basic-auth credentials.

docker compose up -d
docker compose logs -f
```

| URL                     | What |
|-------------------------|------|
| `http://localhost:6380` | Engine HTTP API (auth-required). |
| `http://localhost:8080` | Fovea web UI (Caddy in front). |

For TLS: front the compose stack with another Caddy (on the host),
or use a cloud LB. The engine itself doesn't speak TLS; see
[auth.md → TLS is mandatory](./auth.md#-tls-is-mandatory-in-production).

### B. `deploy/deploy` on a $5 VPS

```bash
ssh user@your-vps
curl -fsSL https://raw.githubusercontent.com/aphorion/heather-db/main/deploy/deploy | sudo bash
```

The script:

1. Installs system deps (`build-essential`, OpenSSL, Rust if missing).
2. Creates the `heatherdb` user, `/opt/heatherdb/data`, `/etc/heatherdb/env`.
3. Builds + installs the binary at `/opt/heatherdb/heather`.
4. Drops the systemd unit at `/etc/systemd/system/heatherdb.service`.
5. Enables + restarts the service.
6. Smoke-tests `/health`.

Before first restart, set the admin in `/etc/heatherdb/env`:

```ini
HEATHER_DATA_DIR=/opt/heatherdb/data
HEATHER_DIMENSION=128
HEATHER_PORT=6380
HEATHER_HOST=127.0.0.1                # loopback only — TLS proxy in front
HEATHER_MAP_SIZE_MB=4096
HEATHER_REQUEST_TIMEOUT=600
HEATHER_ADMIN_USER=admin
HEATHER_ADMIN_PASSWORD=<your-strong-password>
RUST_LOG=info
```

Then `sudo systemctl restart heatherdb` and put Caddy or nginx in
front (see [auth.md](./auth.md#-tls-is-mandatory-in-production)).

### C. `cargo run --release` (development)

```bash
cargo build --release -p heather_server
HEATHER_DATA_DIR=/tmp/heatherdb HEATHER_DIMENSION=128 \
HEATHER_AUTH_DISABLED=1 \
  ./target/release/heather
```

Auth disabled is fine for `127.0.0.1` development. Don't bind to `0.0.0.0`
without auth.

---

## Configuration knobs

All accept either a CLI flag or an env var. Env wins when both are set.

### Server

| Flag                       | Env                           | Default       | Notes |
|----------------------------|-------------------------------|---------------|-------|
| `--data-dir`               | `HEATHER_DATA_DIR`            | (required)    | Root for `db/`, `system/` (user store), and snapshots/backups. |
| `--dimension`              | `HEATHER_DIMENSION`           | 128           | Used only when first-creating the `default` DB. |
| `--port`                   | `HEATHER_PORT`                | 6380          | TCP listen port. |
| `--host`                   | `HEATHER_HOST`                | 0.0.0.0       | Bind address. **Set to `127.0.0.1`** when behind a reverse proxy. |
| `--max-body-size`          | `HEATHER_MAX_BODY_SIZE`       | 2097152 (2MB) | Largest single request. Raise for big batch writes. |
| `--request-timeout`        | `HEATHER_REQUEST_TIMEOUT`     | 30            | Per-request seconds. Algebra ops can take a while at scale. |
| `--map-size-mb`            | `HEATHER_MAP_SIZE_MB`         | 256           | LMDB map size for the auto-created `default` DB. |

### Auth

| Flag                | Env                          | Default | Notes |
|---------------------|------------------------------|---------|-------|
| `--admin-user`      | `HEATHER_ADMIN_USER`         | `admin` | First-boot admin username. Ignored once user store is non-empty. |
| `--admin-password`  | `HEATHER_ADMIN_PASSWORD`     | (random)| First-boot admin password. Random if unset (printed once). |
| `--auth-disabled`   | `HEATHER_AUTH_DISABLED`      | false   | Disable auth entirely. **Dev only.** |

### Logging

`RUST_LOG=debug` for verbose. Default `info`. The engine emits
`tracing` spans per request including the path and HTTP status.

---

## systemd unit

`deploy/systemd/heatherdb.service` is the canonical unit:

```ini
[Unit]
Description=HeatherDB
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=heatherdb
Group=heatherdb
WorkingDirectory=/opt/heatherdb
ExecStart=/opt/heatherdb/heather
Restart=always
RestartSec=2
LimitNOFILE=65536

EnvironmentFile=-/etc/heatherdb/env
Environment=HEATHER_DATA_DIR=/opt/heatherdb/data
# (more defaults — see deploy/systemd/heatherdb.service)

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/heatherdb/data

[Install]
WantedBy=multi-user.target
```

After editing `/etc/heatherdb/env`: `sudo systemctl restart heatherdb`.

---

## Monitoring

### Liveness

```
GET /health → 200 {"status":"ok"}
```

Always returns 200 if the process is alive — *doesn't* check that the
LMDB env is mountable. For a "ready to serve" probe, hit
`GET /db/{name}` on a known DB with valid auth.

### Per-database health

```bash
curl -u admin:… http://127.0.0.1:6380/db | jq
```

`collections`, `dimension`, `map_size_mb` per DB. Diff against
yesterday to spot anomalies.

### Process metrics

The engine doesn't currently emit Prometheus. Use:

- `journalctl -u heatherdb -f` for live logs (request paths, status,
  latency in `tracing` spans).
- `du -sh /var/lib/heatherdb/db/*/data` for per-DB disk use.
- `ps`/`top` for the usual RSS/CPU.

A `/metrics` endpoint is on the v0.3 roadmap.

---

## Backups

The engine ships three subcommands that handle this end-to-end —
no `mdb_copy` ceremonies, no "did you remember the user store?"
gotchas (the `system/` env is just another LMDB env that the backup
walks alongside `db/<name>/`). Pick the one that matches the
situation:

| Need                                                        | Use |
|-------------------------------------------------------------|-----|
| Nightly cold archive (everything: DBs + users + config)     | `backup create` |
| Just one database, packaged for transfer                    | `backup create --db NAME` |
| **Live consistent copy** of one DB while the engine serves  | `snapshot create --db NAME` |
| Pull a backup or snapshot back into a data dir              | `restore restore --input ...` |

The engine stores everything under `$HEATHER_DATA_DIR/`:

```
$HEATHER_DATA_DIR/
├── server.toml             # tag file
├── system/data/            # user store (LMDB env)
├── db/<name>/              # per-database state
└── snapshots/              # default landing zone for `snapshot create`
```

### Cold backup (engine running fine — but expect a brief pause)

```bash
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  backup create --output /backups/heatherdb-$(date +%F).tar.gz
```

Tarball contents:

```
server.toml
system/data/{data,lock}.mdb
db/<name>/db.toml
db/<name>/data/{data,lock}.mdb
…
```

Per-database flavour:

```bash
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  backup create --db memoria --output /backups/memoria-$(date +%F).tar.gz
```

This is the **cold** flow — it tars the on-disk files directly. Run
it at a moment when the engine is idle (or briefly stop the service)
for a guaranteed-consistent capture. For zero-downtime, see
*Live snapshot* below.

### Live snapshot (engine serving traffic)

Wraps LMDB's `Env::copy_to_file` (the same primitive `mdb_copy` uses) —
**safe with active writers**:

```bash
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  snapshot create --db memoria
# ✓ snapshot: /var/lib/heatherdb/snapshots/memoria-1730000000

sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  snapshot list
# NAME                                       SIZE  CREATED
# memoria-1730000000                       2.7 MB  2026-05-11 09:42:07Z
```

The snapshot is one self-contained directory — `db.toml` + `data/data.mdb`
(LMDB compaction is on by default, smaller files). Treat it as a
durable, restorable artefact.

```bash
# Drop a snapshot when you're done with it.
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  snapshot delete memoria-1730000000
```

### Restore

Both `backup` tarballs and `snapshot` directories restore through the
same command. Stop the engine first (LMDB locks the env you're
restoring into).

```bash
sudo systemctl stop heatherdb

# Full data-dir restore (tarball — covers every DB + the user store).
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  restore restore --input /backups/heatherdb-2026-05-11.tar.gz

# Single-database restore from a full backup.
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  restore restore --input /backups/heatherdb-2026-05-11.tar.gz --db memoria

# Restore a live snapshot directory.
sudo -u heatherdb heather --data-dir /var/lib/heatherdb \
  restore restore --input /var/lib/heatherdb/snapshots/memoria-1730000000

sudo systemctl start heatherdb
```

If a per-DB restore would clobber an existing database, the command
refuses — move the existing `db/<name>/` aside first, or restore into
a different `--data-dir` to inspect first.

### Scheduling nightly backups

systemd timer is the boring choice:

```ini
# /etc/systemd/system/heatherdb-backup.service
[Unit]
Description=HeatherDB nightly cold backup

[Service]
Type=oneshot
User=heatherdb
ExecStart=/usr/local/bin/heather \
  --data-dir /var/lib/heatherdb \
  backup create --output /backups/heatherdb-%%i.tar.gz
```

```ini
# /etc/systemd/system/heatherdb-backup.timer
[Unit]
Description=Run HeatherDB cold backup nightly

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now heatherdb-backup.timer
```

For zero-downtime per-DB rolling snapshots: drop `snapshot create` in
a similar timer.

---

## Upgrades

Pre-1.0: minor versions can break wire shapes. The legacy
`/collections/*` and `/algebra/*` aliases are kept stable across
the v0.x line, so most clients survive without code changes — only
add `/db/{db}/...` callers when you opt into multi-tenancy.

### Process

```bash
# 1. Backup (above).
# 2. Pull and rebuild.
cd /opt/heatherdb-src
sudo -u heatherdb git pull
sudo -u heatherdb cargo build --release -p heather_server
sudo install -m 0755 -o heatherdb -g heatherdb \
  target/release/heather /opt/heatherdb/heather
# 3. Restart.
sudo systemctl restart heatherdb
# 4. Verify.
curl http://127.0.0.1:6380/health
```

LMDB sub-DB layouts are append-only across pre-1.0 minor bumps. New
optional named sub-DBs may appear; existing keys never change shape.

---

## Resource sizing

| Workload               | Suggested map size | Notes |
|------------------------|--------------------|-------|
| 1k-5k attractors / DB   | 256 MB             | Default. Plenty for the demo path. |
| 50k attractors / DB     | 1 GB               | A small projet. |
| 500k attractors / DB    | 4 GB               | The MovieLens 25M scale. |
| Multi-million / DB      | 16+ GB             | Plan ahead — `map_size_mb` is set at create-time. |

Memory at steady state is roughly **1.5× the on-disk size** per open
DB — LMDB mmap plus working set. Shutdown is clean (LMDB durability
guarantees), so SIGTERM the systemd unit if you need to move the
process.

CPU: writes are the bottleneck. The engine releases the LMDB write
lock per batch — a 1000-vector batch write is ~3× cheaper than 1000
single writes.

---

## Disaster recovery

| Scenario                         | Recovery |
|----------------------------------|----------|
| Process crashed mid-write         | Restart. LMDB rolls back the partial transaction; no corruption. |
| Lost the admin password           | Stop engine. `rm -rf $HEATHER_DATA_DIR/system/`. Set `HEATHER_ADMIN_PASSWORD=…` on next boot — first-boot path runs again. (Per-DB data is untouched.) |
| Accidentally dropped a DB         | Stop engine. `mv $ROOT/_trash/<name>-<ts> $ROOT/db/<name>`. Restart. |
| `map_size_mb` exhausted (writes 500) | Stop engine. Edit `db/<name>/db.toml`, raise `map_size_mb`. Restart. (Can't do this live in v0.x; planned for v0.3.) |
| Disk full                        | Free space. LMDB writes simply fail with `MDB_MAP_FULL`; no data loss. |

---

## Connecting clients

| Client                    | How |
|---------------------------|-----|
| `curl`                    | `curl -u user:pass http://...` |
| Python (requests, httpx) | `auth=(user, pw)` |
| Node (fetch)              | `Authorization: Basic <btoa(user:pw)>` |
| Rust (reqwest)            | `.basic_auth(user, Some(pw))` |
| [Fovea](./fovea.md)       | UI fields on the Connections screen. |

For a HeatherDB behind a Caddy/nginx reverse proxy: hit the proxy URL
(`https://api.heather.example.com`) and let the proxy unwrap TLS. The
auth header passes through unchanged.
