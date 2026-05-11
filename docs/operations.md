# Operations

Production deployment, monitoring, backups, upgrades. Anything you need
to keep HeatherDB alive and authenticated.

## Three deployment shapes

Pick one — they're not mutually exclusive but mixing is rare.

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
3. Builds + installs the binary at `/opt/heatherdb/heather_server`.
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
  ./target/release/heather_server
```

Auth disabled is fine for `127.0.0.1` development. Don't bind to `0.0.0.0`
without auth.

---

## Configuration knobs

All accept either a CLI flag or an env var. Env wins when both are set.

### Server

| Flag                       | Env                           | Default       | Notes |
|----------------------------|-------------------------------|---------------|-------|
| `--data-dir`               | `HEATHER_DATA_DIR`            | (required)    | Root for `db/` + `users.json`. |
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
ExecStart=/opt/heatherdb/heather_server
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

The engine stores everything in two places:

1. `$HEATHER_DATA_DIR/users.json` — credentials.
2. `$HEATHER_DATA_DIR/db/<name>/` — per-database state.

### Cold backup (engine stopped)

```bash
sudo systemctl stop heatherdb
sudo tar -C /var/lib/heatherdb -czf /backups/heatherdb-$(date +%F).tar.gz \
  users.json server.toml db/
sudo systemctl start heatherdb
```

Restoring: stop the engine, untar over `$HEATHER_DATA_DIR`, start.

### Hot backup (engine running)

LMDB has `mdb_copy` for consistent live snapshots:

```bash
for db in /var/lib/heatherdb/db/*/data; do
  name=$(basename $(dirname $db))
  sudo -u heatherdb mdb_copy "$db" /backups/$name-$(date +%F)/
done
sudo cp /var/lib/heatherdb/users.json /backups/
sudo cp /var/lib/heatherdb/server.toml /backups/
```

Each `mdb_copy` produces a consistent point-in-time snapshot of one
database, even with active writers. Glue them together for a full
backup; restore by laying them back into `db/<name>/data/`.

### Per-database snapshot

Just one DB:

```bash
sudo systemctl stop heatherdb       # or use mdb_copy live
tar -czf memoria-snap.tar.gz -C /var/lib/heatherdb db/memoria
sudo systemctl start heatherdb

# Restore on the same or a different host
tar -xzf memoria-snap.tar.gz -C /var/lib/heatherdb
sudo systemctl restart heatherdb
```

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
  target/release/heather_server /opt/heatherdb/heather_server
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
| Lost the admin password           | Stop engine. Delete `users.json`. Set `HEATHER_ADMIN_PASSWORD=…` on next boot — first-boot path runs again. |
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
