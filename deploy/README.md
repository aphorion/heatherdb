# deploy/

Run-on-the-VPS deployment for HeatherDB: build from source and install,
for any Debian/Ubuntu/Fedora box. If you're on Debian/Ubuntu and just want
the pre-built binary instead, grab the `.deb` from a
[release](https://github.com/aphorion/heather-db/releases) — see
[`../packaging/deb/`](../packaging/deb/) — and skip this script entirely.

| File | Purpose |
|------|---------|
| [`deploy`](./deploy) | The whole thing. SSH into your VPS, run it, walk away. |

Both install paths land on the exact same layout — `/usr/bin/heather`,
`/var/lib/heatherdb`, `/etc/heatherdb/env` — and this script installs the
same systemd unit the `.deb` ships
([`../packaging/deb/heatherdb.service`](../packaging/deb/heatherdb.service)),
so it doesn't matter which one you used; they're interchangeable and
upgrading from one to the other later just works.

## Usage

SSH into a fresh Debian/Ubuntu/Fedora VPS as a sudo-capable user.

```bash
# Option 1 — repo cloned on the VPS
git clone https://github.com/aphorion/heather-db
cd heather-db
sudo ./deploy/deploy

# Option 2 — one-liner from the internet
curl -fsSL https://raw.githubusercontent.com/aphorion/heather-db/main/deploy/deploy | sudo bash
```

The script is idempotent — re-run it to update the binary or apply config
changes. It installs system deps + Rust if missing, creates the `heatherdb`
user, drops the binary at `/usr/bin/heather`, generates a one-time admin
password into `/etc/heatherdb/env` (mode `0600`), installs the systemd unit,
enables boot-persistence, and smoke-tests `/health` on `:6380`. The
generated password is also printed in the post-install banner.

## Env overrides

```bash
HEATHER_VERSION=v0.1.0 \
HEATHER_DIM=384 \
HEATHER_ADMIN_USER=ops \
HEATHER_ADMIN_PASSWORD='your-strong-password' \
DATA_DIR=/srv/heatherdb \
SERVICE_USER=heather \
  sudo -E ./deploy/deploy
```

| Var | Default | Notes |
|-----|---------|-------|
| `HEATHER_VERSION`        | `main`                          | git ref (tag / branch / sha) to check out when cloning |
| `HEATHER_DIM`            | `128`                           | vector dimension written to `/etc/heatherdb/env` on first deploy |
| `HEATHER_ADMIN_USER`     | `admin`                         | first-boot HTTP Basic Auth user |
| `HEATHER_ADMIN_PASSWORD` | *(generated)*                   | first-boot password. If unset, the script generates 24 URL-safe chars and prints them in the post-install banner. |
| `DATA_DIR`               | `/var/lib/heatherdb`            | data directory — same default the `.deb` package uses |
| `SERVICE_USER`           | `heatherdb`                     | systemd user to run as |
| `SRC_DIR`                | `/usr/local/src/heather-db`     | where to clone the repo when not invoked from a local checkout |
| `SKIP_SMOKE_TEST`        | `0`                             | set to `1` to skip the `/health` poll |

## After it lands

```bash
curl http://localhost:6380/health                                  # smoke (no auth)
curl -u admin:'<password>' http://localhost:6380/db                # list databases
journalctl -u heatherdb -f                                         # live logs
systemctl restart heatherdb                                        # after editing /etc/heatherdb/env

# User management — LMDB-backed, mutations visible to the live engine.
heather --data-dir /var/lib/heatherdb user list
heather --data-dir /var/lib/heatherdb user create alice --password '…' --scope memoria
heather --data-dir /var/lib/heatherdb user passwd admin --password '…'

# Backup / restore / live snapshot.
heather --data-dir /var/lib/heatherdb backup create --output /backups/h-$(date +%F).tar.gz
heather --data-dir /var/lib/heatherdb snapshot create --db default          # live, safe with engine running
heather --data-dir /var/lib/heatherdb restore restore --input /backups/h-2026-05-11.tar.gz
```

For HTTPS, put Caddy or nginx in front. Caddy:

```caddy
api.heather.example.com {
  reverse_proxy 127.0.0.1:6380
  encode zstd gzip
}
```

For the Raspberry Pi cross-compile flow (different target, different deploy
cadence), see
[`heatherdb-pi-demo/deploy/`](https://github.com/aphorion/heatherdb-pi-demo).
