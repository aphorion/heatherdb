# deploy/

Run-on-the-VPS deployment for HeatherDB.

| File | Purpose |
|------|---------|
| [`deploy`](./deploy)                                    | The whole thing. SSH into your VPS, run it, walk away. |
| [`systemd/heatherdb.service`](./systemd/heatherdb.service) | Hardened systemd unit installed by the script. Override env via `/etc/heatherdb/env`. |

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
user, drops the binary at `/opt/heatherdb/heather_server`, installs the
systemd unit, enables boot-persistence, and smoke-tests `/health` on `:6380`.

## Env overrides

```bash
HEATHER_VERSION=v0.1.0 \
HEATHER_DIM=384 \
INSTALL_DIR=/srv/heatherdb \
SERVICE_USER=heather \
  sudo -E ./deploy/deploy
```

| Var | Default | Notes |
|-----|---------|-------|
| `HEATHER_VERSION`  | `main`                          | git ref (tag / branch / sha) to check out when cloning |
| `HEATHER_DIM`      | `128`                           | vector dimension written to `/etc/heatherdb/env` on first deploy |
| `INSTALL_DIR`      | `/opt/heatherdb`                | install root (binary + data dir live here) |
| `SERVICE_USER`     | `heatherdb`                     | systemd user to run as |
| `SRC_DIR`          | `/usr/local/src/heather-db`     | where to clone the repo when not invoked from a local checkout |
| `SKIP_SMOKE_TEST`  | `0`                             | set to `1` to skip the `/health` poll |

## After it lands

```bash
curl http://localhost:6380/health         # smoke
curl http://localhost:6380/stats          # what's in there
journalctl -u heatherdb -f                # live logs
systemctl restart heatherdb               # after editing /etc/heatherdb/env
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
