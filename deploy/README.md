# deploy/

One-shot deployment scripts for HeatherDB. Currently:

| Script | What it does |
|--------|--------------|
| [`deploy-vps.sh`](./deploy-vps.sh)              | SSH-deploy to a $5 VPS (Hetzner / DO / Linode / etc.). Creates user + dirs + env file, builds the binary on the VPS (or uploads a cross-built one with `--upload`), drops the systemd unit, smoke-tests `/health`. |
| [`systemd/heatherdb.service`](./systemd/heatherdb.service) | Hardened systemd unit installed by the script. Override env in `/etc/heatherdb/env`. |

## deploy-vps.sh

```bash
# Build on the VPS (slow first run while it installs Rust)
./deploy-vps.sh user@vps.example.com

# Or cross-build locally and scp the binary up (needs `cargo install cross`)
./deploy-vps.sh user@vps.example.com --upload
```

Useful env overrides:

```bash
HEATHER_VERSION=v0.1.0 \
HEATHER_DIM=384 \
INSTALL_DIR=/srv/heatherdb \
SERVICE_USER=heather \
  ./deploy-vps.sh user@vps.example.com
```

The script is idempotent — re-run it any time to push a new build. The
`/etc/heatherdb/env` file isn't overwritten on subsequent deploys, so your
config tweaks survive.

For the Raspberry Pi cross-compile flow (different target, different deploy
cadence), see [`heatherdb-pi-demo/deploy/`](https://github.com/aphorion/heatherdb-pi-demo).
