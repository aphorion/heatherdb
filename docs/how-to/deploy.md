# Deploy HeatherDB

HeatherDB is a single Rust binary with no GPU, no accelerator, and no system
dependency beyond libc. Every install path below lands on the same layout —
`/usr/bin/heather`, `/var/lib/heatherdb`, `/etc/heatherdb/env` — so they are
interchangeable and you can move between them later.

## Pick a path

| You want | Use |
|---|---|
| A container, locally or on any Docker host | [Docker](#docker) |
| A managed PaaS deploy with TLS handled for you | [Compose / Coolify](#compose--coolify) |
| A Debian or Ubuntu box, pre-built | [`.deb` package](#deb-package) |
| Any Linux VPS, built from source, systemd-managed | [Bare metal](#bare-metal) |
| A Kubernetes cluster | [Kubernetes](#kubernetes) |

Whichever you pick, read [Before you expose it](#before-you-expose-it) at the
bottom.

## Docker

```bash
docker pull aphorion/heatherdb:latest
```

Images are multi-arch (`linux/amd64` + `linux/arm64`), cross-compiled natively
rather than under QEMU emulation, and signed with cosign keyless:

```bash
cosign verify aphorion/heatherdb:latest \
  --certificate-identity-regexp "^https://github.com/aphorion/heatherdb/" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Run it:

```bash
docker run -d --name heatherdb \
  -p 6380:6380 \
  -v heatherdb_data:/var/lib/heatherdb \
  -e HEATHER_ADMIN_USER=admin \
  -e HEATHER_ADMIN_PASSWORD='a-real-password' \
  aphorion/heatherdb:latest
```

The named volume is not optional in any deploy you care about — without it the
data directory dies with the container.

To build locally instead, `docker build -t heatherdb:latest .`. There is also a
`Dockerfile.slim`.

## Compose / Coolify

The bundled `docker-compose.yml` is the canonical
[Coolify](https://coolify.io) deploy path, and works unchanged for plain local
Compose.

```bash
cp .env.example .env
${EDITOR:-vi} .env          # set HEATHER_ADMIN_PASSWORD
docker compose up -d
docker compose logs -f
```

Every tunable in the file is written `${VAR:-default}`, so Coolify's
Environment Variables UI can override any of them without editing the file.

To deploy on Coolify: import the compose file as a **Docker Compose
application** (HeatherDB is semantically a database but operationally an
application — Coolify's "Database" resources are only its own managed
services), set the environment variables in the UI, and assign a domain to
port 6380. Coolify's Traefik proxy fronts the engine and manages TLS.

Do not introduce host-port hardcoding or hostname assumptions in the compose
file; they break behind Traefik.

The container healthcheck hits `/ready`, not `/health` — `/ready` performs a
real LMDB read and returns 503 when storage is broken, whereas `/health` stays
green on a corrupted engine. That is what drives Coolify's app-up status.

If you did not set `HEATHER_ADMIN_PASSWORD`, recover the generated one from the
log:

```bash
docker compose logs heatherdb 2>&1 | grep -A8 first-boot
```

## `.deb` package

```bash
curl -fsSLO https://github.com/aphorion/heatherdb/releases/latest/download/heatherdb_<version>_amd64.deb
sudo apt install ./heatherdb_<version>_amd64.deb
```

This installs a `heatherdb` systemd service, a dedicated system user, and
generates admin credentials into `/etc/heatherdb/env` on first install. Read
that file for the password, then start the service:

```bash
sudo cat /etc/heatherdb/env
sudo systemctl start heatherdb
```

See [`packaging/deb/`](../../packaging/deb/).

## Bare metal

One idempotent script installs system dependencies, Rust if missing, the
`heatherdb` user, the binary, the same systemd unit the `.deb` ships, and runs
a `/health` smoke test:

```bash
curl -fsSL https://raw.githubusercontent.com/aphorion/heatherdb/main/deploy/deploy | sudo bash
```

Re-run it to update the binary or apply config changes. Environment overrides
(`HEATHER_DIM`, `DATA_DIR`, `SERVICE_USER`, and more) are documented in
[`deploy/README.md`](../../deploy/README.md).

After it lands:

```bash
systemctl status heatherdb
journalctl -u heatherdb -f
sudo systemctl restart heatherdb     # after editing /etc/heatherdb/env
```

## Kubernetes

```bash
helm install heatherdb ./charts/heatherdb
```

A single-replica `StatefulSet` with a `volumeClaimTemplate`, a headless
Service, a client `ClusterIP` Service, and optional Ingress.

**There is no `replicaCount`.** LMDB is single-writer, like Postgres — adding
replicas against one volume does not scale reads, it corrupts data. The chart
always deploys exactly one pod.

Supply admin credentials via an existing Secret rather than Helm values, so the
password stays out of release history:

```bash
kubectl create secret generic heatherdb-auth \
  --from-literal=HEATHER_ADMIN_USER=admin \
  --from-literal=HEATHER_ADMIN_PASSWORD='...'
helm install heatherdb charts/heatherdb --set auth.existingSecret=heatherdb-auth
```

Full values reference: [`charts/heatherdb/README.md`](../../charts/heatherdb/README.md).

## Terminate TLS in front

HeatherDB speaks plain HTTP and does not terminate TLS. Put a reverse proxy in
front of it. Caddy gives you automatic certificates in four lines:

```caddy
api.heather.example.com {
  reverse_proxy 127.0.0.1:6380
  encode zstd gzip
}
```

nginx, Traefik and a cloud load balancer all work equally well. On Coolify and
most Kubernetes ingress setups this is already handled for you.

## Before you expose it

The short version is below; [Run it in production](production.md) is the long
version, with the checklist.

- **Set a real admin password.** If you do not, the engine generates one and
  prints it once — recoverable from `$HEATHER_DATA_DIR/initial-admin-password`
  (mode 0600), which you should delete after copying.
- **Never set `HEATHER_AUTH_DISABLED`** anywhere reachable from a network.
  Anyone who can open the port gets full read and write access. The engine
  emits a loud warning at boot in that mode; do not learn to ignore it.
- **Raise `--request-timeout` if you run large algebra ops.** The default is
  30 seconds; the bundled compose file sets 600 for exactly this reason.
- **Size the map ceiling per database, not globally.** `--map-size-mb` sizes
  only the auto-created `default` database, and only on the boot that creates
  it. See [Tune a database](tune-a-database.md#map-size).
- **Only one engine process per data directory.** The engine takes an exclusive
  advisory lock (`engine.lock`) and refuses to double-mount; two engines on one
  data directory corrupt LMDB.
- **Stop it gracefully.** SIGTERM and Ctrl-C flush the staged audit records; a
  SIGKILL loses whatever is still buffered.
