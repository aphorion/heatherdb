# Deploying on Coolify

Coolify is the canonical production deploy target. The repo's
`docker-compose.yml` is already Coolify-friendly — every tunable
is `${VAR:-default}`, the engine has a healthcheck, and the data
volume is named so Coolify can manage it.

This page walks the one-time setup, then the day-2 operations that
are different from a plain `docker compose` deploy.

## One-time: create the application

1. **New Resource → Application → Public Repository** (or Private if you
   wired your GitHub PAT into Coolify already).
2. Pick the repo (`aphorion/heather-db`). Branch: `main` (or pin a tag).
3. **Build Pack: `docker compose`**. Coolify auto-discovers
   `docker-compose.yml` at the repo root.
4. Save. Don't deploy yet.

## Set environment variables

In the application's **Environment Variables** tab, add at minimum:

| Variable                   | Value                          | Notes |
|----------------------------|--------------------------------|-------|
| `HEATHER_ADMIN_USER`       | `admin` (or whatever)          | First-boot admin name. |
| `HEATHER_ADMIN_PASSWORD`   | a strong password              | Mark as **secret**. Used only on first boot — after that, rotate via `heather user passwd …`. |
| `HEATHER_DIMENSION`        | `128`                          | Default DB dimension. Other DBs set their own at create time. |
| `HEATHER_MAP_SIZE_MB`      | `4096`                         | LMDB map size for the default DB. |
| `HEATHER_REQUEST_TIMEOUT`  | `600`                          | Per-request seconds. Bump for big algebra. |
| `RUST_LOG`                 | `info`                         | `debug` for verbose. |

If you don't set `HEATHER_ADMIN_PASSWORD`, the engine generates a 24-char
random one on first boot and prints it once on stderr. Recover it from
**Logs** in Coolify with `grep -A8 first-boot`.

**Don't set `HEATHER_HOST`** — Coolify's proxy reaches the engine on
the docker network where `0.0.0.0` (the default) is correct. Setting
it to `127.0.0.1` would lock Coolify out.

## Assign a domain

1. **General → Domains** on the app page.
2. Add a domain pointed at port `6380`. Coolify provisions a
   Let's Encrypt cert via Traefik automatically.
3. Save.

Now the engine is reachable at `https://your-domain/...` over TLS, with
HTTP Basic Auth required:

```bash
curl -u admin:'your-password' https://your-domain/db
```

`/health` is the only unauthenticated route — useful for Coolify's
own healthcheck pings (it already uses the in-container healthcheck
from the compose file, but having an open `/health` is also safe).

## Deploy

Hit **Deploy**. Coolify clones the repo, runs `docker compose build`
(uses the multi-stage `Dockerfile` — first build pulls Rust 1.84, ~5
minutes; subsequent builds with cache are seconds), then `docker
compose up -d`.

Watch **Logs** for the boot banner. If you supplied
`HEATHER_ADMIN_PASSWORD`, you'll see:

```
INFO heather_server::auth: Auth: bootstrapped admin user from
     HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD  user="admin"
```

Otherwise the random-password fenced box.

## Day-2 operations

### Run an `heather` subcommand inside the container

Coolify exposes a **Terminal** for the container. Or via SSH on the host:

```bash
docker exec -it heatherdb heather user list
docker exec -it heatherdb heather user create alice --password '…' --scope memoria
docker exec -it heatherdb heather user passwd admin --password '…'
docker exec -it heatherdb heather backup create --output /var/lib/heatherdb/backups/h-$(date +%F).tar.gz
docker exec -it heatherdb heather snapshot create --db default
```

Note: writes to `/var/lib/heatherdb/...` land in the named volume, so
backups + snapshots persist across redeploys. Pull them off-host with:

```bash
docker cp heatherdb:/var/lib/heatherdb/backups/h-2026-05-11.tar.gz .
```

### Rotate the admin password

```bash
docker exec -it heatherdb heather user passwd admin --password '<new>'
```

The engine sees the new hash on the next request — no restart needed.

### Add a scoped user (e.g. one DB per app)

```bash
docker exec -it heatherdb heather user create memoria-app \
  --password '<service-password>' --scope memoria
```

Now your app authenticates as `memoria-app` and can only touch
`/db/memoria/...`. See [auth.md → Scope rules](./auth.md#scope-rules).

### Backups

Two flows depending on your appetite for downtime:

**Cold (engine pause not visible to clients on a healthy connection):**

```bash
docker exec heatherdb heather backup create \
  --output /var/lib/heatherdb/backups/full-$(date +%F).tar.gz
```

**Live (engine keeps serving writes):**

```bash
docker exec heatherdb heather snapshot create --db default
docker exec heatherdb heather snapshot create --db memoria
# … one per database; live, consistent, runs against an active env.
```

For scheduled backups inside Coolify, add a cron in the **Scheduled Backups
of Persistent Storages** feature (Coolify Pro), or set up a host-level
systemd timer that runs the `docker exec` above.

### Upgrades

```bash
git push origin main         # or merge a PR
```

Coolify auto-deploys (if you've enabled webhooks) or hit **Redeploy**.
The build is incremental — Rust caches the dep graph, so only the
changed crates rebuild.

If the upgrade introduces a breaking change to LMDB layout (rare, called
out in commit messages), Coolify will fail health checks and roll back
to the previous container by default — check the rollout log.

### Snapshots before a risky migration

```bash
docker exec heatherdb heather snapshot create --db default
docker exec heatherdb heather snapshot create --db memoria
# … then redeploy. If something goes sideways:
docker exec heatherdb heather restore restore \
  --input /var/lib/heatherdb/snapshots/default-1730000000
```

## TLS

Coolify's built-in Traefik handles TLS for any domain you assign.
You don't need a separate Caddy or certbot — Traefik fetches the cert
on first request and renews automatically.

The engine itself **does not speak TLS**. Coolify reaches it over the
docker network in plaintext, then terminates TLS at its edge. This is
the right shape; never bind the container's `:6380` to the public host
port if you've assigned a domain via Coolify.

## Storage layout inside the volume

Coolify mounts the named volume `heatherdb_data` at
`/var/lib/heatherdb` inside the container. Inside that:

```
/var/lib/heatherdb/
├── server.toml             # tag file
├── system/data/            # LMDB env: HTTP Basic Auth user store
├── db/<name>/              # per-database state (one LMDB env per)
├── snapshots/              # default landing zone for `snapshot create`
└── backups/                # convention only — Coolify doesn't enforce
```

When you take a Coolify volume backup, all of this lands in one
artefact. Same for restore: Coolify's restore wipes the volume and
extracts back into place.

## What this isn't (yet)

- **No multi-node** — one engine container per Coolify app. For
  bigger workloads, vertical-scale the host VM (LMDB is single-process,
  multi-reader).
- **No Coolify-driven user provisioning** — the `heather user *` CLI
  is the only way to manage credentials; Coolify can't see into the
  LMDB user store.
- **No metrics endpoint** for Coolify's monitoring widget — use
  `journalctl`-equivalent (Coolify Logs) and the engine's own
  `/db` listing for per-DB counts. A `/metrics` endpoint is on the
  v0.3 roadmap.
