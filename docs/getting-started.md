# Getting started

This page takes you from a clean machine to a HeatherDB engine answering
authenticated requests in about five minutes.

## What you're about to set up

```
┌────────┐                ┌─────────────────────┐
│ curl / │  HTTP Basic    │ heather_server      │
│ Fovea  │ ─────────────▶ │  ↳ data dir         │
│ ...    │                │  ↳ system/  (users) │
│        │                │  ↳ db/<name>/ (data)│
└────────┘                └─────────────────────┘
```

One process, one data dir. Inside: a `system/` LMDB env that holds the
HTTP Basic Auth credentials, and a `db/` directory holding one more
LMDB env per database. That's the whole footprint.

## Prerequisites

- Rust 1.84+ (`rustup install stable`)
- A writable directory for engine state (we'll use `/tmp/heatherdb`
  for this walkthrough; **don't do that in production** — it's
  ephemeral)

## 1. Build the engine

```bash
git clone https://github.com/aphorion/heather-db
cd heather-db
cargo build --release -p heather_server
```

The binary lands at `target/release/heather_server`.

## 2. Boot it

The simplest setup uses two env vars to seed the bootstrap admin
user — no random-password log-grepping required:

```bash
mkdir -p /tmp/heatherdb

HEATHER_DATA_DIR=/tmp/heatherdb \
HEATHER_DIMENSION=128 \
HEATHER_ADMIN_USER=admin \
HEATHER_ADMIN_PASSWORD='change-me-immediately' \
  ./target/release/heather_server
```

Logs you should see:

```
INFO heather_server: Opened server  databases=1 names=["default"]
INFO heather_server::auth: Auth: bootstrapped admin user from
     HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD  user="admin"
INFO heather_server: Auth: HTTP Basic enabled  users=1
INFO heather_server: HeatherDB server listening  addr=0.0.0.0:6380
```

If you skip the env vars, the engine generates a random 24-char
password and prints it once in a fenced box on stderr — copy it.
After first boot the password lives Argon2-hashed inside the `system/`
LMDB env; the random banner never shows again. Rotate via
`heather_server user passwd admin --password '…'` — and because the
user store is LMDB-backed, the running engine sees the new password
immediately, no restart.

## 3. First request

`/health` is always open — useful for liveness probes:

```bash
curl http://127.0.0.1:6380/health
# {"status":"ok"}
```

Anything else needs auth:

```bash
curl http://127.0.0.1:6380/db
# {"error":"missing or malformed Authorization header"}    (401)

curl -u admin:change-me-immediately http://127.0.0.1:6380/db
# {"databases":[{"name":"default","created_at":...,"dimension":128,...}]}
```

## 4. Write a vector, read it back

Every collection lives inside a database. The `default` database was
auto-created on first boot. Write into it:

```bash
curl -u admin:change-me-immediately \
  -X POST http://127.0.0.1:6380/db/default/collections/playground/write \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [[0.1, 0.2, 0.3, 0.4, ...]]}'
# {"count":1}
```

(The vector must have 128 elements to match the database's dimension.)

Read it back:

```bash
curl -u admin:change-me-immediately \
  -X POST http://127.0.0.1:6380/db/default/collections/playground/read \
  -H 'Content-Type: application/json' \
  -d '{"query": [0.1, 0.2, 0.3, ...], "strategy": "iterative"}'
# {"result":[…the converged attractor…]}
```

The legacy alias works too — `/collections/playground/read` is
shorthand for `/db/default/collections/playground/read`. New code
should use the explicit form for clarity.

## 5. Make a second database

Different project, different vector dimension? One engine handles both:

```bash
curl -u admin:change-me-immediately \
  -X POST http://127.0.0.1:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"memoria","dimension":384}'
# {"name":"memoria","dimension":384,...}
```

Now `default` (dim=128) and `memoria` (dim=384) live side-by-side on
the same process. Vectors written to one never touch the other.

## 6. (Optional) make a scoped user

Don't share the admin credential. Make a per-app user:

```bash
./target/release/heather_server --data-dir /tmp/heatherdb user create memoria-app \
  --password 'app-secret' --scope memoria

# Verify:
./target/release/heather_server --data-dir /tmp/heatherdb user list
# NAME              SCOPE          CREATED
# admin             root           2026-05-11 09:20:16Z
# memoria-app       db:memoria     2026-05-11 09:21:00Z
```

`memoria-app` can do anything inside `/db/memoria/...` and nothing else
— `/db` management, other databases, and the legacy `/collections/...`
routes are all 403.

## 7. (Optional) take a backup

Three subcommands cover the lifecycle:

```bash
# Cold full-stack tarball (DBs + users + config).
./target/release/heather_server --data-dir /tmp/heatherdb \
  backup create --output /tmp/heather-$(date +%F).tar.gz

# Live consistent snapshot of one DB while the engine serves.
./target/release/heather_server --data-dir /tmp/heatherdb \
  snapshot create --db memoria

# Bring a tarball or snapshot back into a (stopped-engine) data dir.
./target/release/heather_server --data-dir /tmp/heatherdb \
  restore restore --input /tmp/heather-2026-05-11.tar.gz
```

See [operations.md → Backups](./operations.md#backups) for scheduling
+ disaster-recovery recipes.

## Where to go next

- Setting up real auth → [Authentication](./auth.md)
- More on the database layer → [Databases](./databases.md)
- Every HTTP route + scope rule → [API](./api.md)
- Production deploy → [Operations](./operations.md)
- Visual operator → [Fovea](./fovea.md)
