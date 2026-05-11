# Authentication

HeatherDB uses **HTTP Basic Auth** with **per-database scopes**, on by
default. This page covers everything you need to set it up, run it, and
not get burned.

## Mental model

- A **user** has a name, a password (Argon2id-hashed), and a **scope**.
- A **scope** is either `Root` or `Database(name)`.
- An **HTTP request** is checked against the user's scope.
- `/health` is the only route that bypasses auth entirely.

```
       request                 user store              scope check
┌────────────────┐    ┌────────────────────────┐    ┌────────────────┐
│ Authorization: │ →  │ argon2id verify        │ →  │ is_authorized( │
│ Basic <b64>    │    │ $DATA/users.json       │    │   user.scope,  │
└────────────────┘    └────────────────────────┘    │   request.path)│
                                                    └────────────────┘
                                                          ↓
                                                       handler
```

## Scope rules

| Path pattern                        | `Root` | `Database("default")` | `Database("memoria")` |
|-------------------------------------|--------|------------------------|------------------------|
| `/health`                           | ✓      | ✓ (always allowed)     | ✓ (always allowed)     |
| `GET /db`, `POST /db`               | ✓      | —                      | —                      |
| `GET /db/{name}`, `DELETE /db/{name}` | ✓    | —                      | —                      |
| `/db/default/collections/...`       | ✓      | ✓                      | —                      |
| `/db/memoria/collections/...`       | ✓      | —                      | ✓                      |
| `/db/anything-else/...`             | ✓      | —                      | —                      |
| `/collections/...` (legacy alias)   | ✓      | ✓                      | —                      |
| `/algebra/...` (legacy alias)       | ✓      | ✓                      | —                      |
| `/compose/...` (legacy alias)       | ✓      | ✓                      | —                      |

The legacy non-`/db/` routes target `default` — only the `Root` user
and a user scoped to `Database("default")` can reach them.

`/db` (database management — list, create, drop) is **root-only**.

## First-boot bootstrap

The engine refuses to start without an auth scheme — but it won't
refuse to start when there are no users. Instead, **first boot
creates one for you** so you're never locked out.

Two paths:

### A. Env-driven (recommended for any non-laptop deploy)

Set both env vars before the first boot:

```bash
export HEATHER_ADMIN_USER=admin
export HEATHER_ADMIN_PASSWORD='your-strong-password'

heather_server --data-dir /var/lib/heatherdb --dimension 128
```

The user is created silently. A `tracing::info` line confirms which
env it came from. Both vars are ignored on subsequent boots — the
user is already in `users.json` with the password hashed.

### B. Random password (good for laptops + interactive boots)

Skip the env vars. The engine generates a 24-char URL-safe random
password and prints it once on stderr in a fenced ASCII box:

```
┌─────────────────────────────────────────────────────────────────────┐
│ HeatherDB ⋅ first-boot admin user created.                          │
│                                                                     │
│   user      admin                                                   │
│   password  2nD2YxAApJ4nL9cGzzfCVk9W                                │
│   scope     root                                                    │
│                                                                     │
│ Save this password — it's NOT printed again.                        │
└─────────────────────────────────────────────────────────────────────┘
```

Copy it now. It's hashed in `users.json` immediately and never
reprinted; if you lose it, your only recovery is to delete
`users.json` and let first-boot run again.

## The `user` CLI

`heather_server user` runs synchronously and exits — no HTTP server
starts.

```
heather_server user create <name> --password <pw> [--scope root|<db>]
heather_server user list
heather_server user delete <name>
heather_server user passwd <name> --password <new>
```

All commands take the same `--data-dir` / `HEATHER_DATA_DIR` as the
server.

### Examples

```bash
# A read-only-ish user scoped to the memoria database.
heather_server --data-dir /var/lib/heatherdb \
  user create memoria-app --password 'app-secret' --scope memoria

# List (never prints hashes).
heather_server --data-dir /var/lib/heatherdb user list
# NAME              SCOPE          CREATED
# admin             root           2026-05-11 09:20:16Z
# memoria-app       db:memoria     2026-05-11 09:21:00Z

# Rotate.
heather_server --data-dir /var/lib/heatherdb \
  user passwd memoria-app --password 'new-secret'

# Drop.
heather_server --data-dir /var/lib/heatherdb user delete memoria-app
```

### Constraints

| Field      | Rule |
|------------|------|
| Username   | 1..=64 chars, leading letter, then `[a-zA-Z0-9_.-]*`. |
| Password   | ≥ 8 characters. (No max — pass long passphrases.) |
| Scope      | `root` (or `*`) for unlimited; otherwise a database name. Database names follow the engine's [naming rules](./databases.md#names). |

Passwords are hashed with **Argon2id** (PHC-format string), never
stored in plaintext, never logged, never returned by any API.

## ⚠ TLS is mandatory in production

HTTP Basic Auth sends `Authorization: Basic base64(user:password)`. The
base64 step is **encoding, not encryption** — anyone who can read the
wire can decode it in one command:

```bash
echo 'YWRtaW46c2VjcmV0' | base64 -d
# admin:secret
```

This means:

| You're running…                              | Is it safe? |
|----------------------------------------------|-------------|
| Plain HTTP on `127.0.0.1` only               | Yes — loopback never leaves your machine. |
| Plain HTTP exposed on a LAN / VPC            | **No.** Anyone sniffing the network sees credentials in cleartext. |
| Plain HTTP on the public internet            | **Hell no.** Stop. Read on. |
| HTTPS (TLS) anywhere                         | Safe — TLS encrypts the whole request, including the `Authorization` header. |

### The fix: TLS-terminating reverse proxy

The engine speaks plain HTTP. Run it bound to loopback and put a TLS
proxy in front. Caddy is the painless option:

```bash
# 1. Bind the engine to loopback only.
HEATHER_HOST=127.0.0.1 \
HEATHER_DATA_DIR=/var/lib/heatherdb HEATHER_DIMENSION=128 \
HEATHER_ADMIN_USER=admin HEATHER_ADMIN_PASSWORD='…' \
  heather_server
```

```caddy
# 2. /etc/caddy/Caddyfile
api.heather.example.com {
  reverse_proxy 127.0.0.1:6380
  encode zstd gzip
}
```

```bash
# 3. Start Caddy. It auto-fetches a Let's Encrypt cert on first request.
sudo systemctl enable --now caddy
```

Now everyone hits `https://api.heather.example.com/...` — TLS to
Caddy, plain HTTP from Caddy to the engine over loopback, where
plaintext is fine because the packets never leave the kernel.

For a `docker compose` deploy, Fovea's Caddy container already does
this for you (see [`fovea/Dockerfile`](../fovea/Dockerfile)) — set
`HEATHER_HOST=127.0.0.1` on the engine and the engine becomes
unreachable except through Caddy.

### nginx alternative

```nginx
server {
    listen 443 ssl http2;
    server_name api.heather.example.com;

    ssl_certificate     /etc/letsencrypt/live/api.heather.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.heather.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:6380;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

`certbot --nginx -d api.heather.example.com` covers the cert.

### Native TLS in the engine?

Not built in, on purpose. Reverse proxies are operationally cheaper:

- Cert renewal handled by the proxy (Caddy + Let's Encrypt is
  zero-config; certbot is one cron job).
- Independent restart cadence — rotate certs without restarting the
  engine, restart the engine without re-touching certs.
- HTTP/2, gzip / zstd compression, response caching, request logging
  all live in one well-understood layer.

If you genuinely cannot run a proxy, terminate TLS at a load balancer
(every cloud has one). The engine itself never sees TLS material; that
keeps its threat surface tiny.

## Sending credentials

HTTP Basic Auth — `Authorization: Basic base64(user:password)`.

(Always over HTTPS in production. See above.)

```bash
curl -u admin:'your-password' http://127.0.0.1:6380/db
```

In code:

```python
import requests
r = requests.get("http://127.0.0.1:6380/db",
                 auth=("admin", "your-password"))
```

```ts
const res = await fetch("http://127.0.0.1:6380/db", {
  headers: {
    Authorization: "Basic " + btoa("admin:your-password"),
  },
});
```

```rust
let auth = base64::engine::general_purpose::STANDARD
    .encode(b"admin:your-password");
let resp = reqwest::Client::new()
    .get("http://127.0.0.1:6380/db")
    .header("Authorization", format!("Basic {auth}"))
    .send()
    .await?;
```

## Failure modes

| Status | Body                                                          | Cause |
|--------|---------------------------------------------------------------|-------|
| 401    | `{"error":"missing or malformed Authorization header"}`       | No `Authorization` header, or not `Basic`. Response carries `WWW-Authenticate: Basic realm="heatherdb"`. |
| 401    | `{"error":"invalid credentials"}`                             | Username unknown OR password wrong. The two cases are deliberately indistinguishable — and the engine runs an Argon2 verify against a decoy hash even when the user doesn't exist, so the timing matches too. |
| 403    | `{"error":"user 'X' (scope db:Y) is not authorised for /Z"}`  | Authenticated but the scope doesn't permit the path. |

## Disabling auth

For local dev, where you don't want to fiddle with credentials:

```bash
HEATHER_AUTH_DISABLED=1 heather_server --data-dir ~/scratch ...
# or
heather_server --auth-disabled --data-dir ~/scratch ...
```

The engine emits a one-time `WARN` at boot:

```
WARN heather_server: Auth: DISABLED (HEATHER_AUTH_DISABLED=1).
     Anyone who can reach this port has full read+write access. Drop
     the env var or the flag to re-enable.
```

**Do not run with auth disabled in production.** The engine has no
firewall, no rate limiting, and no audit log; the network boundary is
the only thing standing between an unauthenticated request and arbitrary
data destruction.

## Where users live on disk

`$HEATHER_DATA_DIR/system/data/` — its own LMDB env, sibling of every
per-database env at `$ROOT/db/<name>/data/`. One named sub-DB inside
called `users` (key = username, value = bincode of the user record).

Why LMDB instead of a JSON file: **CLI mutations are visible to a
running engine immediately**, no restart. `verify()` opens a fresh
read transaction per request, so `heather_server user create alice …`
in one terminal lets `alice` log in seconds later in another, even
while the engine is mid-flight.

The file is owner-readable on Unix (LMDB's default `0600` mode).

## Backup + restore

The system env is one of N LMDB envs under `$HEATHER_DATA_DIR/`. The
backup tooling already covers it:

```bash
heather_server --data-dir … backup create --output users-and-all.tar.gz
# tar contents include system/ alongside db/<name>/.
```

Live snapshot of just the user store:

```bash
mdb_copy $HEATHER_DATA_DIR/system/data /tmp/users-snap
```

To migrate creds to a new instance: snapshot or tar the `system/`
directory and drop it into the new data dir. Hashes are portable —
Argon2id parameters are encoded in the hash itself.

## What this isn't (yet)

- **No tokens / API keys** — every request is Basic Auth. JWT-style
  bearer tokens are out of scope until v0.3.
- **No row-level / collection-level scopes** — the smallest unit is a
  database. If you need finer access, split projects into more DBs.
- **No rate limiting** — the engine is happy to be DoS'd by an
  authenticated client. Put it behind a real reverse proxy in
  production.
- **No audit log** — every request is traced (`tracing` spans), but
  there's no per-user activity ledger. Scrape the logs.
