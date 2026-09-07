# Manage users and authentication

HeatherDB uses HTTP Basic Auth, on by default, backed by an LMDB-resident user
store at `$HEATHER_DATA_DIR/system/data/`. Passwords are hashed with Argon2id.

The store is shared between the CLI and the running engine, and every request
opens a fresh read transaction — so a user you create with the CLI works
immediately, with no restart.

## Set the first admin

On the first boot of an empty data directory the engine mints exactly one
root-scoped user.

**Preferred — supply the credentials:**

```bash
HEATHER_ADMIN_USER=admin HEATHER_ADMIN_PASSWORD='a-real-password' heather --data-dir /var/lib/heatherdb
```

Nothing is printed or written; you already have the password. It must be at
least 8 characters — the engine refuses to boot on a shorter one rather than
minting a weak account:

```
error: bootstrap admin user: password must be at least 8 characters
```

**Otherwise — the engine generates one:**

A 24-character password is generated, printed once in a box on stderr, and
written to `$HEATHER_DATA_DIR/initial-admin-password` (mode 0600) because
stderr is too easy to lose in container log capture.

```bash
cat /var/lib/heatherdb/initial-admin-password
# user = "admin"
# password = "…"
rm /var/lib/heatherdb/initial-admin-password    # once you have saved it
```

Both environment variables are ignored on every subsequent boot. To change the
admin username later, create the new user and delete the old one.

## Create a scoped user

```bash
heather --data-dir /var/lib/heatherdb user create alice \
  --password 'a-real-password' --scope movies
```

`--scope` is either `root` (or `*`) for full access, or a database name.
Usernames must match `[a-zA-Z][a-zA-Z0-9_.-]{0,63}`; passwords must be at least
8 characters.

What a scope reaches:

| Scope | Can reach |
|---|---|
| `root` | everything |
| `movies` | `/db/movies/...` only |
| `default` | `/db/default/...`, plus the legacy `/collections/...`, `/algebra/...`, `/compose/...` |

Two things a non-root scope never reaches: `/db` itself (database management is
root-only), and `/vec/*` (the stateless vector-math routes are root-only too —
they are not on the legacy prefix list).

By default a scoped user also cannot read `/db/{name}/audit`; see
[Read the audit log](read-the-audit-log.md#let-a-databases-own-users-read-it).

## List, rotate, delete

```bash
heather --data-dir /var/lib/heatherdb user list
heather --data-dir /var/lib/heatherdb user passwd alice --password 'new-password'
heather --data-dir /var/lib/heatherdb user delete alice
```

`user list` prints name, scope and creation time — never hashes.

In a container, run the same commands through the engine's own container:

```bash
docker compose exec heatherdb heather user passwd admin --password '…'
```

The `--data-dir` is already set in the container environment.

## Authenticate a request

```bash
curl -u alice:a-real-password http://localhost:6380/db/movies/collections
```

Or explicitly:

```
Authorization: Basic base64(alice:a-real-password)
```

The user store always runs an Argon2 verify — against a decoy hash when the
username does not exist — so response timing does not reveal whether a username
is valid.

## Use a session token for bulk work

Argon2 verification costs roughly 11 ms per request, which dominates a read
that takes about 2 ms of engine time. If you are about to issue thousands of
reads, pay it once:

```bash
TOKEN=$(curl -s -u alice:a-real-password -X POST http://localhost:6380/auth/token \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

curl -H "Authorization: Bearer $TOKEN" \
     -X POST http://localhost:6380/db/movies/collections/taste/read \
     -H 'Content-Type: application/json' -d '{"query":[…]}'
```

Things to know:

- **Minting requires Basic auth.** A bearer-authenticated caller gets `403`, so
  a token cannot renew itself without paying Argon2 again.
- **The TTL is one hour** and is not configurable.
- **The scope is frozen at mint time.** Changing a user's scope afterwards does
  not affect tokens already issued — revoke them, or wait for expiry.
- **Tokens are 32 random bytes, base64url-encoded.** Treat them like passwords:
  do not log them, do not share one across clients.
- Tokens survive a restart. Expired ones are swept at boot and every 5 minutes.

Revoke the token you are currently using:

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:6380/auth/token/revoke
```
```json
{"revoked": true}
```

There is no bulk revoke and no "revoke another user's tokens". If you need that
today, stop the engine and delete `$HEATHER_DATA_DIR/system/data`'s `tokens`
sub-database — which also deletes the user store, so recreate the users after.

## Disable auth for local development

```bash
HEATHER_AUTH_DISABLED=1 heather --data-dir /tmp/scratch
```

Or `--auth-disabled`. Every route except the health checks becomes open, and
audit entries record the user as `-`. The engine emits a loud warning at boot.

Never do this on anything reachable from a network.

## What is not in the model

The scope enum has exactly two levels and no per-user, per-route or role
dimension. There is no read-only scope, no per-collection permission, no group
membership, and no way to express "read your own audit entries but not your
colleagues'". If you need finer control, put an authorising proxy in front of
the engine.
