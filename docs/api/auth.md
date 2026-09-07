# Auth endpoints

Session tokens. Basic auth pays an Argon2 verify (~11 ms) on every request; a
bearer token skips it. Both routes are exempt from the scope check — minting
and revoking your own token is an identity operation, so any authenticated
user may call them regardless of database scope.

See [Manage users and authentication](../how-to/manage-users.md) and
[Overview](overview.md).

### POST /auth/token

Mint a session token for the caller.

Auth: **Basic only.** A bearer-authenticated caller gets `403`, so a token
cannot extend its own lifetime without paying the Argon2 cost. Any scope.

Request body: no body.

The token inherits the user's scope *at mint time*; a later scope change does
not affect tokens already issued. TTL is 1 hour (`tokens::TOKEN_TTL`), not
configurable. Expired tokens are swept every 5 minutes.

```bash
curl -u alice:pw -X POST http://localhost:6380/auth/token
```

```json
{ "token": "…43 chars…", "token_type": "Bearer", "expires_at": 1767225600 }
```

`expires_at` is unix **seconds**.

| Status | Cause |
|---|---|
| `401 Unauthorized` | missing, malformed, or invalid credentials |
| `403 Forbidden` | the request authenticated with a bearer token |

### POST /auth/token/revoke

Revoke the bearer token that authenticated this request.

Auth: **Bearer only** in practice — there is no token to revoke on a
Basic-authenticated request. Any scope.

Request body: no body.

```bash
curl -H 'Authorization: Bearer eyJ…' \
  -X POST http://localhost:6380/auth/token/revoke
```

```json
{ "revoked": true }
```

`{"revoked": false}` when the token was already gone (expired or revoked
earlier) — revocation is idempotent.

| Status | Cause |
|---|---|
| `400 Bad Request` | the request used Basic auth, so it carries no token |
| `401 Unauthorized` | invalid or expired bearer token |
