# HTTP API overview

The front door to the HTTP API section. Everything here applies to every
endpoint; the per-resource pages document only what is specific to a route.

Verified against `heather_server` 0.4.1. Bodies are JSON
(`Content-Type: application/json`); responses are JSON.

Source of truth: `heather_server/src/main.rs` (route table),
`heather_server/src/routes.rs` and `routes_db.rs` (handlers),
`heather_server/src/models.rs` (wire types).

## Pages

| Page | Covers |
|---|---|
| [Ops](ops.md) | `/health`, `/healthz`, `/ready`, `/readyz` |
| [Auth](auth.md) | session tokens |
| [Databases](databases.md) | `/db` management |
| [Collections](collections.md) | create, list, drop, stats, config, locations, compress, fingerprint |
| [Writes](writes.md) | `write`, `bulk_load` |
| [Reads](reads.md) | `read`, `attention`, `analyze` |
| [Documents](documents.md) | document storage and similarity search |
| [Algebra](algebra.md) | algebra over collections, `compose/read` |
| [Vectors](vectors.md) | stateless `/vec/*` math |
| [Audit](audit.md) | the access log |
| [Dream](dream.md) | on-demand consolidation |

A complete method+path index lives in
[the HTTP API reference](../reference/http-api.md).

## The two route shapes

Every collection and algebra route exists twice:

| Shape | Prefix | Targets |
|---|---|---|
| **Legacy** | `/collections/...`, `/algebra/...`, `/compose/...` | the `default` database |
| **Scoped** | `/db/{db}/collections/...`, `/db/{db}/algebra/...`, `/db/{db}/compose/...` | the named database |

The two are the same handler: the scoped shape resolves a database from the
path and delegates to the legacy one (`routes_db.rs` wraps `routes.rs`).
Behaviour, request bodies and responses are identical.

**This section documents the scoped form.** Each endpoint's heading carries the
`/db/{db}/...` path, and the body names the legacy equivalent. To use the
legacy form, drop the `/db/{db}` prefix; you get the `default` database.

The `/vec/*` routes are **not** mirrored under `/db/{db}/` — they are stateless
vector math and never touch a database.

## Authentication

HTTP Basic on every route except `/health`, `/healthz`, `/ready`, `/readyz`
(`auth::is_unauth_route`). A bearer token from
[`POST /auth/token`](auth.md#post-authtoken) may be sent instead. See
[Manage users and authentication](../how-to/manage-users.md).

```
Authorization: Basic base64(user:password)
Authorization: Bearer <token>
```

## Scope

A user is `root` or scoped to one database.

| Path | Root | `Database(name)` | `Database("default")` |
|---|---|---|---|
| `/health`, `/healthz`, `/ready`, `/readyz` | ✅ (no auth) | ✅ | ✅ |
| `/auth/token`, `/auth/token/revoke` | ✅ | ✅ | ✅ |
| `/db` | ✅ | ❌ 403 | ❌ 403 |
| `/db/{name}/...` | ✅ | ✅ only when `name` matches | ✅ only for `default` |
| `/db/{name}/audit` | ✅ | only if that database sets `visibility = "DbUsers"` | same |
| `/collections/...`, `/algebra/...`, `/compose/...` | ✅ | ❌ 403 | ✅ |
| `/vec/...` | ✅ | ❌ 403 | ❌ 403 |

`/auth/*` is exempt from the scope check entirely — minting and revoking your
own token is an identity operation, not a data one (`auth::middleware`).

`/vec/*` is root-only. The classifier admits a `default`-scoped user to the
`/collections`, `/algebra` and `/compose` prefixes only; `/vec` is not on that
list (`heather_server/src/users.rs::is_authorized`).

## Errors

Failures return `{"error": "<message>"}` with one of:

| Status | Meaning |
|---|---|
| `400 Bad Request` | malformed body, dimension mismatch, unknown strategy, a cap exceeded, an engine validation error |
| `401 Unauthorized` | missing, malformed, or invalid credentials; expired bearer token. Carries `WWW-Authenticate: Basic realm="heatherdb", charset="UTF-8"` |
| `403 Forbidden` | authenticated but out of scope; dropping the `default` database; minting a token with a bearer token |
| `404 Not Found` | no such database, collection, document, or source collection |
| `408 Request Timeout` | exceeded `--request-timeout` (default 30s) |
| `409 Conflict` | database name already exists; a concurrent write invalidated an algebra or `bulk_load` target (retryable); audit read on a database with logging disabled |
| `413 Payload Too Large` | body over `--max-body-size` (default 2 MiB) |
| `500 Internal Server Error` | storage failure or a panicked worker task |
| `503 Service Unavailable` | `/ready` could not reach the storage layer |

Every scoped route returns `404` with `database not found: {db}` when the `db`
segment names no mounted database. That case is not repeated per endpoint.

## Create-on-write asymmetry

Note the asymmetry between reads and writes: `write`, `read`, `attention`,
`analyze`, `batch_analyze`, `documents/query` and `bulk_load` call
`get_or_create_collection`, so an unknown collection is **created**, not a 404.
`stats`, `config`, `locations`, `fingerprint`, `compress`, `documents` and the
algebra source collections require the collection to exist and return 404.

## Server-side caps

Bodies are bounded by `--max-body-size`, but a small body can describe a large
computation. Three caps bound the *result*; `0` disables one
(`heather_server/src/limits.rs`).

| Cap | Flag / env | Default | Applies to |
|---|---|---|---|
| Algebra result locations | `--max-algebra-locations` / `HEATHER_MAX_ALGEBRA_LOCATIONS` | 250 000 | `algebra/add`, `sub`, `bind`, and `intersect` with a non-positive threshold |
| Bulk-load items | `--max-bulk-items` / `HEATHER_MAX_BULK_ITEMS` | 100 000 | `bulk_load` |
| Batch queries | `--max-batch-queries` / `HEATHER_MAX_BATCH_QUERIES` | 1 000 | `batch_analyze` |

See [Configuration](../reference/configuration.md) for every knob.
