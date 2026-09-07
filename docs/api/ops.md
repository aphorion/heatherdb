# Ops endpoints

Liveness and readiness. These four are the only unauthenticated routes: the
auth middleware whitelists them before it looks at any header
(`auth::is_unauth_route`). See [Overview](overview.md) for conventions.

### GET /health

Liveness. Never touches storage — it answers as long as the process is
serving.

Auth: none.

Request body: no body.

```bash
curl http://localhost:6380/health
```

```json
{ "status": "ok" }
```

Non-200: none. If the process is up, this is `200`.

### GET /healthz

Alias of [`GET /health`](#get-health) — same handler, same response. Present
for orchestrators that expect the `z` spelling.

Auth: none.

Request body: no body.

```bash
curl http://localhost:6380/healthz
```

```json
{ "status": "ok" }
```

Non-200: none.

### GET /ready

Readiness. Lists databases and performs a real LMDB read against the `default`
hive, so it fails when the storage layer is broken even though the process
still answers `/health`. Point an orchestrator's readiness probe here and its
liveness probe at `/health`: traffic stops being routed instead of the engine
being restart-looped.

Auth: none.

Request body: no body.

```bash
curl http://localhost:6380/ready
```

```json
{ "status": "ok" }
```

| Status | Cause |
|---|---|
| `503 Service Unavailable` | listing databases failed, the `default` database is missing, or its collection list could not be read |

### GET /readyz

Alias of [`GET /ready`](#get-ready) — same handler, same checks, same
statuses.

Auth: none.

Request body: no body.

```bash
curl http://localhost:6380/readyz
```

```json
{ "status": "ok" }
```

| Status | Cause |
|---|---|
| `503 Service Unavailable` | the storage layer could not be reached |
