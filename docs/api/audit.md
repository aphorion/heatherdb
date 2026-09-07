# Audit endpoint

The per-database access log. One route, and no legacy form — the audit log is
addressed by database only. See
[The audit log and privacy](../explanation/audit-and-privacy.md) and
[Read the audit log](../how-to/read-the-audit-log.md).

### GET /db/{db}/audit

The access log, newest first.

Auth: Basic or Bearer. **Root only by default**; a database can opt its own
users in with `visibility = "DbUsers"` under `[audit]` in its `db.toml`. The
middleware is the only place that sees both the caller's scope and the
database's config, so this is enforced there, not in the handler.

Request body: no body. Filters are query parameters:

| Param | Type | Default | Meaning |
|---|---|---|---|
| `user` | string | — | Exact username match |
| `collection` | string | — | Exact collection match |
| `route` | string | — | Logical route: `read`, `attention`, `analyze`, `documents/query` |
| `since` | integer | — | Inclusive lower bound, unix **milliseconds** |
| `until` | integer | — | Inclusive upper bound, unix milliseconds |
| `limit` | integer | `100` | Clamped to `1..=1000` |

```bash
curl -u admin:pw \
  'http://localhost:6380/db/movies/audit?user=alice&route=read&limit=50'
```

```json
{ "database": "movies", "count": 2, "limit": 100,
  "entries": [
    { "seq": 41, "timestamp_ms": 1767225600123, "user": "alice",
      "scope": "db:movies", "database": "movies", "collection": "taste",
      "route": "read", "status": 200, "result_count": 1,
      "location_ids": [], "document_ids": [],
      "query_hash": "9f2c1ab30de4f781", "query_raw": null }
  ] }
```

`limit` in the response is the clamped value actually applied. `query_raw` is
`null` unless the database sets `store_raw_query = true`.

Only reads are recorded — `read`, `attention`, `analyze` and
`documents/query`. Writes, algebra and `batch_analyze` leave no entry.
Records are staged in a buffer and flushed on a threshold, on a timer, and on
graceful shutdown; a read here flushes first, so it never misses a record the
engine has already accepted.

| Status | Cause |
|---|---|
| `403 Forbidden` | a database-scoped caller on a database whose `visibility` is not `DbUsers` |
| `404 Not Found` | no such database |
| `409 Conflict` | audit logging is disabled for that database (`[audit] enabled = false`) |
| `500 Internal Server Error` | the log could not be flushed or read |
