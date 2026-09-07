# Dream endpoint

On-demand consolidation. One route, and no legacy form — dreaming is
addressed by database only. See [Dreaming](../explanation/dreaming.md).

### POST /db/{db}/dream

Trigger idle-time consolidation now, ignoring the idle gate and the
`dream.enabled` opt-in: a manual trigger should not require the database to
have opted in. Runs synchronously and answers with what it did.

Auth: Basic or Bearer, root or `Database(db)`.

Request body: no body. Query parameters override the persisted `[dream]`
config for this call only:

| Param | Type | Meaning |
|---|---|---|
| `mode` | `replay` \| `ladder` | Reorganise in place, or build `<name>__L1`, `__L2`… derived collections. Anything other than `ladder` (case-insensitive) is read as `replay` |
| `levels` | integer | Replay: granularity passes. Ladder: levels to climb |
| `tau_cohere` | float | Ladder: minimum counter coherence to join a family |
| `tau_split` | float | Ladder: minimum address similarity for a candidate family |
| `collections` | comma-separated string | Restrict to these collections. Default: all |

```bash
curl -u admin:pw -X POST \
  'http://localhost:6380/db/movies/dream?mode=replay&levels=1&collections=taste'
```

```json
{ "database": "movies",
  "collections": [
    { "collection": "taste", "passes": 1, "locations_before": 812,
      "locations_after": 806, "merged": 6,
      "tau_split": 0.0, "tau_cohere": 0.0 }
  ] }
```

`tau_split` and `tau_cohere` are the boundaries the ladder actually used
(auto-calibrated when the caller left them unset); both are `0.0` for replay
passes.

The call blocks for the length of the dream, so it is subject to
`--request-timeout` (default 30s) on a large database. The background dream
loop is unaffected by this route.

| Status | Cause |
|---|---|
| `403 Forbidden` | the caller is scoped to a different database |
| `404 Not Found` | no such database |
| `408 Request Timeout` | the dream outlasted `--request-timeout` |
| `500 Internal Server Error` | storage failure during consolidation |
