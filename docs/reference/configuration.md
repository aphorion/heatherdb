# Configuration reference

Three layers, in order of scope:

1. **Process flags / environment variables** — set at launch, apply server-wide.
2. **`db.toml`** — per database, persisted, re-read on every mount.
3. **Request fields** — per call.

## Process flags

Every flag has an environment variable. The flag wins when both are set.

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--data-dir` | `HEATHER_DATA_DIR` | *(required)* | Storage root |
| `--port` | `HEATHER_PORT` | `6380` | Listen port |
| `--host` | `HEATHER_HOST` | `0.0.0.0` | Bind address |
| `--dimension` | `HEATHER_DIMENSION` | `128` | Dimension for the auto-created `default` database, on the boot that creates it |
| `--l0` | `HEATHER_L0` | `0` | Initial hard-location count for `default`, on the boot that creates it. `0` is data-seeded |
| `--map-size-mb` | `HEATHER_MAP_SIZE_MB` | `4096` | LMDB map ceiling for `default`, on the boot that creates it |
| `--request-timeout` | `HEATHER_REQUEST_TIMEOUT` | `30` | Seconds. Raise for large algebra ops |
| `--max-body-size` | `HEATHER_MAX_BODY_SIZE` | `2097152` | Max request body, bytes |
| `--max-algebra-locations` | `HEATHER_MAX_ALGEBRA_LOCATIONS` | `250000` | Cap on an algebra result's location count. `0` disables |
| `--max-bulk-items` | `HEATHER_MAX_BULK_ITEMS` | `100000` | Cap on `bulk_load` items. `0` disables |
| `--max-batch-queries` | `HEATHER_MAX_BATCH_QUERIES` | `1000` | Cap on `batch_analyze` queries. `0` disables |
| `--admin-user` | `HEATHER_ADMIN_USER` | `admin` | First-boot admin username |
| `--admin-password` | `HEATHER_ADMIN_PASSWORD` | *(generated)* | First-boot admin password. Minimum 8 characters — a shorter one aborts the boot |
| `--auth-disabled` | `HEATHER_AUTH_DISABLED` | `false` | Skip auth entirely. **Dev only** |
| — | `RUST_LOG` | `info` | Tracing filter. `debug` for verbose |

Three of these — `--dimension`, `--l0`, `--map-size-mb` — apply **only to the
`default` database, and only on the boot that creates it**. Once
`db/default/db.toml` exists, the persisted values win and the flags are ignored
for it. They never size the databases you create; those carry their own values.
That is why the boot log reports `max_body_size` and `request_timeout_secs`
but deliberately does not report `map_size_mb`.

The three `--max-*` caps bound the *result* of a request rather than its wire
size: a 2 MB JSON body can describe an algebra op whose n×m cross product
allocates gigabytes.

## Storage layout

```
$HEATHER_DATA_DIR/
  server.toml                # layout tag; the engine refuses an unknown layout
  engine.lock                # exclusive advisory lock, held while an engine runs
  initial-admin-password     # mode 0600, only when the password was generated
  system/data/               # LMDB env: user store + session tokens
  db/<name>/db.toml          # per-database config
  db/<name>/data/            # LMDB env, one per database
  snapshots/                 # default landing zone for `snapshot create`
  backups/                   # conventional location listed by `backup list`
  _trash/<name>-<ts>/        # dropped databases; recoverable by moving back
```

Each database's LMDB environment holds six named sub-databases: `_registry`,
`_locations`, `_metadata`, `_documents`, `_doc_index`, `_audit`.

The engine takes `engine.lock` exclusively for its whole lifetime and refuses
to start if another process holds it. The cold `backup` and `restore`
subcommands take the same lock, which is how they refuse to run against a live
engine.

## `db.toml`

Written by `POST /db` (or on first boot for `default`), read on every mount.
Editing it and restarting is a supported way to change a database — with the
exception of `eam.d`, which must never change.

```toml
name = "movies"
created_at = 1788260246
map_size_mb = 32768

[eam]
d = 384
l_0 = 0
k = 20
eta_0 = 0.01
lambda = 0.9999
eta_min = 0.001
tau_split = 0.3
tau_merge = 0.95
gamma = 1.0
tau_damp = 10.0
tau_overload = 12.0
beta = 5.0
t_max = 10
epsilon = 1e-6
neighbor_cap = 96
num_landmarks = 32
mdl_gate = false
competitive = true

[dream]
enabled = false
collections = []
idle_secs = 60
passes = 1
mode = "replay"
tau_cohere = 0.0
tau_split = 0.0

[audit]
enabled = true
max_entries = 1000000
retention_days = 90
flush_threshold = 256
flush_interval_secs = 2
max_buffer = 16384
store_raw_query = false
visibility = "Root"
```

`db.toml` is written atomically (temp file plus rename), so a crash mid-write
cannot leave a config that fails to parse on the next boot.

### Database names

1–63 characters, first character `a-z`, rest `a-z0-9_-`. Must not start with
`_` (reserved for system directories). Mirrors PostgreSQL identifier
conventions, slightly tighter.

## EAM knobs

`[eam]` in `db.toml`, returned by `GET …/collections/{name}/config`. Six of
them (`k`, `t_max`, `beta`, `neighbor_cap`, `num_landmarks`, `l_0`) are
settable at creation through `POST /db`'s `eam` object; the rest need a file
edit and a restart.

| Field | Default | Meaning |
|---|---|---|
| `d` | *(required)* | Vector dimension. **The source of truth for the database's dimension.** Immutable |
| `l_0` | `0` | Initial hard-location count. `0` grows the codebook from the first writes rather than pre-seeding random vectors |
| `k` | `20` | Nearest locations activated per read and per write |
| `eta_0` | `0.01` | Initial learning rate |
| `lambda` | `0.9999` | Learning-rate decay per write |
| `eta_min` | `0.001` | Learning-rate floor |
| `tau_split` | `0.3` | Novelty split threshold (cosine). Below it, a write spawns a location |
| `tau_merge` | `0.95` | Merge threshold (cosine). Above it, two locations fold together |
| `gamma` | `1.0` | Conscience factor in winner selection — biases against locations that keep winning |
| `tau_damp` | `10.0` | Damping time constant for the learning rate |
| `tau_overload` | `max(d/32, 8)` | Write count at which a saturating location splits. Tracks the measured per-read capacity wall of ≈ d/32 |
| `beta` | `5.0` | Softmax inverse temperature for reads |
| `t_max` | `10` | Maximum Hopfield iterations |
| `epsilon` | `1e-6` | Convergence threshold |
| `neighbor_cap` | `max(d/4, 20)` | Non-zero enables the navigable graph. Effective capacity is computed dynamically as `max((k-1)·⌈ln L⌉, 2k)`. `0` disables the graph |
| `num_landmarks` | `32` | Landmark entry points for graph search |
| `mdl_gate` | `false` | Allocate by description length (`surprise·recurrence > engram bits`) instead of `tau_split` / `tau_overload` |
| `competitive` | `true` | `false` makes writes verbatim streaming appends — no activation, no merge, no address migration. Turns the collection into an exact growing key→value store |

Validation, enforced whenever a config is built or merged:

- `d > 0`, `k > 0`, `t_max > 0`
- `eta_0 > 0`, `eta_min > 0`, `eta_min ≤ eta_0`
- `lambda` in `(0, 1]`
- `tau_split ≥ 0`, `tau_merge ≤ 1`, and **`tau_split < tau_merge`**
- `gamma ≥ 0`, `tau_damp > 0`, `tau_overload > 0`, `beta > 0`, `epsilon > 0`
- `num_landmarks > 0` whenever `neighbor_cap > 0`
- every float must be finite

## `[dream]`

Idle-time consolidation. Off by default. See
[Dreaming](../explanation/dreaming.md).

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Opt-in switch for the background loop |
| `collections` | `[]` | Collections to dream over. Empty means all |
| `idle_secs` | `60` | Seconds of server-wide inactivity before a pass may run |
| `passes` | `1` | Replay: granularity passes per dream. Ladder: levels to climb |
| `mode` | `"replay"` | `"replay"` reorganises in place; `"ladder"` builds derived `__L1`, `__L2` collections |
| `tau_cohere` | `0.0` | Ladder only: minimum counter coherence to join a family |
| `tau_split` | `0.0` | Ladder only: minimum address similarity for a candidate family |

`POST /db/{db}/dream` ignores `enabled` and `idle_secs` and accepts per-request
overrides for the rest.

## `[audit]`

Access logging. **On** by default. See
[Read the audit log](../how-to/read-the-audit-log.md).

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turning it off silently stops recording who read what |
| `max_entries` | `1000000` | Hard ceiling on stored records; oldest pruned past it |
| `retention_days` | `90` | Age bound. `0` disables it (`max_entries` still applies) |
| `flush_threshold` | `256` | Buffered records that trigger an out-of-band flush |
| `flush_interval_secs` | `2` | Periodic flush cadence |
| `max_buffer` | `16384` | Hard memory cap; records past it are dropped and counted |
| `store_raw_query` | `false` | Store the raw query vector alongside its hash |
| `visibility` | `"Root"` | `"Root"` or `"DbUsers"` — who may read this database's log |

## Compose environment

`docker-compose.yml` reads every tunable as `${VAR:-default}`, so Coolify's UI
or a local `.env` can override any of them. `.env.example` is the template.

The compose file's own defaults differ from the binary's in one place:
`HEATHER_REQUEST_TIMEOUT` is 600 rather than 30, because containerised
deployments tend to run larger algebra operations.

## Minimum supported Rust version

**1.88**, declared by every crate and enforced by CI. Three constraints stack:
`resolver = "3"` needs 1.84, `edition = "2024"` needs 1.85, and the let-chains
in `heather_db::read` need 1.88.
