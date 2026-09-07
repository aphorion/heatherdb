# Tune a database

Most deployments never touch any of this. Reach for it when you have a
concrete symptom.

Every knob below lives in one place: `$HEATHER_DATA_DIR/db/<name>/db.toml`,
written when the database is created and re-read on every mount. Set it at
creation through `POST /db`, or edit the file and restart.

## Dimension

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' -d '{"name":"movies","dimension":384}'
```

Dimension is per database, immutable after creation. The engine refuses to load
a collection at a dimension that does not match `db.toml`. To change it, create
a new database and re-write the data — there is no in-place conversion, and
there never will be, because the stored codebook is dimensioned.

## Map size

`map_size_mb` is the LMDB address-space ceiling for one database's environment.
It is a ceiling, not an allocation: the map is sparse, so a generous value
costs address space, not disk.

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"movies","dimension":384,"map_size_mb":32768}'
```

**`--map-size-mb` / `HEATHER_MAP_SIZE_MB` does not size the databases you
create.** It applies to the auto-created `default` database, and only on the
boot that creates it. Every other database carries its own value. Check what
each one actually has:

```bash
curl -u admin:pw http://localhost:6380/db     # map_size_mb, per database
```

Getting it wrong is recoverable. The ceiling belongs to the LMDB environment
handle rather than the file, and `db.toml` is re-read on every mount, so
raising it is an edit and a restart:

```bash
vi $HEATHER_DATA_DIR/db/movies/db.toml     # map_size_mb = 32768
systemctl restart heatherdb
```

The restart is required — LMDB refuses to reopen an environment with different
options in the same process. **Grow only.** Never set a ceiling below the data
already stored.

## EAM knobs

Set any subset at creation. Unset fields keep the engine default, and the
merged config is validated before the database exists.

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"encoded","dimension":1024,
       "eam":{"k":3840,"t_max":1,"neighbor_cap":0}}'
```

| Knob | Default | Raise it when | Lower it when |
|---|---|---|---|
| `k` | 20 | reads look under-informed; set it above the location count for a full softmax | reads are slow and you have many locations |
| `t_max` | 10 | reads report `converged: false` | you want single-step reads (`1` for a raw one-shot read) |
| `beta` | 5.0 | reads blend too many patterns into mush | one location dominates and drowns real alternatives |
| `neighbor_cap` | `max(d/4, 20)` | — | `0` disables the navigable graph entirely (brute-force activation) |
| `num_landmarks` | 32 | graph search enters at a bad starting point | — |
| `l_0` | 0 | you specifically want a pre-seeded random codebook | leave at 0 — data-seeded is the default |

Only these six are settable over HTTP. The rest (`eta_0`, `lambda`, `eta_min`,
`tau_split`, `tau_merge`, `gamma`, `tau_damp`, `tau_overload`, `epsilon`,
`competitive`) are editable in `db.toml`, followed by a restart. See
[Configuration](../reference/configuration.md#eam-knobs) for what each one
means and the validation rules between them.

Read back what a collection is actually running:

```bash
curl -u admin:pw http://localhost:6380/db/movies/collections/taste/config
```

## The MDL allocation gate

```bash
curl -u admin:pw -X POST http://localhost:6380/db \
  -H 'Content-Type: application/json' \
  -d '{"name":"movies","dimension":384,"mdl_gate":true}'
```

With `mdl_gate` on, the decision to spawn a new hard location stops being two
fixed thresholds (`tau_split` for novelty, `tau_overload` for saturation) and
becomes one rule: allocate a location when accumulated surprise, in bits,
exceeds what an engram costs — `log2(L+1) + log2(d/32)`.

The address term rises with the location count, so the bar for a new location
self-anneals as the index fills. Worth trying when you cannot find a `tau_split`
that behaves across the whole range of your data.

## Symptom-driven changes

**`num_locations` grows faster than you expected.** That is a readout of data
complexity, not a bug. If it is genuinely runaway, either your data is more
diverse than assumed, or `tau_split` is too high for it. Run
`POST …/collections/{name}/compress` to fold locations back down to the MDL
minimum, or enable [dreaming](../explanation/dreaming.md) to have the engine do
it while idle.

**Reads return `converged: false` from `analyze`.** Raise `t_max`, or accept it
— a non-converged iterative read still returns the state it reached. If the
collection holds many mutually dissimilar patterns and `k` covers most of them,
the softmax may simply be too soft; raise `beta`.

**Reads blend everything together.** Too many locations are being activated at
too soft a temperature. Use `attention` with an explicit high `beta`, or
`attention/mdl` to let the engine pick per query, or lower `k`.

**You do not know what `beta` should be.** Ask:

```bash
curl -u admin:pw -X POST http://localhost:6380/db/movies/collections/taste/attention/calibrate \
  -H 'Content-Type: application/json' -d '{}'
```
```json
{"beta": 101.9, "dl": 127.7}
```

That sweeps 81 log-spaced temperatures across `[0.5, 200]` and picks the one
minimising leave-one-out value-reconstruction description length.

**Large algebra ops time out.** Raise `--request-timeout` (default 30 seconds).

**An algebra op is refused for size.** The cross-product cap
(`--max-algebra-locations`, default 250 000) fired. Pass a smaller
`max_cross_k` in the request, or raise the cap.

## Server-wide limits

These are process flags, not per-database, and bound the *result* of a request
rather than its wire size — a 2 MB body can describe an algebra op that
allocates gigabytes. `0` disables any of them.

| Flag | Default |
|---|---|
| `--max-algebra-locations` | 250 000 |
| `--max-bulk-items` | 100 000 |
| `--max-batch-queries` | 1 000 |
| `--max-body-size` | 2 097 152 bytes |
| `--request-timeout` | 30 seconds |
