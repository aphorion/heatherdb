# Parameter reference

Every knob, what it controls, and what a wrong value looks like from outside.

[Configuration](configuration.md) says where each setting lives and what its
default is. This page says how to choose one.

Source of truth: `heather_db/src/config.rs`, `db_config.rs`, `collection.rs`,
`heather_server/src/main.rs`, `limits.rs`, `models.rs`.

## Mutability

Three classes, and the first two are the ones that hurt.

| Class | Knobs | Rule |
|---|---|---|
| **Immutable after creation** | `dimension` (`eam.d`) | The stored codebook is dimensioned. Create a new database and re-write the data. A mismatch is loud (`400 DimensionMismatch`) |
| **Restart required** | `map_size_mb`, every `[eam]` field, `[dream]`, `[audit]` | Edit `db.toml` and restart. LMDB refuses to reopen an environment with different options in the same process |
| **Per request** | `strategy`, `scale`, `exclude_id`, `alpha`, `threshold`, `power`, `max_cross_k`, `routing_sharpness`, `n`, `cleanup`, `kappa`, `lambda` (compress) | Free to vary call to call |

**No EAM knob can differ per collection.** A collection is mounted with the
database's config (`hive.rs:99`), so two workloads that need different `beta`,
`k` or `tau_overload` need two databases.

Six `[eam]` fields — `k`, `t_max`, `beta`, `neighbor_cap`, `num_landmarks`,
`l_0` — are settable at creation through `POST /db`'s `eam` object. The rest
need a file edit.

## Dimension and storage

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `dimension` / `eam.d` | `POST /db`, `--dimension` for `default` on first boot only | Vector width for the whole database. **Immutable** | Too low: reads blend, schema slots collapse, capacity walls arrive early. Too high: proportional memory and CPU per read, nothing else | Choose for the interference you want, **not** for the encoder's native width. An encoder wider than the database is projected down (see [Encodings](encodings.md)); one narrower is zero-padded. Start from [Capacity](capacity.md): `D/32` items per pool, `√(D/32)` schema slots. 512–1024 covers most workloads; 4096+ only for deep schemas |
| `map_size_mb` | `POST /db`, `db.toml`; `--map-size-mb` for `default` on first boot only | LMDB address-space ceiling for one database's environment | `MDB_MAP_FULL` on write, at the ceiling. A generous value costs address space, not disk — the map is sparse | Set well above the projected dataset. **Grow only** — never below what is already stored. Raising it is an edit and a restart |

## Read shape

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `beta` | `db.toml` `[eam]`, `POST /db` | Softmax inverse temperature for reads. Low β averages many locations; high β concentrates on the winner | Too low: reads blend several stored patterns into a plausible vector matching none. Too high: a read is an exact nearest-neighbour lookup and the memory stops generalising | **Must scale with pool size.** At D=1024 the default β=5 holds ≥90% recall only to N≈16; β=30 reaches the exact-NN ceiling. Use [`attention/calibrate`](../api/reads.md) to pick one per collection, or `attention/mdl` per query |
| `cleanup_beta` | request `cleanup: {"beta": b}` | Temperature for denoising a recovered filler against the codebook, in multi-role scoring | Below `30.0` it is **clamped**, silently, and the value actually used comes back in the response's `cleanup_beta`. A soft β yields a plausible-looking but wrong ordering | β=10 recovers 60% of the available signal, β=30 recovers 81%, flat from 100 up. Send `"mdl"` (the default) unless you have measured otherwise |
| `k` | `db.toml` `[eam]`, `POST /db` | Nearest locations activated per read and per write | Too high: more locations enter the mix, reads blend. Too low: the read misses the right location and reconstructs from neighbours | Default 20. Lower it as a capacity lever before touching `beta`. `k` ≥ the location count is a full softmax |
| `t_max` | `db.toml` `[eam]` | Maximum Hopfield iterations per read | Too low: `analyze` reports `converged: false` and the result is mid-descent. Too high: latency, with no accuracy change past convergence | Default 10. Check `analyze`'s `iterations` and `converged` on representative queries; raise only if `converged` is false |
| `epsilon` | `db.toml` `[eam]` | Convergence threshold for the Hopfield loop | Too large: the loop stops early on a partly-settled vector. Too small: every read runs to `t_max` | Default `1e-6`. Leave it; reach for `t_max` instead |
| `strategy` | request field on `read`, `analyze`, `batch_analyze` | `"iterative"` (default) settles the Hopfield loop; `"fast"` takes a single step | `"fast"` on a heavily corrupted query returns a partial reconstruction. Any other string is a `400` | `"iterative"` for reconstruction, `"fast"` for a similarity probe where the exact vector does not matter |
| `scale` | request field on `attention` | Inverse temperature for the raw dot-product attention read. Analogous to β, but on `Q·Kᵢ` rather than a cosine | Too low: uniform blend. Too high: single-key lookup | Measurement-driven. Depends on the vectors' norms, which the engine does not normalise on this path. Use `attention/mdl` to have the engine pick, or `attention/calibrate` to fix one |
| `exclude_id` | request field on `attention` | Drops one stored location from the activated set (leave-one-out) | An id past the location count is a `400` | Only for leave-one-out evaluation; leave unset in production |
| `routing_sharpness` | request field on `compose/read` | Inverse temperature for the softmax over per-collection match scores | Too low: the read averages across every shard and the answer is mush. Too high: the read commits to one shard and a query that straddles two loses half its evidence | Default 20.0. Raise when shards are well separated, lower when they overlap by design |

## Write shape

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `eta_0` | `db.toml` `[eam]` | Initial learning rate — how far a location's address migrates toward each write | Too high: addresses chase the most recent write and earlier patterns drift out. Too low: the codebook barely adapts and everything looks novel | Default 0.01. Measurement-driven; raise only for a stream whose distribution genuinely moves |
| `eta_min` | `db.toml` `[eam]` | Floor the decayed learning rate cannot fall below | `0` is rejected. Too high defeats the decay | Default 0.001. Must satisfy `0 < eta_min ≤ eta_0` |
| `lambda` | `db.toml` `[eam]` | Learning-rate decay per write. `eta ← max(eta·λ, eta_min)` | Too low: the rate hits `eta_min` within a few hundred writes and the memory freezes. `1.0` disables decay | Default 0.9999 — the rate halves at roughly 7000 writes. Must be in `(0, 1]` |
| `tau_damp` | `db.toml` `[eam]` | Damping time constant applied to the learning rate | Interacts with `lambda`; the observable is the same as a mis-set `lambda` | Default 10.0, `> 0`. Measurement-driven — change `lambda` first |
| `gamma` | `db.toml` `[eam]` | Conscience factor in winner selection: biases against locations that keep winning | `0` disables the conscience and a hot location absorbs everything. Too high spreads writes across locations that do not fit them, blurring the codebook | Default 1.0, `≥ 0`. Raise on a skewed key distribution where `stats.max_write_count` far exceeds `avg_write_count` |
| `tau_split` | `db.toml` `[eam]` | Novelty threshold (cosine). A write below it against its winner spawns a new location | Too high: every write spawns a location and the collection becomes a list. Too low: dissimilar patterns merge into one blurred address | Default 0.3. **Must be `< tau_merge`** or the config is rejected |
| `tau_merge` | `db.toml` `[eam]` | Merge threshold (cosine). Two locations above it fold together | Too low: distinct clusters merge and recall drops. Too high: near-duplicate locations accumulate | Default 0.95, `≤ 1` |
| `tau_overload` | `db.toml` `[eam]` | Write count at which a saturating location splits | Too high: a location holds more patterns than `D/32` and reads from it blend. Too low: constant splitting, an inflated location count and a large index | Default `max(d/32, 8)`, which tracks the measured per-read capacity wall. Override only when running a non-default `beta` — a sharper β supports a larger pool |
| `l_0` | `db.toml` `[eam]`, `POST /db`, `--l0` for `default` | Initial hard-location count | A positive value pre-seeds *random* unit vectors, so early reads match noise | Leave at `0` (data-seeded). Positive only when a random initial codebook is explicitly wanted |
| `competitive` | `db.toml` `[eam]` | `true` is adaptive EAM. `false` makes writes verbatim streaming appends — no activation, no merge, no address migration, no normalisation | With `false`, `tau_split`/`tau_merge`/`tau_overload`/`eta_*`/`gamma` do nothing and the location count equals the write count | `false` turns the collection into an exact growing key→value store — the mode a raw dot read (`attention`) needs to reproduce trained attention |
| `mdl_gate` | `db.toml` `[eam]` | Replaces `tau_split` and `tau_overload` with one rule: allocate a location iff accumulated surprise clears `engram_bits = log2(L+1) + log2(d/32)` | With it on, a high-traffic but coherent location carries ~0 debt and does **not** split even past `tau_overload` — `max_write_count` climbs while `num_locations` stays flat | `false` by default. Turn it on when you want the location count to self-calibrate to the data's structure rather than to two hand-set thresholds. The spawn bar self-anneals as `L` grows |

## Graph

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `neighbor_cap` | `db.toml` `[eam]`, `POST /db` | Non-zero enables the navigable graph. Effective capacity is computed dynamically as `max((k-1)·⌈ln L⌉, 2k)` | `0` disables the graph and every read is an exhaustive scan — correct, but linear in `L`. Non-zero makes graph quality depend on write order, since pruning drops the oldest edges | Default `max(d/4, 20)`. Set `0` when `L` is small, or when write order is uncontrollable and exact recall matters more than latency |
| `num_landmarks` | `db.toml` `[eam]`, `POST /db` | Landmark entry points for graph search | Too few: the search enters in the wrong region and misses the true winner. Too many: every read pays a wider entry scan | Default 32. Must be `> 0` whenever `neighbor_cap > 0` |

## Algebra and compress

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `alpha` | request field on `algebra/scale` | Multiplier applied to every location's counter | — | Operands have **mass**. Balance by the mass ratio before subtracting: `0.02` is a no-op, `0.5` subtracts cleanly, `2.0` overshoots into noise. Scale the subtrahend to the minuend's counter magnitude first, then `algebra/sub` |
| `threshold` | request field on `algebra/intersect` | Cosine floor for a cross-pair to count as shared | Too low: the intersection is the cross product and the location cap fires. Too high: an empty target | Default 0.95. A non-positive value bypasses the fast path and the result is bounded by `--max-algebra-locations` |
| `power` | request field on `algebra/permute` | Exponent `k` in `ρ^k`. Negative applies `ρ⁻¹` | A `power` that does not undo the one used to encode leaves the vector permuted — no error, just no match | Default 1. Use `-k` to invert. Identify the permutation with exactly one of `seed` or `name`; neither or both is a `400` |
| `max_cross_k` | request field on `algebra/add`, `sub`, `bind` | Caps how many of `source_b`'s locations each of `source_a`'s pairs with | Unset, an n×m cross product from a small JSON body can allocate gigabytes, and `--max-algebra-locations` refuses it with a `400` | Set it whenever both sources are large. The right value is the number of `b`-locations that are plausibly relevant to one `a`-location |
| `kappa` | request field on `compress` | Bits charged to specify one location (the model term in `L·kappa + W·log2 L`) | Too low: compression barely merges. Too high: it merges distinct clusters | Defaults to the collection's `d`. Raise to compress harder |
| `lambda` (compress) | request field on `compress` | Weight on a merge's data-fit cost (the variance increase) | Too low: aggressive merging, recall drops. Too high: no merges pay and the location count is unchanged | Default 1.0. Compare `description_length_before`/`_after` and `locations_after` across values |

## Documents

| Knob | Set at | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `n` | request field on `documents/query` | Number of results returned | With `role_pairs`, a small `n` silently drops documents that would rank well under your weighting — ranking ignores the pairs | Default 10. **Over-fetch** whenever `role_pairs` or client-side weighting is in play, and re-rank in the consumer |
| `unbind_role` | request field on `documents/query` | Reads `query` as the filler expected at this role instead of comparing the stored superposition whole | Mutually exclusive with `role_pairs` | Use when the question is "which document has X at this role", not "which document is like X" |
| `cleanup` | request field on `documents/query` | `"mdl"` (default), `"off"`, or `{"beta": b}` | `"off"` scores the raw recovered filler, crosstalk and all — a genuine accuracy-for-speed trade, honoured exactly. `{"beta": b}` below 30 is clamped | `"mdl"` unless profiling says the cleanup transform is the bottleneck |

## Server limits

Process-wide, set by flag or environment variable, `0` disables. They bound the
*result* of a request rather than its wire size: a 2 MB body can describe a
computation that allocates gigabytes.

| Knob | Default | Controls | Wrong value looks like | Choosing |
|---|---|---|---|---|
| `--request-timeout` | 30 s | Per-request deadline | `408 Request Timeout` on a large algebra op or a wide `batch_analyze` | Raise for algebra-heavy workloads. The compose file already defaults to 600 for that reason |
| `--max-body-size` | 2 MiB | Maximum request body | `413 Payload Too Large`. A 1024-dim `bulk_load` of 100 items is already ~2 MB of JSON; `/vec/bundle` hits it at 128 terms of 1024 dimensions | Raise for wide-dimension bulk work, or split the batch |
| `--max-bulk-items` | 100 000 | Items per `bulk_load` | `400` naming the cap and the flag | Bound by memory: one location is `2·d` f64s in flight |
| `--max-batch-queries` | 1 000 | Queries per `batch_analyze` | `400` naming the cap | Bound by `--request-timeout` × per-query latency |
| `--max-algebra-locations` | 250 000 | Location count of an algebra result. Applies to `add`, `sub`, `bind`, and `intersect` with a non-positive `threshold` | `400` before the allocation, not an OOM after it | Prefer setting `max_cross_k` on the request to raising this |

## Validation

Rejected at every config build or merge, so a bad `db.toml` fails at mount
rather than at first read:

- `d > 0`, `k > 0`, `t_max > 0`
- `eta_0 > 0`, `eta_min > 0`, `eta_min ≤ eta_0`
- `lambda` in `(0, 1]`
- `tau_split ≥ 0`, `tau_merge ≤ 1`, and **`tau_split < tau_merge`**
- `gamma ≥ 0`, `tau_damp > 0`, `tau_overload > 0`, `beta > 0`, `epsilon > 0`
- `num_landmarks > 0` whenever `neighbor_cap > 0`
- every float finite

## Related

- [Configuration](configuration.md) — where each knob lives and its default.
- [Capacity](capacity.md) — the measured walls these knobs move.
- [Behaviour](behaviour.md) — the knobs whose wrong value is silent.
- [Tune a database](../how-to/tune-a-database.md) — symptom-first procedure.
