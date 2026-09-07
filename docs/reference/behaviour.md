# Behaviour and silent failures

Behaviour a caller must know before it bites. The **Loud?** column is the one
to read first: a loud behaviour returns a status code you can branch on, a
silent one returns `200` and a plausible-looking result.

`file:line` references are into this repository at `heather_server` 0.3.0.

## Silent — returns 200, does the wrong thing

| Behaviour | Loud? | Do this instead | Source |
|---|---|---|---|
| A `write` without `metadata` takes the `write_batch` branch: the pattern shapes the memory and counts in `stats.total_writes`, but no document row and no document-index posting exist, so it can never be returned by `documents/query`, fetched by id, or named as a contributor | silent | Send `metadata` for anything you intend to retrieve as a document. `write_with_metadata` is the only path that mints a document id | `routes.rs:301`, `collection.rs:1432` |
| `bulk_load` and every algebra write replace the location set wholesale via `load_snapshot_checked`, which clears and rewrites `_locations` but never touches `_documents` or `_doc_index` — existing postings are orphaned | silent | Treat a `bulk_load`/algebra target as a vector-only collection, or rewrite its documents afterwards. Never point one at a collection holding documents you still need | `collection.rs:2098-2109` |
| `bulk_load` reassigns location ids from the request's array position (`LocationId(i)`), and `next_id` resets to `max(id)+1` — ids from before the load do not survive | silent | Re-derive any id you cached from the new response; do not persist location ids across a `bulk_load` | `routes.rs:392`, `collection.rs:2081-2086` |
| A misspelled collection name on `write`, `read`, `attention`, `attention/mdl`, `attention/calibrate`, `analyze`, `batch_analyze`, `bulk_load`, `documents/query` or any algebra *target* calls `get_or_create_collection` — the typo creates an empty collection and the read returns a zero-ish reconstruction | silent | Create collections explicitly and check `GET …/collections` in CI. `stats`, `config`, `locations`, `compress`, `fingerprint`, the document routes and algebra *sources* do 404 on the same typo | `routes.rs:519` (create), `routes.rs:685-690` (404) |
| `POST …/dream?mode=` matches `"ladder"` case-insensitively and falls through to `Replay` for every other value — `?mode=ladderr` reorganises in place instead of building `__L1` | silent | Send exactly `ladder` or `replay`; assert on the response's reported mode rather than on the request | `dream.rs:525-530` |
| A manual `POST …/dream` forces `enabled = true` on the resolved config, so it runs on databases that never opted in and, in `replay` mode, mutates locations in place | silent (and destructive) | Snapshot before triggering a dream on a database whose `[dream]` block is off. There is no dry-run mode | `dream.rs:519-523` |
| A `cleanup` of `{"beta": b}` with `b < 30.0` is clamped to `30.0`. No warning — the engine has no log sink | silent, but **visible in the response** | Read `cleanup_beta` in the response, which reports the temperature actually used. β=10 recovers 60% of the available signal, β=30 recovers 81%, flat from 100 | `collection.rs:87`, `collection.rs:1836` |
| `role_pairs` do not steer recall. Ranking is plain full-bundle cosine against `query`; the pairs only score documents that already survived. A document that would rank well under your weighting can be absent from the candidate set entirely | silent | **Over-fetch.** Ask for an `n` far larger than you intend to display and re-rank in the consumer. No value of `n` makes this exact | `collection.rs:1746-1752` |
| `DELETE …/documents/{id}` is a retrieval and citation tombstone. The write's contribution to the merged location addresses and counters stays; superposition is not invertible | silent | For material with an erasure obligation, use a separate collection or database so the unit of erasure is one the substrate can drop | `collection.rs:1573-1586` |
| A cleanup read repairs a degraded query but does **not** verify it. Reading a deliberately corrupted vector back against a stored law recovers it to 1.000 — and reading *pure noise* against the same collection also returns 1.000, because a read always settles somewhere | silent | Never use "the read returned a high score" as evidence that the input was meaningful. Verification is a separate step: compare against a [null](../terms/null-model.md), and score on data the memory has never seen | measured against a live engine |
| The competitive `write` path merges near-identical patterns, which destroys a per-item familiarity signal. Writing four distinct-but-similar items produced **2** locations, and a stored item then scored 0.49 against an unstored one at 0.48 — indistinguishable. `bulk_load` of the same four gives 4 locations and 1.00 against 0.50 | silent | When the question is "have I seen *this specific thing*", load the items with `bulk_load` so each is its own attractor. Use `write` when you want generalisation, which is the opposite need | measured against a live engine |
| An `iterative` read drifts when a collection's stored values differ from its addresses — the two-field case. Reading a known key returned cosine 0.27 against the intended value, where `"strategy": "fast"` on the same collection returned 0.999 | silent | Use `"strategy": "fast"` for a key→value collection, and reserve `iterative` for a codebook where address and value are the same vector | measured against a live engine |
| An overload split spawns the new location by perturbing the winner's address with a fresh random unit vector at 0.1 — location counts and ids are not reproducible run to run for the same input sequence | silent | Do not assert on `num_locations` in tests. Assert on recall, fidelity or description length | `write.rs:342` |
| Neighbour pruning drops the *oldest* edges once a location exceeds its adaptive cap, so graph quality is a function of write order | silent | Interleave rather than block-load related patterns, or set `neighbor_cap = 0` and take the exhaustive scan when order is not controllable | `write.rs:281-284` |
| Turning off `[audit].enabled` silently stops recording who read what; nothing in a read response changes | silent | Check `GET /db` / `db.toml` as part of a deployment audit | `configuration.md` → `[audit]` |
| A hard kill (SIGKILL, power loss) loses whatever audit records are still buffered — at most `flush_threshold` records or `flush_interval_secs` of activity. A graceful shutdown flushes | silent | Lower `flush_threshold` / `flush_interval_secs` if the log is compliance-bearing; accept the write contention that buys | `heather_db/src/audit.rs:59-64` |
| `--map-size-mb` / `HEATHER_MAP_SIZE_MB` sizes only the auto-created `default` database, and only on the boot that creates it. It never sizes databases created through `POST /db`, and it is ignored for `default` once `db.toml` exists | silent | Set `map_size_mb` per database on `POST /db`, and verify with `GET /db` | `main.rs:68-78` |
| EAM knobs come from the *database* config at collection mount, so every collection in a database shares them. A per-collection knob does not exist | silent | Split into databases when two workloads need different `beta`, `k` or `tau_overload` | `hive.rs:99` |

## Loud — returns an error you can branch on

| Behaviour | Loud? | What you get | Source |
|---|---|---|---|
| Loading a collection whose stored addresses do not match the database's `eam.d` | **loud** | `400`, `DimensionMismatch { expected, got }`. Dimension is the one EAM knob that fails loudly; every other mismatch is silent | `collection.rs:410-416` |
| `/vec/*` for a database-scoped user | **loud** | `403`. The default-scoped whitelist covers `/collections`, `/algebra` and `/compose` only; `/vec` is not on it, and is root-only for every scope | `users.rs:304-308` |
| A concurrent write landing between an algebra op's snapshot and its load | **loud** | `409 Conflict`, retryable | `collection.rs:2092-2095` |
| An algebra result, `bulk_load` or `batch_analyze` over its server cap | **loud** | `400` naming the cap and the flag that raises it | `limits.rs` |

## Neither — true, and easy to get wrong anyway

| Behaviour | Notes | Source |
|---|---|---|
| A session token's scope is frozen at mint. Changing the user's scope in the user store leaves already-issued tokens on the old scope until they expire or are revoked | Revoke explicitly after any scope change | `tokens.rs:18-20` |
| `analyze` reports cosines in `similarity`; `attention` reports raw dot products `Q·Kᵢ` in the same field name | Do not compare the two, and do not threshold one with a constant tuned on the other | `models.rs:109-117` |
| A read returns a *reconstruction*, which may be a vector nobody ever wrote | This is not a nearest-neighbour index. Fetch documents by id if you need the exact stored bytes | `explanation/associative-memory.md` |

## Related

- [Configuration](configuration.md) — where each knob lives.
- [Parameters](parameters.md) — what each knob does and what a wrong value looks like.
- [Capacity](capacity.md) — the walls these behaviours show up at.
- [HTTP API overview](../api/overview.md) — status codes and the create-on-write asymmetry.
