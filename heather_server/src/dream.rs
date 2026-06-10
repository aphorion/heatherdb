//! Idle-time dreaming: the engine reprocesses its own memory while no one is
//! querying. A dream pass operates on a collection **in place** — no external
//! input, no second collection.
//!
//! - **Pass 0 (replay + merge):** every stored attractor is re-presented to
//!   the EAM's own write rule. Re-competing the data sharpens attractors and
//!   lets novelty/overload splits reorganize; `merge` then dedups nearby
//!   locations and enforces capacity. This is hippocampal replay.
//! - **Pass k≥1 (coarser novelty organization):** the operands become
//!   clusters of locations rather than single locations. Cluster centroids are
//!   replayed, growing coarse super-attractors alongside the fine ones — the
//!   substrate stays flat, organization emerges from passes at increasing
//!   operand granularity. Experimental; enabled only when `passes > 1`.
//!
//! The loop is driven by `run_dream_loop`, a background task that fires only
//! after the server has been idle for `idle_secs` and only for collections
//! whose location count changed since their last dream.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Extension, Path};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use heather_algebra::unbind_vec;
use heather_db::{Collection, DreamConfig, DreamMode, Hive, Server, WriteOpts};

/// Per-collection outcome of one dream.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionDream {
    pub collection: String,
    pub passes: usize,
    pub locations_before: usize,
    pub locations_after: usize,
    pub merged: usize,
}

/// Outcome of dreaming a whole database.
#[derive(Debug, Clone, Serialize)]
pub struct DreamReport {
    pub database: String,
    pub collections: Vec<CollectionDream>,
}

/// Current unix time in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Greedy cosine clustering of location addresses. `threshold` is the minimum
/// cosine to join an existing cluster; lower = coarser. Returns one normalized
/// centroid per cluster.
fn cluster_centroids(addrs: &[Vec<f64>], threshold: f64) -> Vec<Vec<f64>> {
    let mut centroids: Vec<Vec<f64>> = Vec::new(); // running sums
    let mut counts: Vec<f64> = Vec::new();
    for a in addrs {
        let mut best = (-1.0f64, usize::MAX);
        for (i, c) in centroids.iter().enumerate() {
            let mean = heather_db_normalize(c);
            let cos = dot(&mean, a);
            if cos > best.0 {
                best = (cos, i);
            }
        }
        if best.0 >= threshold {
            let i = best.1;
            for (x, y) in centroids[i].iter_mut().zip(a) {
                *x += y;
            }
            counts[i] += 1.0;
        } else {
            centroids.push(a.clone());
            counts.push(1.0);
        }
    }
    centroids.iter().map(|c| heather_db_normalize(c)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalize to unit length (duplicated here to avoid leaning on an engine
/// internal; identical semantics to `heather_db::vec_ops::normalize`).
fn heather_db_normalize(v: &[f64]) -> Vec<f64> {
    let n = dot(v, v).sqrt();
    if n < 1e-12 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Dream a single collection in place. Returns the before/after location counts
/// and how many locations were merged across all passes.
pub fn dream_collection(
    col: &Collection,
    passes: usize,
) -> heather_db::error::Result<CollectionDream> {
    let name = col.name().to_string();
    let before = col.num_locations()?;
    let mut merged_total = 0usize;
    let passes = passes.max(1);

    for g in 0..passes {
        let (locs, _cfg) = col.snapshot()?;
        if locs.is_empty() {
            break;
        }
        let addrs: Vec<Vec<f64>> = locs.iter().map(|l| l.address.clone()).collect();

        // Operands at this granularity. g==0: the attractors themselves
        // (replay). g>=1: cluster centroids, coarser as g grows.
        let operands: Vec<Vec<f64>> = if g == 0 {
            addrs
        } else {
            // Threshold loosens with depth: 0.7, 0.5, 0.3, ... (floored).
            let threshold = (0.7 - 0.2 * (g as f64 - 1.0)).max(0.1);
            cluster_centroids(&addrs, threshold)
        };

        for op in &operands {
            col.write(op)?;
        }
        merged_total += col.merge()?;
    }

    col.flush()?;
    Ok(CollectionDream {
        collection: name,
        passes,
        locations_before: before,
        locations_after: col.num_locations()?,
        merged: merged_total,
    })
}

/// Suffix marking a ladder level collection (`<base>__L1`, `__L2`, …). Skipped
/// when selecting base collections so the ladder doesn't climb its own output.
const LEVEL_SUFFIX: &str = "__L";

/// One consolidation rung: read each `(address, counter)` location of `src`,
/// strip the context with `unbind(counter, address)` to get the law it carries,
/// and route that law by its context into `dst` via the gated two-field write.
/// `dst` is cleared first. Out come `(context prototype, accumulated law)`
/// locations — the families/mnemonics of `src`.
fn consolidate_rung(src: &Collection, dst: &Collection, opts: WriteOpts) -> heather_db::error::Result<()> {
    let (locs, _) = src.snapshot()?;
    let cfg = dst.config()?;
    dst.load_snapshot(Vec::new(), cfg)?; // clear
    for l in &locs {
        let law = unbind_vec(&l.counter, &l.address);
        dst.write_two(&l.address, &law, opts)?;
    }
    Ok(())
}

/// Climb the consolidation ladder from a base episodic collection: base →
/// `base__L1` → `base__L2` → … for up to `levels` rungs, stopping when a level
/// has fewer than two locations (nothing left to abstract). Each rung is the
/// same operation — `(address, counter)` is closed under it — so the ladder is
/// just `consolidate_rung` applied to its own output.
fn dream_ladder(
    hive: &Hive,
    base: &str,
    levels: usize,
    opts: WriteOpts,
) -> heather_db::error::Result<Vec<CollectionDream>> {
    let mut reports = Vec::new();
    let mut src_name = base.to_string();
    for k in 0..levels.max(1) {
        let src = match hive.get_collection(&src_name)? {
            Some(c) => c,
            None => break,
        };
        let before = src.num_locations()?;
        if before < 2 {
            break; // can't form a higher-order family from <2 locations
        }
        let dst_name = format!("{base}{LEVEL_SUFFIX}{}", k + 1);
        let dst = hive.get_or_create_collection(&dst_name)?;
        consolidate_rung(&src, &dst, opts)?;
        reports.push(CollectionDream {
            collection: dst_name.clone(),
            passes: k + 1,
            locations_before: before,
            locations_after: dst.num_locations()?,
            merged: 0,
        });
        src_name = dst_name;
    }
    Ok(reports)
}

/// Dream every collection a `DreamConfig` selects in one hive. An empty
/// `collections` list means "every collection in the database". In `Replay`
/// mode each collection is reorganized in place; in `Ladder` mode each base
/// collection is consolidated into a derived stack (generated `__L` levels are
/// skipped as bases).
pub fn dream_database(
    db: &str,
    hive: &Hive,
    cfg: &DreamConfig,
) -> heather_db::error::Result<DreamReport> {
    let names = if cfg.collections.is_empty() {
        hive.list_collections()?
    } else {
        cfg.collections.clone()
    };
    let opts = WriteOpts {
        gate: true,
        tau_cohere: cfg.tau_cohere,
    };
    let mut collections = Vec::new();
    for name in names {
        match cfg.mode {
            DreamMode::Ladder => {
                if name.contains(LEVEL_SUFFIX) {
                    continue; // don't re-ladder a generated level
                }
                collections.extend(dream_ladder(hive, &name, cfg.passes, opts)?);
            }
            DreamMode::Replay => {
                if let Some(col) = hive.get_collection(&name)? {
                    collections.push(dream_collection(&col, cfg.passes)?);
                }
            }
        }
    }
    Ok(DreamReport {
        database: db.to_string(),
        collections,
    })
}

/// Background task: dream idle databases. Fires every `tick`; for each database
/// with `dream.enabled`, runs a pass only if the server has been idle for
/// `idle_secs` AND some selected collection changed size since its last dream
/// (so a quiescent database is dreamed once, not every tick).
pub async fn run_dream_loop(server: Arc<Server>, last_activity_ms: Arc<AtomicU64>) {
    let tick = Duration::from_secs(15);
    // (db, collection) -> location count at last dream.
    let mut last_dreamed: HashMap<(String, String), usize> = HashMap::new();

    loop {
        tokio::time::sleep(tick).await;

        let idle_ms = now_ms().saturating_sub(last_activity_ms.load(Ordering::Relaxed));
        let databases = match server.databases() {
            Ok(d) => d,
            Err(_) => continue,
        };

        for db in databases {
            let cfg = match server.database_config(&db) {
                Ok(c) => c.dream,
                Err(_) => continue,
            };
            if !cfg.enabled || idle_ms < cfg.idle_secs.saturating_mul(1000) {
                continue;
            }
            let hive = match server.database(&db) {
                Some(h) => h,
                None => continue,
            };
            let names = if cfg.collections.is_empty() {
                hive.list_collections().unwrap_or_default()
            } else {
                cfg.collections.clone()
            };

            let opts = WriteOpts {
                gate: true,
                tau_cohere: cfg.tau_cohere,
            };
            for name in names {
                if cfg.mode == DreamMode::Ladder && name.contains(LEVEL_SUFFIX) {
                    continue; // don't re-ladder a generated level
                }
                let col = match hive.get_collection(&name) {
                    Ok(Some(c)) => c,
                    _ => continue,
                };
                let count = col.num_locations().unwrap_or(0);
                let key = (db.clone(), name.clone());
                if last_dreamed.get(&key) == Some(&count) {
                    continue; // unchanged since last dream — nothing new to consolidate
                }
                let result = match cfg.mode {
                    DreamMode::Replay => dream_collection(&col, cfg.passes).map(|r| {
                        tracing::info!(
                            db = %db, collection = %name,
                            before = r.locations_before, after = r.locations_after,
                            merged = r.merged, passes = r.passes, "dreamed collection"
                        );
                    }),
                    DreamMode::Ladder => dream_ladder(&hive, &name, cfg.passes, opts).map(|rungs| {
                        tracing::info!(
                            db = %db, base = %name, levels = rungs.len(),
                            "climbed consolidation ladder"
                        );
                    }),
                };
                match result {
                    Ok(()) => {
                        // Settled count: replay mutates in place; ladder leaves the
                        // base unchanged. Either way, store the post-dream count so a
                        // quiescent collection is dreamed once, not every tick.
                        let settled = col.num_locations().unwrap_or(count);
                        last_dreamed.insert(key, settled);
                    }
                    Err(e) => {
                        tracing::warn!(db = %db, collection = %name, error = %e, "dream failed")
                    }
                }
            }
        }
    }
}

/// Middleware: stamp the last-activity clock on every request, so the dream
/// loop can tell when the server has gone quiet.
pub async fn stamp_activity(
    axum::extract::State(last): axum::extract::State<Arc<AtomicU64>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    last.store(now_ms(), Ordering::Relaxed);
    next.run(req).await
}

/// `POST /db/{db}/dream` — trigger a consolidation pass now (bypasses the idle
/// gate). Query params override the persisted config for this run:
/// `?mode=ladder|replay&levels=N&tau_cohere=F&collections=a,b`.
#[derive(Debug, Default, serde::Deserialize)]
pub struct DreamQuery {
    pub mode: Option<String>,
    pub levels: Option<usize>,
    pub tau_cohere: Option<f64>,
    pub collections: Option<String>,
}

/// gate). Useful for operators and tests.
pub async fn trigger(
    Extension(server): Extension<Arc<Server>>,
    Path(db): Path<String>,
    axum::extract::Query(q): axum::extract::Query<DreamQuery>,
) -> Response {
    let hive = match server.database(&db) {
        Some(h) => h,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("database not found: {db}") })),
            )
                .into_response();
        }
    };
    // Use the persisted dream config if present; otherwise default knobs over
    // every collection (a manual trigger shouldn't require opt-in).
    let mut cfg = server
        .database_config(&db)
        .map(|c| c.dream)
        .unwrap_or_default();
    cfg.enabled = true;
    // Per-request overrides.
    if let Some(m) = q.mode.as_deref() {
        cfg.mode = if m.eq_ignore_ascii_case("ladder") {
            DreamMode::Ladder
        } else {
            DreamMode::Replay
        };
    }
    if let Some(l) = q.levels {
        cfg.passes = l;
    }
    if let Some(tc) = q.tau_cohere {
        cfg.tau_cohere = tc;
    }
    if let Some(cols) = q.collections.as_deref() {
        cfg.collections = cols.split(',').map(|s| s.trim().to_string()).collect();
    }

    match dream_database(&db, &hive, &cfg) {
        Ok(report) => (StatusCode::OK, Json(serde_json::json!(report))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::{DbConfig, Server};

    /// Replay + merge must not lose the attractors: after dreaming, every
    /// original pattern still reads back to itself, and redundant duplicates
    /// collapse.
    #[test]
    fn replay_merge_preserves_attractors_and_dedups() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        let cfg = DbConfig::new("mem", 64).unwrap();
        let hive = server.create_database(cfg).unwrap();
        let col = hive.create_collection("c").unwrap();

        // Three well-separated patterns (cos ≈ 0, below tau_split so they stay
        // distinct), each written twice, plus a near-duplicate of the first
        // (cos ≈ 1, which merge should fold away).
        let mut base = vec![
            unit(&[1.0, 0.0, 0.0, 0.0]),
            unit(&[0.0, 1.0, 0.0, 0.0]),
            unit(&[0.0, 0.0, 1.0, 0.0]),
        ];
        base.push(unit(&[1.0, 0.05, 0.0, 0.0])); // near-dup of base[0]
        for p in &base {
            col.write(p).unwrap();
            col.write(p).unwrap();
        }
        let before = col.num_locations().unwrap();

        let report = dream_collection(&col, 1).unwrap();
        assert_eq!(report.locations_before, before);
        // Dreaming reorganizes in place: it must not grow the collection, and
        // merge should fold the near-duplicate away.
        assert!(report.locations_after <= before);
        assert!(report.locations_after >= 3, "distinct attractors collapsed");

        // Every distinct pattern still has a location pointing its way — the
        // attractors survived replay + merge (checked against the location set,
        // independent of read temperature).
        let (locs, _) = col.snapshot().unwrap();
        for p in &base[..3] {
            let best = locs
                .iter()
                .map(|l| dot(&l.address, p))
                .fold(f64::MIN, f64::max);
            assert!(
                best > 0.9,
                "attractor lost after dream (best cos {best:.3})"
            );
        }
    }

    /// One ladder rung: from raw episodes that share an identical context but
    /// carry different procedures (the level-1 grid case), the gated two-field
    /// write must recover two families, each whose accumulated law aligns with
    /// its own wiring. This is the engine deriving a schema from episodes — in
    /// Rust, no external math.
    #[test]
    fn ladder_rung_separates_families_by_law() {
        use heather_algebra::bind_vec;
        use heather_db::vec_ops as vops;
        use heather_db::{HardLocation, LocationId};
        use rand::SeedableRng;

        let d = 256;
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), d).unwrap();
        let hive = server
            .create_database(DbConfig::new("mem", d).unwrap())
            .unwrap();
        let episodes = hive.create_collection("episodes").unwrap();
        let dst = hive.create_collection("episodes__L1").unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let context = vops::random_unit_vector(d, &mut rng); // shared situation
        let k_a = vops::random_unit_vector(d, &mut rng); // family A wiring key
        let k_b = vops::random_unit_vector(d, &mut rng); // family B wiring key

        // Episodes: address = shared context, counter = bind(context, filler),
        // where filler ≈ the family key + per-member noise. unbind(counter,
        // context) recovers the filler — the law the family carries.
        let mut locs = Vec::new();
        let mut id = 0u64;
        for k_fam in [&k_a, &k_b] {
            for _ in 0..5 {
                let noise = vops::random_unit_vector(d, &mut rng);
                let filler = vops::normalize(
                    &k_fam
                        .iter()
                        .zip(&noise)
                        .map(|(k, n)| k + 0.15 * n)
                        .collect::<Vec<_>>(),
                );
                let procedure = bind_vec(&context, &filler);
                let mut l = HardLocation::new(LocationId(id), context.clone());
                l.counter = procedure;
                l.write_count = 1.0;
                locs.push(l);
                id += 1;
            }
        }
        let cfg = episodes.config().unwrap();
        episodes.load_snapshot(locs, cfg).unwrap();

        consolidate_rung(
            &episodes,
            &dst,
            WriteOpts {
                gate: true,
                tau_cohere: 0.2,
            },
        )
        .unwrap();

        let (fams, _) = dst.snapshot().unwrap();
        assert_eq!(
            fams.len(),
            2,
            "rung should recover two families from a shared-context episode set"
        );
        let best_align = |k: &[f64]| {
            fams.iter()
                .map(|f| dot(&heather_db_normalize(&f.counter), &heather_db_normalize(k)))
                .fold(f64::MIN, f64::max)
        };
        assert!(
            best_align(&k_a) > 0.5,
            "no family law aligns with A (best {:.2})",
            best_align(&k_a)
        );
        assert!(
            best_align(&k_b) > 0.5,
            "no family law aligns with B (best {:.2})",
            best_align(&k_b)
        );
    }

    fn unit(v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; 64];
        for (i, x) in v.iter().enumerate() {
            out[i] = *x;
        }
        heather_db_normalize(&out)
    }
}
