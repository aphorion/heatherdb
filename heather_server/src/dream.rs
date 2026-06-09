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

use heather_db::{Collection, DreamConfig, Hive, Server};

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
pub fn dream_collection(col: &Collection, passes: usize) -> heather_db::error::Result<CollectionDream> {
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

/// Dream every collection a `DreamConfig` selects in one hive. An empty
/// `collections` list means "every collection in the database".
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
    let mut collections = Vec::new();
    for name in names {
        if let Some(col) = hive.get_collection(&name)? {
            collections.push(dream_collection(&col, cfg.passes)?);
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

            for name in names {
                let col = match hive.get_collection(&name) {
                    Ok(Some(c)) => c,
                    _ => continue,
                };
                let count = col.num_locations().unwrap_or(0);
                let key = (db.clone(), name.clone());
                if last_dreamed.get(&key) == Some(&count) {
                    continue; // unchanged since last dream — nothing new to consolidate
                }
                match dream_collection(&col, cfg.passes) {
                    Ok(r) => {
                        tracing::info!(
                            db = %db, collection = %name,
                            before = r.locations_before, after = r.locations_after,
                            merged = r.merged, passes = r.passes,
                            "dreamed collection"
                        );
                        last_dreamed.insert(key, r.locations_after);
                    }
                    Err(e) => tracing::warn!(db = %db, collection = %name, error = %e, "dream failed"),
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
/// gate). Useful for operators and tests.
pub async fn trigger(
    Extension(server): Extension<Arc<Server>>,
    Path(db): Path<String>,
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
            assert!(best > 0.9, "attractor lost after dream (best cos {best:.3})");
        }
    }

    fn unit(v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; 64];
        for (i, x) in v.iter().enumerate() {
            out[i] = *x;
        }
        heather_db_normalize(&out)
    }
}
