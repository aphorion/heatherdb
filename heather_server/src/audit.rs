//! HTTP surface of the access log: instrumentation of the read routes, plus
//! the operator endpoint that reads it back.
//!
//! The storage design, the buffering trade-off, and the reason the query
//! vector is hashed rather than stored all live in `heather_db::audit`. This
//! module is the plumbing:
//!
//!   - [`AuditCtx`] — an infallible extractor that reads the authenticated
//!     identity the auth middleware left in the request extensions. Read
//!     handlers in `routes.rs` take one; the `routes_db.rs` wrappers overwrite
//!     its `db` field with the database from the path. That is why the read
//!     routes are instrumented **once each in `routes.rs`** and the scoped
//!     shapes inherit it — same rule as every other handler in the dual
//!     router.
//!   - [`record`] — the single call site shape for "this read happened".
//!   - [`query_audit`] — `GET /db/{db}/audit`, the raw paginated log. Its
//!     visibility policy (`AuditConfig::visibility`, Root-only by default) is
//!     enforced in `auth::middleware` / `users::is_authorized`; see
//!     `docs/api.md` for the reasoning. This endpoint hands back raw events
//!     only — any rollup or "what does this mean" computation belongs in a
//!     consumer built on top, not in the engine.
//!   - [`run_flush_loop`] — the periodic half of the flush policy.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Extension, FromRequestParts, Path, Query};
use axum::http::StatusCode;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use heather_db::{AuditQuery, AuditRecord, DEFAULT_DB, Hive, Server};

use crate::auth::AuthedUser;
use crate::models::ErrorResponse;

/// Rows returned by `GET /db/{db}/audit` when the caller doesn't say.
pub const DEFAULT_AUDIT_LIMIT: usize = 100;
/// Hard ceiling on rows per audit request. A record is small but unbounded
/// paging over a million-row log would materialise the whole thing in memory
/// and in one JSON body — page with `until` instead.
pub const MAX_AUDIT_LIMIT: usize = 1_000;

/// How often the background flusher wakes. The per-database
/// `audit.flush_interval_secs` decides whether that database is actually due.
const FLUSH_TICK: Duration = Duration::from_secs(1);

/* ─── identity for a recorded read ─────────────────────────────────────────── */

/// Who is making this request, and against which database.
///
/// Infallible on purpose: a missing identity (auth disabled) must degrade to
/// an anonymous audit entry, never to a failed read.
#[derive(Clone, Debug)]
pub struct AuditCtx {
    pub user: String,
    pub scope: String,
    pub db: String,
}

impl AuditCtx {
    /// Re-point the context at the database named in the path. Used by the
    /// `/db/{db}/...` wrappers, which know the database the legacy handler
    /// they delegate to does not.
    pub fn for_db(mut self, db: &str) -> Self {
        db.clone_into(&mut self.db);
        self
    }
}

impl<S> FromRequestParts<S> for AuditCtx
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let (user, scope) = match parts.extensions.get::<AuthedUser>() {
            Some(a) => (a.user.name.clone(), a.user.scope.label()),
            // Auth disabled: there is no identity to record. `-` is a value an
            // operator can filter on, and a username can never be `-` (the
            // validator demands a leading letter), so it can't collide.
            None => ("-".to_string(), "-".to_string()),
        };
        Ok(Self {
            user,
            scope,
            db: DEFAULT_DB.to_string(),
        })
    }
}

/// What a read returned, in audit terms.
pub struct ReadEvent<'a> {
    pub collection: &'a str,
    /// Logical route name — the same string on the legacy and scoped shapes.
    pub route: &'a str,
    pub status: StatusCode,
    pub result_count: usize,
    pub location_ids: Vec<u64>,
    pub document_ids: Vec<u64>,
    pub query_hash: Option<String>,
    /// The raw query vector, if the caller has one to offer. Only actually
    /// persisted when the database's `AuditConfig::store_raw_query` is set —
    /// [`record`] is the single place that decision is made.
    pub query_raw: Option<Vec<f64>>,
}

/// Stage one access record. Cheap: a mutex, a push, and (rarely) a spawned
/// blocking flush. Never opens an LMDB transaction on the calling thread.
pub fn record(hive: &Arc<Hive>, ctx: &AuditCtx, ev: ReadEvent<'_>) {
    if !hive.audit().enabled() {
        return;
    }
    // Single point of truth for the raw-vs-hashed decision: even if a caller
    // passed a raw vector along, it's only actually persisted when this
    // database's `store_raw_query` knob says so.
    let query_raw = if hive.audit().config().store_raw_query {
        ev.query_raw
    } else {
        None
    };
    let due = hive.record_audit(AuditRecord {
        seq: 0,          // assigned by the log
        timestamp_ms: 0, // assigned by the log
        user: ctx.user.clone(),
        scope: ctx.scope.clone(),
        database: ctx.db.clone(),
        collection: ev.collection.to_string(),
        route: ev.route.to_string(),
        status: ev.status.as_u16(),
        result_count: ev.result_count,
        location_ids: ev.location_ids,
        document_ids: ev.document_ids,
        query_hash: ev.query_hash,
        query_raw,
    });
    if due {
        // Threshold reached — drain on a blocking thread so the response path
        // returns now and the LMDB write never runs on the async reactor.
        let hive = hive.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = hive.flush_audit() {
                tracing::warn!(error = %e, "audit: threshold flush failed");
            }
        });
    }
}

/* ─── background flushing ──────────────────────────────────────────────────── */

/// Periodic half of the flush policy: wake every second, flush each database
/// whose `audit.flush_interval_secs` has elapsed and which has staged records.
pub async fn run_flush_loop(server: Arc<Server>) {
    let mut tick = tokio::time::interval(FLUSH_TICK);
    let mut last: HashMap<String, std::time::Instant> = HashMap::new();
    loop {
        tick.tick().await;
        let names = server.databases().unwrap_or_default();
        for name in names {
            let Some(hive) = server.database(&name) else {
                continue;
            };
            let cfg = hive.audit().config();
            if !cfg.enabled {
                continue;
            }
            let due = last
                .get(&name)
                .map(|t| t.elapsed() >= Duration::from_secs(cfg.flush_interval_secs.max(1)))
                .unwrap_or(true);
            if !due || hive.audit().pending() == 0 {
                continue;
            }
            last.insert(name.clone(), std::time::Instant::now());
            let _ = tokio::task::spawn_blocking(move || hive.flush_audit()).await;
        }
    }
}

/// Flush every database's staged records. Called on graceful shutdown so the
/// loss window only opens on a hard kill.
pub async fn flush_all(server: &Arc<Server>) {
    for name in server.databases().unwrap_or_default() {
        let Some(hive) = server.database(&name) else {
            continue;
        };
        if let Ok(Err(e)) = tokio::task::spawn_blocking(move || hive.flush_audit()).await {
            tracing::warn!(error = %e, database = %name, "audit: shutdown flush failed");
        }
    }
}

/* ─── read-back endpoints ──────────────────────────────────────────────────── */

#[derive(Debug, Default, Deserialize)]
pub struct AuditParams {
    pub user: Option<String>,
    pub collection: Option<String>,
    pub route: Option<String>,
    /// Inclusive lower bound, unix milliseconds.
    pub since: Option<u64>,
    /// Inclusive upper bound, unix milliseconds.
    pub until: Option<u64>,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
pub struct AuditResponse {
    pub database: String,
    pub count: usize,
    pub limit: usize,
    pub entries: Vec<AuditRecord>,
}

/// `GET /db/{db}/audit` — the access log, newest first. **Root-only.**
pub async fn query_audit(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Query(p): Query<AuditParams>,
) -> Response {
    let Some(hive) = server.database(&db_name) else {
        return err(
            StatusCode::NOT_FOUND,
            format!("database not found: {db_name}"),
        );
    };
    if !hive.audit().enabled() {
        return err(
            StatusCode::CONFLICT,
            format!("audit logging is disabled for database '{db_name}' (see [audit] in db.toml)"),
        );
    }
    let limit = p
        .limit
        .unwrap_or(DEFAULT_AUDIT_LIMIT)
        .clamp(1, MAX_AUDIT_LIMIT);
    let q = AuditQuery {
        user: p.user,
        collection: p.collection,
        route: p.route,
        since_ms: p.since,
        until_ms: p.until,
        ..AuditQuery::new(limit)
    };

    // Blocking: `query_audit` flushes first, which opens a write txn.
    match tokio::task::spawn_blocking(move || hive.query_audit(&q)).await {
        Ok(Ok(entries)) => Json(AuditResponse {
            database: db_name,
            count: entries.len(),
            limit,
            entries,
        })
        .into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

fn err(status: StatusCode, msg: impl ToString) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::{AuditConfig, DbConfig};

    fn ctx() -> AuditCtx {
        AuditCtx {
            user: "alice".into(),
            scope: "root".into(),
            db: DEFAULT_DB.into(),
        }
    }

    fn server_with(audit: AuditConfig) -> (tempfile::TempDir, Arc<Server>) {
        let dir = tempfile::tempdir().unwrap();
        let server = Arc::new(Server::open(dir.path(), 8).unwrap());
        let mut cfg = DbConfig::new("garden", 8).unwrap();
        cfg.audit = audit;
        server.create_database(cfg).unwrap();
        (dir, server)
    }

    async fn body_json(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn event<'a>(collection: &'a str, route: &'a str, docs: Vec<u64>) -> ReadEvent<'a> {
        ReadEvent {
            collection,
            route,
            status: StatusCode::OK,
            result_count: docs.len(),
            location_ids: vec![],
            document_ids: docs,
            query_hash: Some(heather_db::query_hash(&[1.0, 2.0])),
            query_raw: Some(vec![1.0, 2.0]),
        }
    }

    #[test]
    fn for_db_repoints_the_context() {
        assert_eq!(ctx().for_db("garden").db, "garden");
    }

    /// End-to-end through the HTTP handler: record, then read back with the
    /// filters the route exposes.
    #[tokio::test]
    async fn audit_endpoint_returns_recorded_reads() {
        let (_d, server) = server_with(AuditConfig::default());
        let hive = server.database("garden").unwrap();
        let c = ctx().for_db("garden");
        record(&hive, &c, event("bids", "attention", vec![3]));
        record(&hive, &c, event("hr", "read", vec![]));

        let resp = query_audit(
            Extension(server.clone()),
            Path("garden".into()),
            Query(AuditParams::default()),
        )
        .await;
        let (status, body) = body_json(resp).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["count"], 2);
        assert_eq!(body["limit"], DEFAULT_AUDIT_LIMIT);
        assert_eq!(body["entries"][0]["collection"], "hr");
        assert_eq!(body["entries"][1]["database"], "garden");
        assert_eq!(body["entries"][1]["user"], "alice");
        assert_eq!(body["entries"][1]["document_ids"][0], 3);
        // The vector itself is never in the record by default — only its hash.
        assert!(body["entries"][1]["query_hash"].is_string());
        assert!(body["entries"][1]["query_raw"].is_null());

        // Filter by collection.
        let resp = query_audit(
            Extension(server.clone()),
            Path("garden".into()),
            Query(AuditParams {
                collection: Some("bids".into()),
                ..Default::default()
            }),
        )
        .await;
        let (_, body) = body_json(resp).await;
        assert_eq!(body["count"], 1);
        assert_eq!(body["entries"][0]["collection"], "bids");
    }

    #[tokio::test]
    async fn audit_limit_is_clamped_to_the_maximum() {
        let (_d, server) = server_with(AuditConfig::default());
        let resp = query_audit(
            Extension(server),
            Path("garden".into()),
            Query(AuditParams {
                limit: Some(usize::MAX),
                ..Default::default()
            }),
        )
        .await;
        let (_, body) = body_json(resp).await;
        assert_eq!(body["limit"], MAX_AUDIT_LIMIT);
    }

    #[tokio::test]
    async fn store_raw_query_opts_in_the_raw_vector() {
        let (_d, server) = server_with(AuditConfig {
            store_raw_query: true,
            ..Default::default()
        });
        let hive = server.database("garden").unwrap();
        let c = ctx().for_db("garden");
        record(&hive, &c, event("bids", "attention", vec![3]));

        let resp = query_audit(
            Extension(server),
            Path("garden".into()),
            Query(AuditParams::default()),
        )
        .await;
        let (_, body) = body_json(resp).await;
        assert_eq!(
            body["entries"][0]["query_raw"],
            serde_json::json!([1.0, 2.0])
        );
    }

    #[tokio::test]
    async fn disabled_audit_records_nothing_and_the_routes_say_so() {
        let (_d, server) = server_with(AuditConfig {
            enabled: false,
            ..Default::default()
        });
        let hive = server.database("garden").unwrap();
        let c = ctx().for_db("garden");
        for _ in 0..5 {
            record(&hive, &c, event("bids", "attention", vec![1]));
        }
        assert_eq!(hive.audit().pending(), 0);
        assert_eq!(hive.flush_audit().unwrap(), 0);

        let (status, body) = body_json(
            query_audit(
                Extension(server.clone()),
                Path("garden".into()),
                Query(AuditParams::default()),
            )
            .await,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(body["error"].as_str().unwrap().contains("disabled"));
    }

    #[tokio::test]
    async fn unknown_database_is_404() {
        let (_d, server) = server_with(AuditConfig::default());
        let (status, _) = body_json(
            query_audit(
                Extension(server.clone()),
                Path("nope".into()),
                Query(AuditParams::default()),
            )
            .await,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
