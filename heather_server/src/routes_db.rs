//! Multi-tenant routes — `/db` management + `/db/{db}/...` scoped operations.
//!
//! These sit alongside the legacy `/collections/...` and `/algebra/...`
//! routes (which target the `default` database). Scoped handlers are
//! mostly thin wrappers that resolve a `Hive` from the path and delegate
//! to the legacy handler — keeps the per-collection logic single-sourced.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use heather_db::{DEFAULT_MAP_SIZE_MB, DbConfig, Hive, Server};

use crate::audit::AuditCtx;
use crate::models::*;
use crate::routes;

/// Readiness probe: unlike `/health` (alive-only), this proves the engine
/// can actually serve — it lists databases and runs a real LMDB read on
/// the default hive. Returns 503 when the storage layer is broken, so
/// orchestrators stop routing traffic instead of restart-looping a
/// corrupted engine that still answers `/health`.
pub async fn ready(Extension(server): Extension<Arc<Server>>) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        let _dbs = server.databases()?;
        match server.database(heather_db::DEFAULT_DB) {
            Some(hive) => hive.list_collections().map(|_| ()),
            None => Err(heather_db::HeatherError::Storage(
                "default database missing".into(),
            )),
        }
    })
    .await;

    match result {
        Ok(Ok(())) => Json(HealthResponse {
            status: "ok".to_string(),
        })
        .into_response(),
        Ok(Err(e)) => err(StatusCode::SERVICE_UNAVAILABLE, e),
        Err(e) => err(StatusCode::SERVICE_UNAVAILABLE, e),
    }
}

/// Resolve a DB name to its `Arc<Hive>`. Returns a `404 Response` if the
/// name doesn't exist on this server.
// The `Err` variant is axum's own `Response`, which is genuinely large.
// Every caller immediately `?`-propagates it into a handler's return, so
// boxing here would only add an indirection that is unwrapped one frame up.
#[allow(clippy::result_large_err)]
fn resolve_db(server: &Arc<Server>, name: &str) -> Result<Arc<Hive>, Response> {
    server.database(name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("database not found: {name}"),
            }),
        )
            .into_response()
    })
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

/* ─── /db management ───────────────────────────────────────────────────────── */

pub async fn list_databases(Extension(server): Extension<Arc<Server>>) -> Response {
    let names = match server.databases() {
        Ok(n) => n,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let mut databases = Vec::with_capacity(names.len());
    for name in names {
        let cfg = match server.database_config(&name) {
            Ok(c) => c,
            Err(_) => continue, // skip half-initialized dirs
        };
        let collections = server
            .database(&name)
            .and_then(|h| h.list_collections().ok())
            .map(|v| v.len())
            .unwrap_or(0);
        let dimension = cfg.dimension();
        databases.push(DatabaseInfo {
            name: cfg.name,
            created_at: cfg.created_at,
            dimension,
            map_size_mb: cfg.map_size_mb,
            collections,
        });
    }
    Json(ListDatabasesResponse { databases }).into_response()
}

pub async fn create_database(
    Extension(server): Extension<Arc<Server>>,
    Json(req): Json<CreateDatabaseRequest>,
) -> Response {
    let mut cfg = match DbConfig::new(&req.name, req.dimension) {
        Ok(c) => c,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    cfg.map_size_mb = req.map_size_mb.unwrap_or(DEFAULT_MAP_SIZE_MB);
    cfg.eam.mdl_gate = req.mdl_gate;

    if let Some(o) = req.eam {
        if let Some(v) = o.k {
            cfg.eam.k = v;
        }
        if let Some(v) = o.t_max {
            cfg.eam.t_max = v;
        }
        if let Some(v) = o.beta {
            cfg.eam.beta = v;
        }
        if let Some(v) = o.neighbor_cap {
            cfg.eam.neighbor_cap = v;
        }
        if let Some(v) = o.num_landmarks {
            cfg.eam.num_landmarks = v;
        }
        if let Some(v) = o.l_0 {
            cfg.eam.l_0 = v;
        }
        if let Err(e) = cfg.eam.validate() {
            return err(StatusCode::BAD_REQUEST, e);
        }
    }

    match server.create_database(cfg) {
        Ok(_) => match server.database_config(&req.name) {
            Ok(c) => {
                let dimension = c.dimension();
                Json(DatabaseInfo {
                    name: c.name,
                    created_at: c.created_at,
                    dimension,
                    map_size_mb: c.map_size_mb,
                    collections: 0,
                })
                .into_response()
            }
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        },
        Err(e) => {
            // duplicate / invalid name → 409 / 400
            let s = e.to_string();
            if s.contains("already exists") {
                err(StatusCode::CONFLICT, e)
            } else {
                err(StatusCode::BAD_REQUEST, e)
            }
        }
    }
}

pub async fn get_database(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
) -> Response {
    let cfg = match server.database_config(&db_name) {
        Ok(c) => c,
        Err(_) => {
            return err(
                StatusCode::NOT_FOUND,
                format!("database not found: {db_name}"),
            );
        }
    };
    let collections = server
        .database(&db_name)
        .and_then(|h| h.list_collections().ok())
        .map(|v| v.len())
        .unwrap_or(0);
    let dimension = cfg.dimension();
    Json(DatabaseInfo {
        name: cfg.name,
        created_at: cfg.created_at,
        dimension,
        map_size_mb: cfg.map_size_mb,
        collections,
    })
    .into_response()
}

pub async fn drop_database(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
) -> Response {
    match server.drop_database(&db_name) {
        Ok(dropped) => Json(DropDatabaseResponse { dropped }).into_response(),
        Err(e) => {
            let s = e.to_string();
            if s.contains("default") {
                err(StatusCode::FORBIDDEN, e)
            } else {
                err(StatusCode::BAD_REQUEST, e)
            }
        }
    }
}

/* ─── /db/{db}/collections/... ──────────────────────────────────────────────
 *
 * These wrap the legacy handlers in routes.rs. The legacy handlers take
 * `State<AppState> = State<Arc<Hive>>` and `Path<String>` (collection name);
 * we resolve the DB to a Hive and synthesise both extractors. Single-sourced
 * collection logic, zero duplication.
 *
 * Whenever a legacy handler grows new arguments, the wrapper here needs the
 * matching update — keep both lists in sync (callsites are line-aligned).
 */

pub async fn list_collections(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::list_collections(State(hive)).await
}

pub async fn create_collection(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::create_collection(State(hive), Json(req)).await
}

pub async fn drop_collection(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::drop_collection(State(hive), Path(col)).await
}

pub async fn write(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<WriteRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::write(State(hive), Path(col), Json(req)).await
}

pub async fn read(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    ctx: AuditCtx,
    Json(req): Json<ReadRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::read(State(hive), Path(col), ctx.for_db(&db_name), Json(req)).await
}

pub async fn attention(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    ctx: AuditCtx,
    Json(req): Json<AttentionRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::attention(State(hive), Path(col), ctx.for_db(&db_name), Json(req)).await
}

pub async fn attention_mdl(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<AttentionMdlRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::attention_mdl(State(hive), Path(col), Json(req)).await
}

pub async fn calibrate(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<CalibrateRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::calibrate(State(hive), Path(col), Json(req)).await
}

pub async fn bulk_load(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<BulkLoadRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::bulk_load(State(hive), Path(col), Json(req)).await
}

pub async fn stats(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::stats(State(hive), Path(col)).await
}

pub async fn compress(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<CompressRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::compress(State(hive), Path(col), Json(req)).await
}

pub async fn collection_config(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::collection_config(State(hive), Path(col)).await
}

pub async fn locations(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    q: axum::extract::Query<LocationsQuery>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::locations(State(hive), Path(col), q).await
}

pub async fn fingerprint(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::fingerprint(State(hive), Path(col)).await
}

pub async fn analyze(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    ctx: AuditCtx,
    Json(req): Json<AnalyzeRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::analyze(State(hive), Path(col), ctx.for_db(&db_name), Json(req)).await
}

pub async fn batch_analyze(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    Json(req): Json<BatchAnalyzeRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::batch_analyze(State(hive), Path(col), Json(req)).await
}

pub async fn get_documents(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::get_documents(State(hive), Path(col)).await
}

pub async fn get_document(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col, doc_id)): Path<(String, String, u64)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::get_document(State(hive), Path((col, doc_id))).await
}

pub async fn delete_document(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col, doc_id)): Path<(String, String, u64)>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::delete_document(State(hive), Path((col, doc_id))).await
}

pub async fn query_documents(
    Extension(server): Extension<Arc<Server>>,
    Path((db_name, col)): Path<(String, String)>,
    ctx: AuditCtx,
    Json(req): Json<QueryDocumentsRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::query_documents(State(hive), Path(col), ctx.for_db(&db_name), Json(req)).await
}

/* ─── /db/{db}/algebra/... ────────────────────────────────────────────────── */

pub async fn algebra_add(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraAddRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_add(State(hive), Json(req)).await
}

pub async fn algebra_sub(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraSubRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_sub(State(hive), Json(req)).await
}

pub async fn algebra_scale(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraScaleRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_scale(State(hive), Json(req)).await
}

pub async fn algebra_bind(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraBindRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_bind(State(hive), Json(req)).await
}

pub async fn algebra_permute(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraPermuteRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_permute(State(hive), Json(req)).await
}

pub async fn algebra_unbind(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraUnbindRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_unbind(State(hive), Json(req)).await
}

pub async fn algebra_intersect(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<AlgebraIntersectRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::algebra_intersect(State(hive), Json(req)).await
}

pub async fn compose_read(
    Extension(server): Extension<Arc<Server>>,
    Path(db_name): Path<String>,
    Json(req): Json<ComposeReadRequest>,
) -> Response {
    let hive = match resolve_db(&server, &db_name) {
        Ok(h) => h,
        Err(r) => return r,
    };
    routes::compose_read(State(hive), Json(req)).await
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use heather_db::AuditQuery;

    fn ctx() -> AuditCtx {
        AuditCtx {
            user: "alice".into(),
            scope: "root".into(),
            db: "default".into(),
        }
    }

    /// The scoped wrapper is the only place that knows which database the
    /// request targets, so this is the assertion that the `for_db` hand-off
    /// works end to end: a read through `/db/garden/...` must be recorded
    /// against `garden`, not against the legacy `default`.
    #[tokio::test]
    async fn scoped_reads_are_audited_against_the_path_database() {
        let dir = tempfile::tempdir().unwrap();
        let server = Arc::new(Server::open(dir.path(), 8).unwrap());
        server
            .create_database(DbConfig::new("garden", 8).unwrap())
            .unwrap();

        // Seed documents so the query has something to return.
        let vectors: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let mut v = vec![0.0; 8];
                v[i] = 1.0;
                v
            })
            .collect();
        let resp = write(
            Extension(server.clone()),
            Path(("garden".into(), "bids".into())),
            Json(WriteRequest {
                metadata: Some(
                    (0..5)
                        .map(|i| serde_json::json!({ "title": format!("tender-{i}") }))
                        .collect(),
                ),
                vectors,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let req: QueryDocumentsRequest = serde_json::from_value(serde_json::json!({
            "query": query,
            "n": 5,
        }))
        .unwrap();
        let resp = query_documents(
            Extension(server.clone()),
            Path(("garden".into(), "bids".into())),
            ctx(),
            Json(req),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let hive = server.database("garden").unwrap();
        let entries = hive.query_audit(&AuditQuery::new(10)).unwrap();
        assert_eq!(entries.len(), 1, "writes are not audited, reads are");
        let e = &entries[0];
        assert_eq!(e.database, "garden");
        assert_eq!(e.collection, "bids");
        assert_eq!(e.route, "documents/query");
        assert_eq!(e.user, "alice");
        assert_eq!(e.status, 200);
        assert!(e.result_count > 0, "the query returned nothing to audit");
        assert_eq!(e.document_ids.len(), e.result_count);
        assert_eq!(e.query_hash, Some(heather_db::query_hash(&query)));

        // The legacy `default` database saw none of it.
        let default_hive = server.database("default").unwrap();
        assert!(
            default_hive
                .query_audit(&AuditQuery::new(10))
                .unwrap()
                .is_empty()
        );
    }

    /// A read that fails is still recorded — a refused access attempt is
    /// exactly what an access log exists to capture.
    #[tokio::test]
    async fn failed_reads_are_audited_with_their_status() {
        let dir = tempfile::tempdir().unwrap();
        let server = Arc::new(Server::open(dir.path(), 4).unwrap());
        server
            .create_database(DbConfig::new("garden", 4).unwrap())
            .unwrap();

        // Wrong dimension → 400 from the engine.
        let resp = read(
            Extension(server.clone()),
            Path(("garden".into(), "bids".into())),
            ctx(),
            Json(serde_json::from_value(serde_json::json!({"query": [1.0, 2.0]})).unwrap()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let hive = server.database("garden").unwrap();
        let entries = hive.query_audit(&AuditQuery::new(10)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].route, "read");
        assert_eq!(entries[0].status, 400);
        assert_eq!(entries[0].result_count, 0);
    }
}
