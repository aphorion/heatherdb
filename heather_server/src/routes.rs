use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use heather_algebra::{
    EAMSnapshot, bind_vec, compose_read as algebra_compose_read, ops, unbind_vec,
};
use heather_db::{HardLocation, Hive, LocationId, ReadStrategy};
use rayon::prelude::*;

use crate::models::*;

/// `POST /vec/bind` — circular-convolution bind of two raw vectors.
pub async fn vec_bind(Json(req): Json<VecPairRequest>) -> Response {
    if req.a.len() != req.b.len() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "a and b must have equal length" })),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "result": bind_vec(&req.a, &req.b) })),
    )
        .into_response()
}

/// `POST /vec/unbind` — circular-correlation unbind (approximate inverse).
pub async fn vec_unbind(Json(req): Json<VecPairRequest>) -> Response {
    if req.a.len() != req.b.len() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "a and b must have equal length" })),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "result": unbind_vec(&req.a, &req.b) })),
    )
        .into_response()
}

pub type AppState = Arc<Hive>;

fn error_response(status: StatusCode, msg: impl ToString) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

pub async fn health() -> Response {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
    .into_response()
}

pub async fn create_collection(
    State(hive): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> Response {
    let name = req.name.clone();
    let already_exists = hive.list_collections().is_ok_and(|c| c.contains(&name));

    match hive.create_collection(&req.name) {
        Ok(_col) => Json(CreateCollectionResponse {
            name: req.name,
            created: !already_exists,
        })
        .into_response(),
        Err(e) => {
            tracing::warn!(error = %e, "Create collection failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
    }
}

pub async fn list_collections(State(hive): State<AppState>) -> Response {
    match hive.list_collections() {
        Ok(collections) => Json(ListCollectionsResponse { collections }).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "List collections failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn drop_collection(State(hive): State<AppState>, Path(name): Path<String>) -> Response {
    match hive.drop_collection(&name) {
        Ok(dropped) => Json(DropCollectionResponse { dropped }).into_response(),
        Err(e) => {
            tracing::warn!(error = %e, collection = %name, "Drop collection failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn write(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<WriteRequest>,
) -> Response {
    let num_vectors = req.vectors.len();
    if num_vectors == 0 {
        return error_response(StatusCode::BAD_REQUEST, "vectors array must not be empty");
    }

    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => {
            tracing::warn!(error = %e, collection = %name, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    let has_metadata = req.metadata.is_some();
    let col_clone = col.clone();
    let result = tokio::task::spawn_blocking(move || {
        if let Some(metadata) = &req.metadata {
            if metadata.len() != req.vectors.len() {
                return Err(heather_db::HeatherError::InvalidInput(
                    "metadata length must match vectors length".to_string(),
                ));
            }
            let mut ids = Vec::with_capacity(req.vectors.len());
            for (vec, meta) in req.vectors.iter().zip(metadata.iter()) {
                let meta_bytes = serde_json::to_vec(meta).map_err(|e| {
                    heather_db::HeatherError::InvalidInput(format!("invalid metadata JSON: {e}"))
                })?;
                let id = col_clone.write_with_metadata(vec, &meta_bytes)?;
                ids.push(id);
            }
            Ok((ids.len(), Some(ids)))
        } else {
            col_clone.write_batch(&req.vectors)?;
            Ok::<_, heather_db::HeatherError>((req.vectors.len(), None))
        }
    })
    .await;

    match result {
        Ok(Ok((count, ids))) => {
            tracing::info!(
                collection = %name,
                vectors_written = count,
                vectors_requested = num_vectors,
                has_metadata,
                "Write completed"
            );
            Json(WriteResponse { count, ids }).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, collection = %name, "Write failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
        Err(e) => {
            tracing::error!(error = %e, "Write task panicked");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn bulk_load(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BulkLoadRequest>,
) -> Response {
    if req.addresses.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "addresses must not be empty");
    }
    if req.addresses.len() != req.counters.len() {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "addresses ({}) and counters ({}) must have equal length",
                req.addresses.len(),
                req.counters.len()
            ),
        );
    }
    if let Some(wc) = &req.write_counts
        && wc.len() != req.addresses.len()
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "write_counts ({}) must match addresses ({})",
                wc.len(),
                req.addresses.len()
            ),
        );
    }

    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let result = tokio::task::spawn_blocking(move || {
        // Reuse the target's current config (preserves the per-DB
        // dimension and EAM hyperparameters); load_snapshot validates.
        let (_, config) = col.snapshot()?;
        let dim = config.d;

        let mut locations: Vec<HardLocation> = Vec::with_capacity(req.addresses.len());
        for (i, (addr, counter)) in req.addresses.iter().zip(req.counters.iter()).enumerate() {
            if addr.len() != dim || counter.len() != dim {
                return Err(heather_db::HeatherError::DimensionMismatch {
                    expected: dim,
                    got: addr.len().max(counter.len()),
                });
            }
            let mut loc = HardLocation::new(LocationId(i as u64), addr.clone());
            loc.counter = counter.clone();
            loc.write_count = req.write_counts.as_ref().map(|wc| wc[i]).unwrap_or(1.0);
            locations.push(loc);
        }

        let n = locations.len();
        col.load_snapshot(locations, config)?;
        Ok::<_, heather_db::HeatherError>((n, dim))
    })
    .await;

    match result {
        Ok(Ok((n_loaded, dim))) => {
            tracing::info!(collection = %name, n_loaded, dim, "Bulk load completed");
            Json(BulkLoadResponse { n_loaded, dim }).into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn read(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<ReadRequest>,
) -> Response {
    let strategy_name = req.strategy.as_deref().unwrap_or("iterative");
    let strategy = match strategy_name {
        "fast" => ReadStrategy::HopfieldSS,
        "iterative" => ReadStrategy::HopfieldIter,
        other => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("unknown strategy: '{other}'. use 'iterative' or 'fast'"),
            );
        }
    };

    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => {
            tracing::warn!(error = %e, collection = %name, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    let dims = req.query.len();
    let result = tokio::task::spawn_blocking(move || col.read(&req.query, strategy)).await;

    match result {
        Ok(Ok(vec)) => {
            tracing::info!(collection = %name, strategy = strategy_name, dims, "Read completed");
            Json(ReadResponse { result: vec }).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, collection = %name, strategy = strategy_name, "Read failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
        Err(e) => {
            tracing::error!(error = %e, "Read task panicked");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn stats(State(hive): State<AppState>, Path(name): Path<String>) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    match col.stats() {
        Ok(s) => Json(StatsResponse {
            num_locations: s.num_locations,
            total_writes: s.total_writes,
            current_eta: s.current_eta,
            avg_write_count: s.avg_write_count,
            max_write_count: s.max_write_count,
        })
        .into_response(),
        Err(e) => {
            tracing::error!(error = %e, "Stats failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn collection_config(State(hive): State<AppState>, Path(name): Path<String>) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    match col.config() {
        Ok(config) => Json(ConfigResponse { config }).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "Config failed");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn locations(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    axum::extract::Query(q): axum::extract::Query<LocationsQuery>,
) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    if q.full {
        match col.snapshot() {
            Ok((locs, _cfg)) => {
                let locations: Vec<LocationFullItem> = locs
                    .into_iter()
                    .map(|l| LocationFullItem {
                        id: l.id.0 as usize,
                        write_count: l.write_count,
                        address: l.address,
                        counter: l.counter,
                    })
                    .collect();
                Json(LocationsFullResponse { locations }).into_response()
            }
            Err(e) => {
                tracing::error!(error = %e, "Full locations snapshot failed");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
            }
        }
    } else {
        match col.locations_summary() {
            Ok(summaries) => {
                let locations = summaries
                    .into_iter()
                    .map(|s| LocationSummaryItem {
                        id: s.id,
                        write_count: s.write_count,
                        avg_counter_magnitude: s.avg_counter_magnitude,
                    })
                    .collect();
                Json(LocationsResponse { locations }).into_response()
            }
            Err(e) => {
                tracing::error!(error = %e, "Locations failed");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
            }
        }
    }
}

pub async fn fingerprint(State(hive): State<AppState>, Path(name): Path<String>) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    let result = tokio::task::spawn_blocking(move || col.fingerprint()).await;

    match result {
        Ok(Ok(fp)) => {
            tracing::info!(collection = %name, has_fingerprint = fp.is_some(), "Fingerprint computed");
            Json(FingerprintResponse { fingerprint: fp }).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, collection = %name, "Fingerprint failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
        Err(e) => {
            tracing::error!(error = %e, "Fingerprint task panicked");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn analyze(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<AnalyzeRequest>,
) -> Response {
    let strategy_name = req.strategy.as_deref().unwrap_or("iterative");
    let strategy = match strategy_name {
        "fast" => ReadStrategy::HopfieldSS,
        "iterative" => ReadStrategy::HopfieldIter,
        other => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("unknown strategy: '{other}'. use 'iterative' or 'fast'"),
            );
        }
    };

    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => {
            tracing::warn!(error = %e, collection = %name, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    let result = tokio::task::spawn_blocking(move || col.analyze_read(&req.query, strategy)).await;

    match result {
        Ok(Ok(trace)) => {
            let activated_locations = trace
                .activated_locations
                .into_iter()
                .map(|al| ActivatedLocationItem {
                    id: al.id,
                    similarity: al.similarity,
                    weight: al.weight,
                })
                .collect();
            Json(AnalyzeResponse {
                iterations: trace.iterations,
                converged: trace.converged,
                activated_locations,
                result: trace.result,
            })
            .into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, collection = %name, "Analyze failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
        Err(e) => {
            tracing::error!(error = %e, "Analyze task panicked");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn batch_analyze(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<BatchAnalyzeRequest>,
) -> Response {
    if req.queries.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "queries array must not be empty");
    }

    let strategy_name = req.strategy.as_deref().unwrap_or("iterative");
    let strategy = match strategy_name {
        "fast" => ReadStrategy::HopfieldSS,
        "iterative" => ReadStrategy::HopfieldIter,
        other => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("unknown strategy: '{other}'. use 'iterative' or 'fast'"),
            );
        }
    };

    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => {
            tracing::warn!(error = %e, collection = %name, "Failed to get collection");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    };

    let num_queries = req.queries.len();
    let result = tokio::task::spawn_blocking(move || {
        req.queries
            .par_iter()
            .map(|query| {
                let trace = col.analyze_read(query, strategy)?;
                let activated_locations = trace
                    .activated_locations
                    .into_iter()
                    .map(|al| ActivatedLocationItem {
                        id: al.id,
                        similarity: al.similarity,
                        weight: al.weight,
                    })
                    .collect();
                Ok::<_, heather_db::HeatherError>(AnalyzeResponse {
                    iterations: trace.iterations,
                    converged: trace.converged,
                    activated_locations,
                    result: trace.result,
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await;

    match result {
        Ok(Ok(results)) => {
            tracing::info!(
                collection = %name,
                queries = num_queries,
                "Batch analyze completed"
            );
            Json(BatchAnalyzeResponse { results }).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, collection = %name, "Batch analyze failed");
            error_response(StatusCode::BAD_REQUEST, e)
        }
        Err(e) => {
            tracing::error!(error = %e, "Batch analyze task panicked");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, e)
        }
    }
}

pub async fn get_documents(State(hive): State<AppState>, Path(name): Path<String>) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let result = tokio::task::spawn_blocking(move || col.list_documents()).await;

    match result {
        Ok(Ok(docs)) => {
            let documents: Vec<DocumentItem> = docs
                .into_iter()
                .filter_map(|(id, _vec, meta)| {
                    serde_json::from_slice(&meta)
                        .ok()
                        .map(|metadata| DocumentItem { id, metadata })
                })
                .collect();
            Json(DocumentsResponse { documents }).into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn get_document(
    State(hive): State<AppState>,
    Path((name, doc_id)): Path<(String, u64)>,
) -> Response {
    let col = match hive.get_collection(&name) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("collection '{name}' not found"),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let result = tokio::task::spawn_blocking(move || col.get_document(doc_id)).await;

    match result {
        Ok(Ok(Some((_vec, meta)))) => {
            let metadata: serde_json::Value =
                serde_json::from_slice(&meta).unwrap_or(serde_json::Value::Null);
            Json(DocumentResponse {
                id: doc_id,
                metadata,
            })
            .into_response()
        }
        Ok(Ok(None)) => error_response(
            StatusCode::NOT_FOUND,
            format!("document {doc_id} not found"),
        ),
        Ok(Err(e)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn query_documents(
    State(hive): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<QueryDocumentsRequest>,
) -> Response {
    let col = match hive.get_or_create_collection(&name) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let result = tokio::task::spawn_blocking(move || col.query_documents(&req.query, req.n)).await;

    match result {
        Ok(Ok(results)) => {
            let results: Vec<QueryDocumentResult> = results
                .into_iter()
                .filter_map(|(id, similarity, meta)| {
                    serde_json::from_slice(&meta)
                        .ok()
                        .map(|metadata| QueryDocumentResult {
                            id,
                            similarity,
                            metadata,
                        })
                })
                .collect();
            Json(QueryDocumentsResponse { results }).into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

// --- Algebra endpoints ---

pub async fn algebra_add(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraAddRequest>,
) -> Response {
    let col_a = match hive.get_collection(&req.source_a) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_a),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_b = match hive.get_collection(&req.source_b) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_b),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let result = tokio::task::spawn_blocking(move || {
        let snap_a = EAMSnapshot::from_collection(&col_a)?;
        let snap_b = EAMSnapshot::from_collection(&col_b)?;
        let combined = ops::add_with_limit(&snap_a, &snap_b, req.max_cross_k.unwrap_or(0))?;
        let num = combined.num_locations();
        combined.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, num_locations, "Algebra add completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn algebra_sub(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraSubRequest>,
) -> Response {
    let col_a = match hive.get_collection(&req.source_a) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_a),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_b = match hive.get_collection(&req.source_b) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_b),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let result = tokio::task::spawn_blocking(move || {
        let snap_a = EAMSnapshot::from_collection(&col_a)?;
        let snap_b = EAMSnapshot::from_collection(&col_b)?;
        let diff = ops::sub_with_limit(&snap_a, &snap_b, req.max_cross_k.unwrap_or(0))?;
        let num = diff.num_locations();
        diff.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, num_locations, "Algebra sub completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn algebra_bind(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraBindRequest>,
) -> Response {
    let col_a = match hive.get_collection(&req.source_a) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_a),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_b = match hive.get_collection(&req.source_b) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_b),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let result = tokio::task::spawn_blocking(move || {
        let snap_a = EAMSnapshot::from_collection(&col_a)?;
        let snap_b = EAMSnapshot::from_collection(&col_b)?;
        let bound =
            heather_algebra::bind::bind_with_limit(&snap_a, &snap_b, req.max_cross_k.unwrap_or(0))?;
        let num = bound.num_locations();
        bound.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, num_locations, "Algebra bind completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn algebra_unbind(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraUnbindRequest>,
) -> Response {
    let col_source = match hive.get_collection(&req.source) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let key_vector = req.key_vector.clone();
    let result = tokio::task::spawn_blocking(move || {
        let snap_source = EAMSnapshot::from_collection(&col_source)?;
        // Build a single-location key snapshot from the raw vector.
        if key_vector.len() != snap_source.dim() {
            return Err(heather_algebra::AlgebraError::DimensionMismatch {
                left: snap_source.dim(),
                right: key_vector.len(),
            });
        }
        let mut key_loc = heather_db::HardLocation::new(
            heather_db::LocationId(0),
            heather_db::vec_ops::normalize(&key_vector),
        );
        key_loc.counter = key_vector.clone();
        key_loc.write_count = 1.0;
        let snap_key = EAMSnapshot::new(vec![key_loc], snap_source.config.clone())?;
        let unbound = heather_algebra::bind::unbind(&snap_source, &snap_key)?;
        let num = unbound.num_locations();
        unbound.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, num_locations, "Algebra unbind completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn algebra_scale(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraScaleRequest>,
) -> Response {
    let col_source = match hive.get_collection(&req.source) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let alpha = req.alpha;
    let result = tokio::task::spawn_blocking(move || {
        let snap = EAMSnapshot::from_collection(&col_source)?;
        let scaled = ops::scale(&snap, alpha)?;
        let num = scaled.num_locations();
        scaled.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, alpha, num_locations, "Algebra scale completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn algebra_permute(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraPermuteRequest>,
) -> Response {
    // Exactly one of `seed` / `name` must be supplied.
    let permutation_key = match (req.seed, req.name.as_ref()) {
        (Some(seed), None) => Ok(PermKey::Seed(seed)),
        (None, Some(name)) => Ok(PermKey::Name(name.clone())),
        _ => Err("provide exactly one of `seed` or `name`".to_string()),
    };
    let permutation_key = match permutation_key {
        Ok(k) => k,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
    };

    let col_source = match hive.get_collection(&req.source) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let power = req.power;
    let result = tokio::task::spawn_blocking(move || {
        let snap = EAMSnapshot::from_collection(&col_source)?;
        let rho = match permutation_key {
            PermKey::Seed(seed) => heather_algebra::Permutation::from_seed(snap.dim(), seed),
            PermKey::Name(name) => heather_algebra::Permutation::from_name(snap.dim(), &name),
        };
        let permuted = heather_algebra::permute_snapshot(&snap, &rho, power)?;
        let num = permuted.num_locations();
        permuted.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, power, num_locations, "Algebra permute completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

enum PermKey {
    Seed(u64),
    Name(String),
}

pub async fn algebra_intersect(
    State(hive): State<AppState>,
    Json(req): Json<AlgebraIntersectRequest>,
) -> Response {
    let col_a = match hive.get_collection(&req.source_a) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_a),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_b = match hive.get_collection(&req.source_b) {
        Ok(Some(col)) => col,
        Ok(None) => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("source collection '{}' not found", req.source_b),
            );
        }
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let col_target = match hive.get_or_create_collection(&req.target) {
        Ok(col) => col,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let target_name = req.target.clone();
    let threshold = req.threshold;
    let result = tokio::task::spawn_blocking(move || {
        let snap_a = EAMSnapshot::from_collection(&col_a)?;
        let snap_b = EAMSnapshot::from_collection(&col_b)?;
        let inter = ops::intersect(&snap_a, &snap_b, threshold)?;
        let num = inter.num_locations();
        inter.into_collection(&col_target)?;
        Ok::<_, heather_algebra::AlgebraError>(num)
    })
    .await;

    match result {
        Ok(Ok(num_locations)) => {
            tracing::info!(target = %target_name, threshold, num_locations, "Algebra intersect completed");
            Json(AlgebraResponse {
                collection: target_name,
                num_locations,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

// --- Compose endpoints ---

pub async fn compose_read(
    State(hive): State<AppState>,
    Json(req): Json<ComposeReadRequest>,
) -> Response {
    if req.collections.len() < 2 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "compose read requires at least 2 collections",
        );
    }

    let mut cols = Vec::with_capacity(req.collections.len());
    for name in &req.collections {
        match hive.get_collection(name) {
            Ok(Some(col)) => cols.push(col),
            Ok(None) => {
                return error_response(
                    StatusCode::NOT_FOUND,
                    format!("collection '{name}' not found"),
                );
            }
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }

    let names = req.collections.clone();
    let routing_sharpness = req.routing_sharpness;
    let query = req.query;

    let result = tokio::task::spawn_blocking(move || {
        let snapshots: Vec<EAMSnapshot> = cols
            .iter()
            .map(|col| EAMSnapshot::from_collection(col))
            .collect::<Result<Vec<_>, _>>()?;

        let refs: Vec<&EAMSnapshot> = snapshots.iter().collect();
        let cr = algebra_compose_read(&query, &refs, routing_sharpness);
        Ok::<_, heather_algebra::AlgebraError>(cr)
    })
    .await;

    match result {
        Ok(Ok(cr)) => {
            let weights: HashMap<String, f64> = names
                .iter()
                .zip(cr.weights.iter())
                .map(|(n, &w)| (n.clone(), w))
                .collect();
            let confidences: HashMap<String, f64> = names
                .iter()
                .zip(cr.confidences.iter())
                .map(|(n, &c)| (n.clone(), c))
                .collect();
            tracing::info!(
                collections = ?names,
                routing_sharpness,
                "Compose read completed"
            );
            Json(ComposeReadResponse {
                result: cr.result,
                weights,
                confidences,
            })
            .into_response()
        }
        Ok(Err(e)) => error_response(StatusCode::BAD_REQUEST, e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
