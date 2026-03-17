mod models;
mod routes;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use axum::Router;
use clap::Parser;
use heather_db::{Hive, EAMConfig};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "heather_server", about = "HeatherDB server")]
struct Args {
    /// Directory for database storage
    #[arg(long, env = "HEATHER_DATA_DIR")]
    data_dir: PathBuf,

    /// Vector dimension (required on first run)
    #[arg(long, env = "HEATHER_DIMENSION")]
    dimension: usize,

    /// Listen port
    #[arg(long, env = "HEATHER_PORT", default_value = "6380")]
    port: u16,

    /// Bind address
    #[arg(long, env = "HEATHER_HOST", default_value = "0.0.0.0")]
    host: String,

    /// Maximum request body size in bytes (default: 2MB)
    #[arg(long, env = "HEATHER_MAX_BODY_SIZE", default_value = "2097152")]
    max_body_size: usize,

    /// Request timeout in seconds (default: 30)
    #[arg(long, env = "HEATHER_REQUEST_TIMEOUT", default_value = "30")]
    request_timeout: u64,

    /// LMDB map size in megabytes (default: 256)
    #[arg(long, env = "HEATHER_MAP_SIZE_MB", default_value = "256")]
    map_size_mb: usize,
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let config = EAMConfig::new(args.dimension).expect("invalid EAM config");
    let hive = Hive::open(&args.data_dir, config, args.map_size_mb)
        .expect("failed to open hive");

    let collections = hive.list_collections().unwrap_or_default();
    tracing::info!(
        path = %args.data_dir.display(),
        collections = collections.len(),
        "Opened hive"
    );

    let state: routes::AppState = Arc::new(hive);

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/collections", post(routes::create_collection))
        .route("/collections", get(routes::list_collections))
        .route("/collections/{name}", delete(routes::drop_collection))
        .route("/collections/{name}/write", post(routes::write))
        .route("/collections/{name}/read", post(routes::read))
        .route("/collections/{name}/stats", get(routes::stats))
        .route("/collections/{name}/config", get(routes::collection_config))
        .route("/collections/{name}/locations", get(routes::locations))
        .route("/collections/{name}/analyze", post(routes::analyze))
        .route("/collections/{name}/batch_analyze", post(routes::batch_analyze))
        .route("/collections/{name}/fingerprint", get(routes::fingerprint))
        .route("/collections/{name}/documents", get(routes::get_documents))
        .route("/collections/{name}/documents/query", post(routes::query_documents))
        .route("/collections/{name}/documents/{doc_id}", get(routes::get_document))
        .route("/algebra/add", post(routes::algebra_add))
        .route("/algebra/sub", post(routes::algebra_sub))
        .route("/algebra/scale", post(routes::algebra_scale))
        .route("/algebra/intersect", post(routes::algebra_intersect))
        .route("/compose/read", post(routes::compose_read))
        .with_state(state)
        .route("/ui", get(ui::redirect_to_ui))
        .route("/ui/", get(ui::serve_ui))
        .route("/ui/{*path}", get(ui::serve_ui))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(TimeoutLayer::with_status_code(
                    axum::http::StatusCode::REQUEST_TIMEOUT,
                    Duration::from_secs(args.request_timeout),
                )),
        )
        .layer(DefaultBodyLimit::max(args.max_body_size));

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!(
        addr = %addr,
        max_body_size = args.max_body_size,
        request_timeout_secs = args.request_timeout,
        map_size_mb = args.map_size_mb,
        "HeatherDB server listening"
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    tracing::info!("Server shut down cleanly");
}
