mod auth;
mod models;
mod routes;
mod routes_db;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Extension};
use axum::middleware;
use axum::routing::{delete, get, post};
use axum::Router;
use clap::Parser;
use heather_db::{Server, DEFAULT_DB};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "heather_server", about = "HeatherDB server")]
struct Args {
    /// Directory for database storage (root for `db/<name>/...`).
    #[arg(long, env = "HEATHER_DATA_DIR")]
    data_dir: PathBuf,

    /// Vector dimension for the auto-created `default` database. Once
    /// `default` exists on disk this value is ignored — the persisted
    /// `db.toml` wins. Other databases set their own dimension at create.
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

    /// LMDB map size in megabytes for the auto-created `default` DB
    /// (default: 256). Other DBs set their own.
    #[arg(long, env = "HEATHER_MAP_SIZE_MB", default_value = "256")]
    map_size_mb: usize,

    /// Optional bearer token. When set, every route except /health requires
    /// `Authorization: Bearer <token>`. When unset, the server runs
    /// unauthenticated (a one-time warning is printed at boot).
    #[arg(long, env = "HEATHER_AUTH_TOKEN")]
    auth_token: Option<String>,
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

    // Open the multi-tenant server. On a fresh boot this lazily creates
    // the `default` database with the dimension we pass here.
    let server = Server::open(&args.data_dir, args.dimension).expect("failed to open server");

    let databases = server.databases().unwrap_or_default();
    tracing::info!(
        path = %args.data_dir.display(),
        databases = databases.len(),
        names = ?databases,
        "Opened server"
    );

    // The legacy /collections/... routes target the `default` database.
    // Resolve that now and pass it as State to the existing handlers; the
    // new /db/{db}/... routes get the full `Server` via Extension.
    let server = Arc::new(server);
    let default_hive = server
        .database(DEFAULT_DB)
        .expect("default database missing immediately after open");

    // Auth.
    let auth_state = match &args.auth_token {
        Some(t) if !t.is_empty() => {
            tracing::info!("Auth: bearer token enabled");
            auth::AuthConfig::with_token(t.clone())
        }
        _ => {
            tracing::warn!(
                "Auth: DISABLED (HEATHER_AUTH_TOKEN unset). Anyone who can reach \
                 this port has full read+write access. Set HEATHER_AUTH_TOKEN to \
                 enable Bearer-token auth on every route except /health."
            );
            auth::AuthConfig::disabled()
        }
    };

    // Legacy default-DB router — every existing /collections/... and
    // /algebra/... route, unchanged. State is the default DB's Hive.
    let legacy_router = Router::new()
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
        .with_state(default_hive);

    // /db management routes.
    let db_admin_router = Router::new()
        .route("/db", get(routes_db::list_databases))
        .route("/db", post(routes_db::create_database))
        .route("/db/{db}", get(routes_db::get_database))
        .route("/db/{db}", delete(routes_db::drop_database));

    // Scoped /db/{db}/collections/... and /db/{db}/algebra/...
    let db_scoped_router = Router::new()
        .route("/db/{db}/collections", post(routes_db::create_collection))
        .route("/db/{db}/collections", get(routes_db::list_collections))
        .route("/db/{db}/collections/{name}", delete(routes_db::drop_collection))
        .route("/db/{db}/collections/{name}/write", post(routes_db::write))
        .route("/db/{db}/collections/{name}/read", post(routes_db::read))
        .route("/db/{db}/collections/{name}/stats", get(routes_db::stats))
        .route("/db/{db}/collections/{name}/config", get(routes_db::collection_config))
        .route("/db/{db}/collections/{name}/locations", get(routes_db::locations))
        .route("/db/{db}/collections/{name}/analyze", post(routes_db::analyze))
        .route("/db/{db}/collections/{name}/batch_analyze", post(routes_db::batch_analyze))
        .route("/db/{db}/collections/{name}/fingerprint", get(routes_db::fingerprint))
        .route("/db/{db}/collections/{name}/documents", get(routes_db::get_documents))
        .route("/db/{db}/collections/{name}/documents/query", post(routes_db::query_documents))
        .route("/db/{db}/collections/{name}/documents/{doc_id}", get(routes_db::get_document))
        .route("/db/{db}/algebra/add", post(routes_db::algebra_add))
        .route("/db/{db}/algebra/sub", post(routes_db::algebra_sub))
        .route("/db/{db}/algebra/scale", post(routes_db::algebra_scale))
        .route("/db/{db}/algebra/intersect", post(routes_db::algebra_intersect))
        .route("/db/{db}/compose/read", post(routes_db::compose_read));

    let app = Router::new()
        .route("/health", get(routes::health))
        .merge(legacy_router)
        .merge(db_admin_router)
        .merge(db_scoped_router)
        // Auth middleware sees every request and short-circuits unauthorized
        // ones before any handler runs. /health is whitelisted in-middleware.
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::middleware,
        ))
        // Server is needed by the /db/* routes — pass via extension so it
        // doesn't fight the legacy State<Arc<Hive>>.
        .layer(Extension(server.clone()))
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
