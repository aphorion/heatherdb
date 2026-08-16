mod audit;
mod auth;
mod backup;
mod cli;
mod dream;
mod limits;
mod models;
mod routes;
mod routes_db;
mod tokens;
mod users;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Extension};
use axum::middleware;
use axum::routing::{delete, get, post};
use clap::{Parser, Subcommand};
use heather_db::{DEFAULT_DB, Server};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "heather_server", about = "HeatherDB server")]
struct Args {
    /// Directory for database storage (root for `db/<name>/...` and the
    /// users.json file).
    #[arg(long, env = "HEATHER_DATA_DIR", global = true)]
    data_dir: Option<PathBuf>,

    /// Vector dimension for the auto-created `default` database. Once
    /// `default` exists on disk this value is ignored — the persisted
    /// `db.toml` wins. Other databases set their own dimension at create.
    #[arg(long, env = "HEATHER_DIMENSION", default_value = "128", global = true)]
    dimension: usize,

    /// Initial hard-location count (`l_0`) for the auto-created `default`
    /// database. 0 (the default) is data-seeded: hard locations are grown from
    /// the first writes, never pre-seeded with random vectors. Ignored once
    /// `default` exists on disk. Set a positive value to pre-seed a random
    /// initial codebook.
    #[arg(long, env = "HEATHER_L0", default_value = "0", global = true)]
    l0: usize,

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

    /// DANGER: disable HTTP Basic Auth entirely. Local dev only — anyone
    /// who can reach the port has full read+write access. Refuses to
    /// accept silently: emits a one-time loud warning at boot.
    #[arg(long, env = "HEATHER_AUTH_DISABLED", default_value = "false")]
    auth_disabled: bool,

    /// First-boot admin username. Used only when the user store is empty
    /// (i.e. brand-new data dir). After the user exists, this is ignored —
    /// rename via `heather_server user delete <old> && user create <new>`.
    #[arg(long, env = "HEATHER_ADMIN_USER", default_value = "admin")]
    admin_user: String,

    /// First-boot admin password. Used only when the user store is empty.
    /// If unset and the store is empty, the engine generates a random
    /// password, prints it ONCE to stderr, and writes it to
    /// `$DATA_DIR/initial-admin-password` (mode 0600).
    #[arg(long, env = "HEATHER_ADMIN_PASSWORD")]
    admin_password: Option<String>,

    /// Max locations an algebra add/sub/bind result may contain. Bounds the
    /// n×m cross product before allocation. 0 disables the cap.
    #[arg(long, env = "HEATHER_MAX_ALGEBRA_LOCATIONS",
          default_value_t = limits::DEFAULT_MAX_ALGEBRA_LOCATIONS)]
    max_algebra_locations: usize,

    /// Max address/counter pairs per bulk_load request. 0 disables the cap.
    #[arg(long, env = "HEATHER_MAX_BULK_ITEMS",
          default_value_t = limits::DEFAULT_MAX_BULK_ITEMS)]
    max_bulk_items: usize,

    /// Max queries per batch_analyze request. 0 disables the cap.
    #[arg(long, env = "HEATHER_MAX_BATCH_QUERIES",
          default_value_t = limits::DEFAULT_MAX_BATCH_QUERIES)]
    max_batch_queries: usize,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Manage HTTP Basic Auth users (created in $HEATHER_DATA_DIR/users.json).
    User {
        #[command(subcommand)]
        cmd: cli::UserCmd,
    },
    /// Cold backup / restore of the data dir (or one database) as tar.gz.
    Backup {
        #[command(subcommand)]
        cmd: backup::BackupCmd,
    },
    Restore {
        #[command(subcommand)]
        cmd: backup::RestoreCmd,
    },
    /// Live consistent snapshot of one database (safe with engine running).
    Snapshot {
        #[command(subcommand)]
        cmd: backup::SnapshotCmd,
    },
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

fn main() -> std::process::ExitCode {
    let args = Args::parse();

    // Sub-commands run synchronously and exit, no HTTP server.
    if let Some(cmd) = &args.command {
        let data_dir = match resolve_data_dir(&args) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        return match cmd {
            Cmd::User { cmd } => cli::run(&data_dir, cmd),
            Cmd::Backup { cmd } => backup::run_backup(&data_dir, cmd),
            Cmd::Restore { cmd } => backup::run_restore(&data_dir, cmd),
            Cmd::Snapshot { cmd } => backup::run_snapshot(&data_dir, cmd),
        };
    }

    serve(args)
}

fn resolve_data_dir(args: &Args) -> Result<PathBuf, String> {
    args.data_dir
        .clone()
        .ok_or_else(|| "--data-dir / HEATHER_DATA_DIR is required".into())
}

#[tokio::main(flavor = "multi_thread")]
async fn serve(args: Args) -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let data_dir = match resolve_data_dir(&args) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    limits::init(limits::Limits {
        max_algebra_locations: args.max_algebra_locations,
        max_bulk_items: args.max_bulk_items,
        max_batch_queries: args.max_batch_queries,
    });

    // Open the multi-tenant server. On a fresh boot this lazily creates
    // the `default` database with the dimension we pass here.
    // Keeps the `--l0` seeding knob while reporting a failed open instead of
    // panicking, which is what the hardening pass was after.
    let server = match Server::open_with(&data_dir, args.dimension, args.l0) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: open server: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let databases = server.databases().unwrap_or_default();
    tracing::info!(
        path = %data_dir.display(),
        databases = databases.len(),
        names = ?databases,
        "Opened server"
    );

    let server = Arc::new(server);
    let default_hive = server
        .database(DEFAULT_DB)
        .expect("default database missing immediately after open");

    // Auth — load (or create) the user store.
    let user_store = match users::UserStore::load(&data_dir) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("error: load user store: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // Session-token store (shares the system LMDB env with the user store).
    let tokens = match tokens::Tokens::load(&data_dir) {
        Ok(t) => Arc::new(t),
        Err(e) => {
            eprintln!("error: load token store: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let auth_state = if args.auth_disabled {
        tracing::warn!(
            "Auth: DISABLED (HEATHER_AUTH_DISABLED=1). Anyone who can reach \
             this port has full read+write access. Drop the env var or the \
             flag to re-enable."
        );
        auth::AuthState::disabled(user_store.clone(), server.clone())
    } else {
        // First-boot bootstrap: mint an admin user. If the operator gave
        // us HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD use those;
        // otherwise generate a random password and print it once.
        if let Err(e) = auth::bootstrap_admin_if_needed(
            &user_store,
            &args.admin_user,
            args.admin_password.as_deref(),
            &data_dir,
        ) {
            eprintln!("error: bootstrap admin user: {e}");
            return std::process::ExitCode::FAILURE;
        }
        tracing::info!(
            users = user_store.list().len(),
            store = %user_store.path().display(),
            "Auth: HTTP Basic enabled"
        );
        auth::AuthState::enabled(user_store.clone(), tokens.clone(), server.clone())
    };

    // Legacy default-DB router — every existing /collections/... and
    // /algebra/... route, unchanged.
    let legacy_router = Router::new()
        .route("/collections", post(routes::create_collection))
        .route("/collections", get(routes::list_collections))
        .route("/collections/{name}", delete(routes::drop_collection))
        .route("/collections/{name}/write", post(routes::write))
        .route("/collections/{name}/bulk_load", post(routes::bulk_load))
        .route("/collections/{name}/read", post(routes::read))
        .route("/collections/{name}/attention", post(routes::attention))
        .route(
            "/collections/{name}/attention/mdl",
            post(routes::attention_mdl),
        )
        .route(
            "/collections/{name}/attention/calibrate",
            post(routes::calibrate),
        )
        .route("/collections/{name}/stats", get(routes::stats))
        .route("/collections/{name}/compress", post(routes::compress))
        .route("/collections/{name}/config", get(routes::collection_config))
        .route("/collections/{name}/locations", get(routes::locations))
        .route("/collections/{name}/analyze", post(routes::analyze))
        .route(
            "/collections/{name}/batch_analyze",
            post(routes::batch_analyze),
        )
        .route("/collections/{name}/fingerprint", get(routes::fingerprint))
        .route("/collections/{name}/documents", get(routes::get_documents))
        .route(
            "/collections/{name}/documents/query",
            post(routes::query_documents),
        )
        .route(
            "/collections/{name}/documents/{doc_id}",
            get(routes::get_document).delete(routes::delete_document),
        )
        .route("/algebra/add", post(routes::algebra_add))
        .route("/algebra/sub", post(routes::algebra_sub))
        .route("/algebra/scale", post(routes::algebra_scale))
        .route("/algebra/intersect", post(routes::algebra_intersect))
        .route("/algebra/bind", post(routes::algebra_bind))
        .route("/algebra/permute", post(routes::algebra_permute))
        .route("/algebra/unbind", post(routes::algebra_unbind))
        .route("/compose/read", post(routes::compose_read))
        .route("/vec/bind", post(routes::vec_bind))
        .route("/vec/unbind", post(routes::vec_unbind))
        .route("/vec/bundle", post(routes::vec_bundle))
        .route("/vec/pow", post(routes::vec_pow))
        .with_state(default_hive);

    // /db management routes.
    let db_admin_router = Router::new()
        .route("/db", get(routes_db::list_databases))
        .route("/db", post(routes_db::create_database))
        .route("/db/{db}", get(routes_db::get_database))
        .route("/db/{db}", delete(routes_db::drop_database))
        .route("/db/{db}/dream", post(dream::trigger))
        // Access log. Visibility policy (Root-only by default, or a
        // database's own users per `AuditConfig::visibility`) is enforced
        // in `users::is_authorized` / `auth::middleware`, not here, because
        // the middleware is the only place that sees both the scope and
        // the per-database config.
        .route("/db/{db}/audit", get(audit::query_audit));

    // Scoped /db/{db}/collections/... and /db/{db}/algebra/...
    let db_scoped_router = Router::new()
        .route("/db/{db}/collections", post(routes_db::create_collection))
        .route("/db/{db}/collections", get(routes_db::list_collections))
        .route(
            "/db/{db}/collections/{name}",
            delete(routes_db::drop_collection),
        )
        .route("/db/{db}/collections/{name}/write", post(routes_db::write))
        .route(
            "/db/{db}/collections/{name}/bulk_load",
            post(routes_db::bulk_load),
        )
        .route("/db/{db}/collections/{name}/read", post(routes_db::read))
        .route(
            "/db/{db}/collections/{name}/attention",
            post(routes_db::attention),
        )
        .route(
            "/db/{db}/collections/{name}/attention/mdl",
            post(routes_db::attention_mdl),
        )
        .route(
            "/db/{db}/collections/{name}/attention/calibrate",
            post(routes_db::calibrate),
        )
        .route("/db/{db}/collections/{name}/stats", get(routes_db::stats))
        .route(
            "/db/{db}/collections/{name}/compress",
            post(routes_db::compress),
        )
        .route(
            "/db/{db}/collections/{name}/config",
            get(routes_db::collection_config),
        )
        .route(
            "/db/{db}/collections/{name}/locations",
            get(routes_db::locations),
        )
        .route(
            "/db/{db}/collections/{name}/analyze",
            post(routes_db::analyze),
        )
        .route(
            "/db/{db}/collections/{name}/batch_analyze",
            post(routes_db::batch_analyze),
        )
        .route(
            "/db/{db}/collections/{name}/fingerprint",
            get(routes_db::fingerprint),
        )
        .route(
            "/db/{db}/collections/{name}/documents",
            get(routes_db::get_documents),
        )
        .route(
            "/db/{db}/collections/{name}/documents/query",
            post(routes_db::query_documents),
        )
        .route(
            "/db/{db}/collections/{name}/documents/{doc_id}",
            get(routes_db::get_document).delete(routes_db::delete_document),
        )
        .route("/db/{db}/algebra/add", post(routes_db::algebra_add))
        .route("/db/{db}/algebra/sub", post(routes_db::algebra_sub))
        .route("/db/{db}/algebra/scale", post(routes_db::algebra_scale))
        .route(
            "/db/{db}/algebra/intersect",
            post(routes_db::algebra_intersect),
        )
        .route("/db/{db}/algebra/bind", post(routes_db::algebra_bind))
        .route("/db/{db}/algebra/permute", post(routes_db::algebra_permute))
        .route("/db/{db}/algebra/unbind", post(routes_db::algebra_unbind))
        .route("/db/{db}/compose/read", post(routes_db::compose_read));

    // Session-token endpoints: mint (Basic-auth only) + revoke.
    let auth_router = Router::new()
        .route("/auth/token", post(tokens::mint))
        .route("/auth/token/revoke", post(tokens::revoke));

    // Last-activity clock for idle detection, stamped by every request.
    let last_activity = Arc::new(AtomicU64::new(dream::now_ms()));

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/healthz", get(routes::health))
        .route("/ready", get(routes_db::ready))
        .route("/readyz", get(routes_db::ready))
        .merge(legacy_router)
        .merge(db_admin_router)
        .merge(db_scoped_router)
        .merge(auth_router)
        .layer(middleware::from_fn_with_state(auth_state, auth::middleware))
        .layer(Extension(tokens.clone()))
        .layer(middleware::from_fn_with_state(
            last_activity.clone(),
            dream::stamp_activity,
        ))
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

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind error: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    // Idle-time dreaming: reprocess each opted-in database's memory while quiet.
    tokio::spawn(dream::run_dream_loop(server.clone(), last_activity.clone()));

    // Periodic half of the audit flush policy (the other half is the buffer
    // threshold, tripped on the read path itself).
    tokio::spawn(audit::run_flush_loop(server.clone()));

    // Sweep expired session tokens every 5 minutes.
    {
        let tokens = tokens.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(300));
            loop {
                tick.tick().await;
                let swept = tokens.gc();
                if swept > 0 {
                    tracing::debug!(swept, "tokens: gc swept expired");
                }
            }
        });
    }

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        eprintln!("server error: {e}");
        return std::process::ExitCode::FAILURE;
    }

    // Graceful shutdown flushes staged audit records, so the "lose the last
    // few records" trade-off only bites on a hard kill.
    audit::flush_all(&server).await;

    tracing::info!("Server shut down cleanly");
    std::process::ExitCode::SUCCESS
}
