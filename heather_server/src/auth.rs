//! HTTP Basic Auth + per-database scope.
//!
//! Auth is **on by default**. On a fresh boot the engine creates an
//! `admin` user with a random 24-char password and prints it ONCE to
//! stderr — that's the operator's chance to copy it. Subsequent boots
//! never reprint; rotate via `heather_server user passwd admin`.
//!
//! Every request except `/health` carries `Authorization: Basic
//! base64(user:password)`. The middleware verifies the credentials
//! against the user store and then checks the user's scope against the
//! request path:
//!
//!   - `Scope::Root`               → any route.
//!   - `Scope::Database("default")`→ /db/default/*, /collections/*, /algebra/*, /compose/*
//!   - `Scope::Database(name)`     → /db/{name}/* only.
//!
//! Auth can be disabled for local dev with `HEATHER_AUTH_DISABLED=1`
//! (the engine emits a loud one-time warning at boot in that mode).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;

use crate::models::ErrorResponse;
use crate::users::{is_authorized, Scope, UserStore};

/// What the auth middleware needs at runtime. Cheap to clone (`Arc`s).
#[derive(Clone)]
pub struct AuthState {
    pub users: Arc<UserStore>,
    pub disabled: bool,
}

impl AuthState {
    pub fn enabled(users: Arc<UserStore>) -> Self {
        Self { users, disabled: false }
    }
    pub fn disabled(users: Arc<UserStore>) -> Self {
        Self { users, disabled: true }
    }
}

/// Routes that always pass without a credential check.
pub fn is_unauth_route(path: &str) -> bool {
    matches!(path, "/health" | "/healthz" | "/ready" | "/readyz")
}

/// Axum middleware. Plug in via:
///
/// ```ignore
/// .layer(middleware::from_fn_with_state(auth.clone(), auth::middleware))
/// ```
pub async fn middleware(
    State(auth): State<AuthState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();

    if auth.disabled || is_unauth_route(&path) {
        return next.run(req).await;
    }

    // Parse the Authorization header.
    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    let creds = match header_val.and_then(parse_basic) {
        Some(c) => c,
        None => return deny_401("missing or malformed Authorization header"),
    };

    // Verify the credentials.
    let user = match auth.users.verify(&creds.user, &creds.password) {
        Some(u) => u,
        None => return deny_401("invalid credentials"),
    };

    // Check scope vs route.
    if !is_authorized(&user.scope, &path) {
        return deny_403(&format!(
            "user '{}' (scope {}) is not authorised for {}",
            user.name,
            user.scope.label(),
            path
        ));
    }

    next.run(req).await
}

/* ─── plumbing ────────────────────────────────────────────────────────────── */

struct BasicCreds {
    user: String,
    password: String,
}

fn parse_basic(header_value: &str) -> Option<BasicCreds> {
    let token = header_value.strip_prefix("Basic ")?.trim();
    let raw = B64.decode(token).ok()?;
    let decoded = String::from_utf8(raw).ok()?;
    let (u, p) = decoded.split_once(':')?;
    Some(BasicCreds {
        user: u.to_string(),
        password: p.to_string(),
    })
}

fn deny_401(msg: &str) -> Response {
    let mut resp = (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse { error: msg.to_string() }),
    )
        .into_response();
    resp.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        "Basic realm=\"heatherdb\", charset=\"UTF-8\"".parse().unwrap(),
    );
    resp
}

fn deny_403(msg: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ErrorResponse { error: msg.to_string() }),
    )
        .into_response()
}

/* ─── boot helpers ────────────────────────────────────────────────────────── */

/// Bootstrap the user store. If empty, mint an `admin` user with a random
/// password and print it once to stderr — operators have one chance to
/// catch it.
pub fn bootstrap_admin_if_needed(users: &UserStore) -> Result<(), String> {
    if !users.is_empty() {
        return Ok(());
    }
    let password = crate::users::generate_password(24);
    users.create("admin", &password, Scope::Root)?;

    eprintln!();
    eprintln!("┌─────────────────────────────────────────────────────────────────────┐");
    eprintln!("│ HeatherDB ⋅ first-boot admin user created.                          │");
    eprintln!("│                                                                     │");
    eprintln!("│   user      admin                                                   │");
    eprintln!("│   password  {:<55}     │", password);
    eprintln!("│   scope     root                                                    │");
    eprintln!("│                                                                     │");
    eprintln!("│ Save this password — it's NOT printed again.                        │");
    eprintln!("│ Rotate any time with:                                               │");
    eprintln!("│   heather_server user passwd admin --password '<new>'               │");
    eprintln!("└─────────────────────────────────────────────────────────────────────┘");
    eprintln!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_ok() {
        let c = parse_basic("Basic YWRtaW46aHVudGVyMjI=").unwrap(); // admin:hunter22
        assert_eq!(c.user, "admin");
        assert_eq!(c.password, "hunter22");
    }
    #[test]
    fn parse_basic_with_colon_in_password() {
        // base64("alice:p:a:s:s")
        let c = parse_basic("Basic YWxpY2U6cDphOnM6cw==").unwrap();
        assert_eq!(c.user, "alice");
        assert_eq!(c.password, "p:a:s:s");
    }
    #[test]
    fn parse_basic_rejects_non_basic() {
        assert!(parse_basic("Bearer abc").is_none());
        assert!(parse_basic("Digest x").is_none());
    }
    #[test]
    fn unauth_routes() {
        assert!(is_unauth_route("/health"));
        assert!(!is_unauth_route("/db"));
    }
}
