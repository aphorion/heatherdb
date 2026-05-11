//! Optional bearer-token auth.
//!
//! Activated by setting `HEATHER_AUTH_TOKEN` (or `--auth-token`). When set,
//! every request except `/health` must carry `Authorization: Bearer <token>`
//! with an exact match. When unset, the engine runs unauthenticated and
//! emits a one-time warning at boot — fine for a single-user laptop or
//! a Docker network, dangerous on anything internet-facing.
//!
//! Why bearer + simple equality and not JWT/OAuth? The engine is a
//! single-tenant secret right now (token == admin). When real per-DB auth
//! lands (RFC 0001 v0.3) this gets replaced with scoped keys; the
//! middleware shape will stay similar.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::models::ErrorResponse;

/// Auth state shared across handlers via axum's [`State`] extractor.
///
/// Cheap to clone — the token is wrapped in `Arc<str>`. Comparison happens
/// in `O(n)` per request which is fine because tokens are tens of bytes
/// and we don't accept high request rates from unauthenticated clients
/// (we reject before doing any expensive work).
#[derive(Clone)]
pub struct AuthConfig {
    /// `None` → no auth enforcement (warn-only).
    /// `Some(t)` → all routes (except [`unauth_routes`]) require Bearer t.
    pub token: Option<Arc<str>>,
}

impl AuthConfig {
    pub fn disabled() -> Self {
        Self { token: None }
    }
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(Arc::from(token.into())),
        }
    }
    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.token.is_some()
    }
}

/// Routes that should never trigger auth — health/liveness checks need to
/// work for monitoring agents that don't carry credentials.
pub fn is_unauth_route(path: &str) -> bool {
    matches!(path, "/health" | "/healthz" | "/ready" | "/readyz")
}

/// Axum middleware. Plug in via:
///
/// ```ignore
/// .layer(middleware::from_fn_with_state(auth_state.clone(), auth::middleware))
/// ```
pub async fn middleware(
    State(auth): State<AuthConfig>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // No-auth fast path.
    let Some(expected) = auth.token.as_deref() else {
        return next.run(req).await;
    };

    // Health-style routes always pass.
    if is_unauth_route(req.uri().path()) {
        return next.run(req).await;
    }

    // Extract `Authorization: Bearer <token>`.
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim());

    match presented {
        Some(t) if constant_time_eq(t.as_bytes(), expected.as_bytes()) => next.run(req).await,
        Some(_) => deny(StatusCode::FORBIDDEN, "invalid bearer token"),
        None => {
            // 401 with WWW-Authenticate per RFC 6750 §3.
            let mut resp = deny(StatusCode::UNAUTHORIZED, "missing Authorization header");
            resp.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                "Bearer realm=\"heatherdb\"".parse().unwrap(),
            );
            resp
        }
    }
}

fn deny(status: StatusCode, msg: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

/// Constant-time byte equality. Avoids leaking token length / prefix
/// information through the timing side-channel. Cheap; no need for a
/// dedicated crate.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_same_len_match() {
        assert!(constant_time_eq(b"abcdef", b"abcdef"));
    }
    #[test]
    fn ct_eq_diff() {
        assert!(!constant_time_eq(b"abcdef", b"abcdez"));
    }
    #[test]
    fn ct_eq_diff_len() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
    #[test]
    fn unauth_routes() {
        assert!(is_unauth_route("/health"));
        assert!(is_unauth_route("/readyz"));
        assert!(!is_unauth_route("/db"));
        assert!(!is_unauth_route("/collections/foo/write"));
    }
}
