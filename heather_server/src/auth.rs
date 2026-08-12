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

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

use crate::models::ErrorResponse;
use crate::tokens::{AuthKind, Tokens};
use crate::users::{Scope, User, UserStore, is_authorized};

/// What the auth middleware needs at runtime. Cheap to clone (`Arc`s).
#[derive(Clone)]
pub struct AuthState {
    pub users: Arc<UserStore>,
    /// Session-token store. `None` only when auth is disabled.
    pub tokens: Option<Arc<Tokens>>,
    pub disabled: bool,
}

impl AuthState {
    pub fn enabled(users: Arc<UserStore>, tokens: Arc<Tokens>) -> Self {
        Self {
            users,
            tokens: Some(tokens),
            disabled: false,
        }
    }
    pub fn disabled(users: Arc<UserStore>) -> Self {
        Self {
            users,
            tokens: None,
            disabled: true,
        }
    }
}

/// Identity resolved by the middleware and attached to the request. Handlers
/// (the token endpoints) read this to know who is calling and how.
#[derive(Clone)]
pub struct AuthedUser {
    pub user: User,
    pub kind: AuthKind,
    /// The bearer token presented, if `kind == Bearer` — lets a client revoke
    /// the very token it is authenticating with.
    pub token: Option<String>,
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
pub async fn middleware(State(auth): State<AuthState>, req: Request<Body>, next: Next) -> Response {
    let path = req.uri().path().to_string();

    if auth.disabled || is_unauth_route(&path) {
        return next.run(req).await;
    }

    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Bearer (session token) skips the ~11ms Argon2 verify; Basic
    // (user:password) pays it. Either resolves to a `User` + how they authed.
    let (user, kind, token) = match header_val.as_deref() {
        Some(h) if h.starts_with("Bearer ") => {
            let tok = h["Bearer ".len()..].trim().to_string();
            match auth.tokens.as_ref().and_then(|t| t.verify(&tok)) {
                Some((u, _exp)) => (u, AuthKind::Bearer, Some(tok)),
                None => return deny_401("invalid or expired bearer token"),
            }
        }
        Some(h) if h.starts_with("Basic ") => match parse_basic(h) {
            Some(c) => match auth.users.verify(&c.user, &c.password) {
                Some(u) => (u, AuthKind::Basic, None),
                None => return deny_401("invalid credentials"),
            },
            None => return deny_401("malformed Basic credentials"),
        },
        _ => return deny_401("missing or malformed Authorization header"),
    };

    // `/auth/*` are identity operations (mint/revoke the caller's own token) —
    // any authenticated user may use them regardless of data scope. Every
    // other route checks the user's scope against the path.
    if !path.starts_with("/auth/") && !is_authorized(&user.scope, &path) {
        return deny_403(&format!(
            "user '{}' (scope {}) is not authorised for {}",
            user.name,
            user.scope.label(),
            path
        ));
    }

    let mut req = req;
    req.extensions_mut()
        .insert(AuthedUser { user, kind, token });
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
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response();
    resp.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        "Basic realm=\"heatherdb\", charset=\"UTF-8\""
            .parse()
            .unwrap(),
    );
    resp
}

fn deny_403(msg: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

/* ─── boot helpers ────────────────────────────────────────────────────────── */

/// Bootstrap the user store on first boot.
///
/// - If the store is non-empty, this is a no-op.
/// - Otherwise creates one Root-scoped user. The username defaults to
///   `admin` (override with `--admin-user` / `HEATHER_ADMIN_USER`).
/// - The password comes from `--admin-password` / `HEATHER_ADMIN_PASSWORD`
///   when set, otherwise a 24-char random string is generated.
/// - When the password was generated (no env), it's printed once in a
///   fenced ASCII box on stderr — that's the operator's only chance to
///   catch it. When supplied via env, nothing is printed (the operator
///   already has it).
pub fn bootstrap_admin_if_needed(
    users: &UserStore,
    user_name: &str,
    explicit_password: Option<&str>,
) -> Result<(), String> {
    if !users.is_empty() {
        return Ok(());
    }
    let (password, was_generated) = match explicit_password {
        Some(p) if !p.is_empty() => (p.to_string(), false),
        _ => (crate::users::generate_password(24), true),
    };
    users.create(user_name, &password, Scope::Root)?;

    if was_generated {
        eprintln!();
        eprintln!("┌─────────────────────────────────────────────────────────────────────┐");
        eprintln!("│ HeatherDB ⋅ first-boot admin user created.                          │");
        eprintln!("│                                                                     │");
        eprintln!("│   user      {:<55}     │", user_name);
        eprintln!("│   password  {:<55}     │", password);
        eprintln!("│   scope     root                                                    │");
        eprintln!("│                                                                     │");
        eprintln!("│ Save this password — it's NOT printed again.                        │");
        eprintln!("│ Set HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD next time to skip   │");
        eprintln!("│ the random-password dance, or rotate this one with:                 │");
        eprintln!("│   heather_server user passwd <user> --password '<new>'              │");
        eprintln!("└─────────────────────────────────────────────────────────────────────┘");
        eprintln!();
    } else {
        tracing::info!(
            user = user_name,
            "Auth: bootstrapped admin user from HEATHER_ADMIN_USER + HEATHER_ADMIN_PASSWORD"
        );
    }
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
