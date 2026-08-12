//! Session tokens — LMDB-backed.
//!
//! Bearer tokens let a client pay the Argon2id cost once (on `POST
//! /auth/token` with Basic auth) and then make many follow-up requests
//! cheaply. The driving use case is bulk forward-pass benchmarks that
//! issue thousands of `/read` calls in a row — the ~11ms Argon2 verify
//! per request dominates over the ~2ms engine compute.
//!
//! Storage lives in `$HEATHER_DATA_DIR/system/data/` (the same env as the
//! user store) in its own `tokens` sub-DB. Tokens survive restarts;
//! `Tokens::load` sweeps anything already expired so a long-stopped
//! engine doesn't come back up with a junk drawer full of dead entries.
//!
//! Tokens are 32 bytes of `OsRng` → base64url no-pad → ~43-char opaque
//! string. They're indistinguishable from random; treat them like
//! passwords (don't log them, don't share them across clients).
//!
//! A token's scope is whatever the minting user had at mint time. If
//! the user's scope is later changed in the user store, **already-issued
//! tokens keep the old scope** until they expire or are revoked.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::auth::AuthedUser;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use heed::types::{Bytes, Str};
use heed::{Database, Env, EnvOpenOptions};
use rand::RngCore;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::users::User;

const SYSTEM_SUBDIR: &str = "system/data";
const TOKEN_DB_NAME: &str = "tokens";
/// Same env as the user store, just a different sub-DB. We bump the
/// max_dbs count to 2 here so opening this after `UserStore::load`
/// doesn't trip the LMDB limit. UserStore already opens with max_dbs=2,
/// so 2 is also the ceiling there — keep them in sync if you add more.
const SYSTEM_MAP_SIZE_MB: usize = 64;

/// How a request authenticated. Attached to request extensions by the
/// auth middleware alongside the resolved `User`. Used by the token-mint
/// endpoint to refuse Bearer-authenticated requests (otherwise tokens
/// could extend their own lifetime indefinitely without paying Argon2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthKind {
    Basic,
    Bearer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenEntry {
    user: User,
    expires_at: u64, // unix seconds
}

/// LMDB-backed session token store with an in-memory cache for the
/// hot path (`verify`). The cache is loaded at startup and kept in
/// sync with every mutating call; it's the source of truth for
/// expiry checks. LMDB is the source of truth across restarts.
///
/// Why a cache: the whole point of session tokens is to skip the ~11ms
/// Argon2 verify on the hot path. Going to LMDB would still be fast
/// (~µs) but a `HashMap` read under an `RwLock` is faster still and
/// avoids opening a read txn per request.
pub struct Tokens {
    env: Env,
    db: Database<Str, Bytes>,
    cache: RwLock<std::collections::HashMap<String, TokenEntry>>,
    #[allow(dead_code)]
    env_path: PathBuf,
}

impl Tokens {
    /// Open (or create) the token store under `data_dir`. Sweeps any
    /// already-expired entries so the live cache is clean from the
    /// start.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        let env_path = data_dir.join(SYSTEM_SUBDIR);
        fs::create_dir_all(&env_path).map_err(|e| format!("create {}: {e}", env_path.display()))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * SYSTEM_MAP_SIZE_MB)
                // 2 sub-DBs in this env: `users` (UserStore) + `tokens`
                // (this struct). Bump if you add a third.
                .max_dbs(2)
                .open(&env_path)
                .map_err(|e| format!("open env {}: {e}", env_path.display()))?
        };

        let mut wtxn = env.write_txn().map_err(|e| e.to_string())?;
        let db: Database<Str, Bytes> = env
            .create_database(&mut wtxn, Some(TOKEN_DB_NAME))
            .map_err(|e| format!("create tokens db: {e}"))?;
        wtxn.commit().map_err(|e| e.to_string())?;

        let mut cache = std::collections::HashMap::new();
        let now = now_secs();
        // Load live entries; mark expired ones for deletion in the same pass.
        let mut to_delete: Vec<String> = Vec::new();
        {
            let rtxn = env.read_txn().map_err(|e| e.to_string())?;
            if let Ok(iter) = db.iter(&rtxn) {
                for entry in iter.flatten() {
                    let (k, v) = entry;
                    match bincode::deserialize::<TokenEntry>(v) {
                        Ok(te) if te.expires_at > now => {
                            cache.insert(k.to_string(), te);
                        }
                        _ => to_delete.push(k.to_string()),
                    }
                }
            }
        }
        if !to_delete.is_empty() {
            let mut wtxn = env.write_txn().map_err(|e| e.to_string())?;
            for k in &to_delete {
                let _ = db.delete(&mut wtxn, k);
            }
            wtxn.commit().map_err(|e| e.to_string())?;
        }

        Ok(Self {
            env,
            db,
            cache: RwLock::new(cache),
            env_path,
        })
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.env_path
    }

    /// Mint a fresh token for `user`, valid for `ttl`. Returns the
    /// opaque token string and the absolute expiry (unix-seconds).
    pub fn mint(&self, user: User, ttl: Duration) -> (String, u64) {
        let token = random_token();
        let expires_at = now_secs().saturating_add(ttl.as_secs());
        let entry = TokenEntry { user, expires_at };

        // Persist first; only insert into the cache after the write
        // commits so a process crash doesn't leave the cache claiming
        // tokens that are gone on disk.
        if let Err(e) = self.persist(&token, &entry) {
            // Persistence failure is rare (disk full, fs gone). Log and
            // still hand the token back from the cache — a restart will
            // invalidate it, which is the conservative behaviour.
            tracing::warn!(error = %e, "tokens: failed to persist mint; cache-only");
        }
        self.cache
            .write()
            .expect("tokens cache poisoned")
            .insert(token.clone(), entry);
        (token, expires_at)
    }

    /// Look up a token. Returns `(user, expires_at)` if the token is
    /// known AND not expired. If the token is expired this lazily
    /// removes it from the cache and from LMDB.
    pub fn verify(&self, token: &str) -> Option<(User, u64)> {
        // Fast path: read lock the cache.
        {
            let guard = self.cache.read().expect("tokens cache poisoned");
            // Unknown token: nothing to sweep, so leave immediately. A known
            // but expired one falls through to the write-lock sweep below.
            let entry = guard.get(token)?;
            if entry.expires_at > now_secs() {
                return Some((entry.user.clone(), entry.expires_at));
            }
        }
        // Expired — sweep cache + LMDB.
        {
            let mut guard = self.cache.write().expect("tokens cache poisoned");
            if let Some(entry) = guard.get(token)
                && entry.expires_at <= now_secs()
            {
                guard.remove(token);
            }
        }
        let _ = self.remove_persisted(token);
        None
    }

    /// Revoke a single token. Returns `true` if the token existed in
    /// the cache.
    pub fn revoke(&self, token: &str) -> bool {
        let existed = self
            .cache
            .write()
            .expect("tokens cache poisoned")
            .remove(token)
            .is_some();
        let _ = self.remove_persisted(token);
        existed
    }

    /// Sweep expired entries from cache + LMDB. Returns the number of
    /// entries removed. Called at boot and on a 5-minute timer.
    pub fn gc(&self) -> usize {
        let now = now_secs();
        let expired: Vec<String> = {
            let guard = self.cache.read().expect("tokens cache poisoned");
            guard
                .iter()
                .filter(|(_, e)| e.expires_at <= now)
                .map(|(k, _)| k.clone())
                .collect()
        };
        if expired.is_empty() {
            return 0;
        }
        {
            let mut guard = self.cache.write().expect("tokens cache poisoned");
            for k in &expired {
                guard.remove(k);
            }
        }
        if let Ok(mut wtxn) = self.env.write_txn() {
            for k in &expired {
                let _ = self.db.delete(&mut wtxn, k);
            }
            let _ = wtxn.commit();
        }
        expired.len()
    }

    /// Test/diagnostic only — number of live entries in the cache.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.cache.read().expect("tokens cache poisoned").len()
    }

    /* ─── internals ─────────────────────────────────────────────── */

    fn persist(&self, token: &str, entry: &TokenEntry) -> Result<(), String> {
        let bytes = bincode::serialize(entry).map_err(|e| format!("serialise token: {e}"))?;
        let mut wtxn = self.env.write_txn().map_err(|e| e.to_string())?;
        self.db
            .put(&mut wtxn, token, &bytes)
            .map_err(|e| format!("write token: {e}"))?;
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    fn remove_persisted(&self, token: &str) -> Result<(), String> {
        let mut wtxn = self.env.write_txn().map_err(|e| e.to_string())?;
        let _ = self.db.delete(&mut wtxn, token);
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Default session-token lifetime: 1 hour.
const TOKEN_TTL: Duration = Duration::from_secs(3600);

/// `POST /auth/token` — mint a session token for the caller. Requires Basic
/// auth: a Bearer-authenticated caller is refused, so a token can't extend its
/// own lifetime without paying Argon2. The token inherits the user's scope at
/// mint time.
pub async fn mint(
    Extension(tokens): Extension<Arc<Tokens>>,
    Extension(authed): Extension<AuthedUser>,
) -> Response {
    if authed.kind == AuthKind::Bearer {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "mint requires Basic auth, not a bearer token"
            })),
        )
            .into_response();
    }
    let (token, expires_at) = tokens.mint(authed.user.clone(), TOKEN_TTL);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "token": token,
            "token_type": "Bearer",
            "expires_at": expires_at,
        })),
    )
        .into_response()
}

/// `POST /auth/token/revoke` — revoke the bearer token used for this request.
/// Basic-authenticated callers have no token to revoke (400).
pub async fn revoke(
    Extension(tokens): Extension<Arc<Tokens>>,
    Extension(authed): Extension<AuthedUser>,
) -> Response {
    match authed.token {
        Some(tok) => {
            let revoked = tokens.revoke(&tok);
            (
                StatusCode::OK,
                Json(serde_json::json!({ "revoked": revoked })),
            )
                .into_response()
        }
        None => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "no bearer token on this request to revoke"
            })),
        )
            .into_response(),
    }
}

fn random_token() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::users::{Scope, UserStore};

    fn store_pair() -> (tempfile::TempDir, Tokens) {
        let dir = tempfile::tempdir().unwrap();
        // Tokens shares the env with UserStore. Loading the user store
        // first creates `system/data/` + the `users` sub-DB; Tokens
        // then adds its own `tokens` sub-DB.

        let _us = UserStore::load(dir.path()).unwrap();
        let t = Tokens::load(dir.path()).unwrap();
        (dir, t)
    }

    fn user(name: &str) -> User {
        User {
            name: name.to_string(),
            password_hash: "x".into(),
            scope: Scope::Root,
            created_at: 0,
        }
    }

    #[test]
    fn mint_and_verify_token() {
        let (_dir, t) = store_pair();
        let (tok, exp) = t.mint(user("alice"), Duration::from_secs(60));
        let (got, exp2) = t.verify(&tok).expect("token should verify");
        assert_eq!(got.name, "alice");
        assert_eq!(exp, exp2);
        // Token is ~43 chars of base64url alphabet.
        assert!(tok.len() >= 40 && tok.len() <= 44, "len was {}", tok.len());
    }

    #[test]
    fn expired_token_rejected() {
        let (_dir, t) = store_pair();
        let (tok, _) = t.mint(user("alice"), Duration::from_secs(0));
        assert!(t.verify(&tok).is_none());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn revoke_invalidates_token() {
        let (_dir, t) = store_pair();
        let (tok, _) = t.mint(user("alice"), Duration::from_secs(60));
        assert!(t.verify(&tok).is_some());
        assert!(t.revoke(&tok));
        assert!(t.verify(&tok).is_none());
        // Revoking a missing token is a no-op (returns false because
        // the cache didn't have it).
        assert!(!t.revoke(&tok));
    }

    #[test]
    fn tokens_isolated_by_string() {
        let (_dir, t) = store_pair();
        let (a, _) = t.mint(user("alice"), Duration::from_secs(60));
        let (b, _) = t.mint(user("alice"), Duration::from_secs(60));
        assert_ne!(a, b);
        assert!(t.revoke(&a));
        assert!(t.verify(&a).is_none());
        assert!(t.verify(&b).is_some());
    }

    #[test]
    fn gc_sweeps_expired() {
        let (_dir, t) = store_pair();
        let (_a, _) = t.mint(user("alice"), Duration::from_secs(0));
        let (b, _) = t.mint(user("bob"), Duration::from_secs(60));
        assert_eq!(t.len(), 2);
        let removed = t.gc();
        assert_eq!(removed, 1);
        assert_eq!(t.len(), 1);
        assert!(t.verify(&b).is_some());
    }

    #[test]
    fn tokens_persist_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let _us = UserStore::load(dir.path()).unwrap();
        let (tok, exp) = {
            let t1 = Tokens::load(dir.path()).unwrap();
            t1.mint(user("alice"), Duration::from_secs(3600))
        };
        // Simulate restart: drop t1, open a fresh handle.
        let t2 = Tokens::load(dir.path()).unwrap();
        let (got, exp2) = t2.verify(&tok).expect("token should survive restart");
        assert_eq!(got.name, "alice");
        assert_eq!(exp, exp2);
    }

    #[test]
    fn expired_tokens_swept_at_boot() {
        let dir = tempfile::tempdir().unwrap();
        let _us = UserStore::load(dir.path()).unwrap();
        let dead_tok = {
            let t1 = Tokens::load(dir.path()).unwrap();
            let (tok, _) = t1.mint(user("alice"), Duration::from_secs(0));
            tok
        };
        // Fresh handle — the load() call should have removed the expired entry.
        let t2 = Tokens::load(dir.path()).unwrap();
        assert_eq!(t2.len(), 0);
        assert!(t2.verify(&dead_tok).is_none());
    }
}
