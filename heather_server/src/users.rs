//! User store + password hashing + per-database scope.
//!
//! Persisted as `$HEATHER_DATA_DIR/users.json`. JSON because it's tiny
//! (a handful of entries), human-readable, and easy to back up alongside
//! the data dirs. We rewrite the file atomically (write to `.tmp` then
//! rename) so a crash mid-save can't leave the engine without a usable
//! credentials file.
//!
//! Passwords are stored as Argon2id PHC-format hashes. The hash itself
//! encodes the salt and parameters; rotating cost factors later is just
//! a matter of re-hashing on next login.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use rand::rngs::OsRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

const USERS_FILE: &str = "users.json";

/// Where a user is allowed to operate.
///
/// `Root` carries every permission — DB management (`POST /db`,
/// `DELETE /db/{name}`), every scoped route, and the legacy
/// `/collections/*` aliases.
///
/// `Database(name)` carries only the routes that target that name —
/// `/db/{name}/...` plus the legacy aliases iff `name == "default"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum Scope {
    Root,
    Database(String),
}

impl Scope {
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("root") || s == "*" {
            Ok(Scope::Root)
        } else {
            // Reuse the engine's database-name validator so users can't be
            // scoped to invalid names. Mirrors the same rules the engine
            // would refuse at create time.
            heather_db::validate_db_name(s)
                .map_err(|e| format!("invalid scope '{s}': {e}"))?;
            Ok(Scope::Database(s.to_string()))
        }
    }

    pub fn label(&self) -> String {
        match self {
            Scope::Root => "root".into(),
            Scope::Database(n) => format!("db:{n}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub name: String,
    pub password_hash: String, // Argon2id PHC string
    pub scope: Scope,
    pub created_at: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct UsersFile {
    #[serde(default)]
    users: Vec<User>,
}

/// In-memory user database. Backed by a JSON file on disk; reloads can be
/// triggered by `reload_from_disk()` if an out-of-band edit happens.
pub struct UserStore {
    path: PathBuf,
    inner: RwLock<UsersFile>,
}

impl UserStore {
    /// Load (or create) the users file under `data_dir`.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        let path = data_dir.join(USERS_FILE);
        let inner = if path.exists() {
            let raw = fs::read_to_string(&path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            serde_json::from_str::<UsersFile>(&raw)
                .map_err(|e| format!("parse {}: {e}", path.display()))?
        } else {
            UsersFile::default()
        };
        Ok(Self {
            path,
            inner: RwLock::new(inner),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().map(|f| f.users.is_empty()).unwrap_or(true)
    }

    pub fn list(&self) -> Vec<User> {
        self.inner
            .read()
            .map(|f| f.users.clone())
            .unwrap_or_default()
    }

    /// Create a new user. Returns an error if the name already exists.
    pub fn create(&self, name: &str, password: &str, scope: Scope) -> Result<User, String> {
        validate_username(name)?;
        if password.len() < 8 {
            return Err("password must be at least 8 characters".into());
        }
        let hash = hash_password(password)?;
        let user = User {
            name: name.to_string(),
            password_hash: hash,
            scope,
            created_at: now_secs(),
        };

        let mut file = self.inner.write().map_err(|_| "user store lock poisoned")?;
        if file.users.iter().any(|u| u.name == name) {
            return Err(format!("user already exists: {name}"));
        }
        file.users.push(user.clone());
        Self::save(&self.path, &file)?;
        Ok(user)
    }

    /// Update a user's password.
    pub fn set_password(&self, name: &str, new_password: &str) -> Result<(), String> {
        if new_password.len() < 8 {
            return Err("password must be at least 8 characters".into());
        }
        let hash = hash_password(new_password)?;
        let mut file = self.inner.write().map_err(|_| "user store lock poisoned")?;
        let u = file
            .users
            .iter_mut()
            .find(|u| u.name == name)
            .ok_or_else(|| format!("user not found: {name}"))?;
        u.password_hash = hash;
        Self::save(&self.path, &file)?;
        Ok(())
    }

    /// Delete a user. Returns `true` if a user was actually removed.
    pub fn delete(&self, name: &str) -> Result<bool, String> {
        let mut file = self.inner.write().map_err(|_| "user store lock poisoned")?;
        let before = file.users.len();
        file.users.retain(|u| u.name != name);
        let removed = file.users.len() < before;
        if removed {
            Self::save(&self.path, &file)?;
        }
        Ok(removed)
    }

    /// Verify a username + password pair against the store.
    /// Returns the user (with its scope) on success, `None` on any failure.
    /// Constant-time-ish — always runs an argon2 verify, even when the
    /// user doesn't exist, to avoid a username-enumeration timing oracle.
    pub fn verify(&self, name: &str, password: &str) -> Option<User> {
        let users = self.inner.read().ok()?;
        // Decoy hash so the timing of "user not found" matches "wrong password".
        const DECOY: &str = "$argon2id$v=19$m=19456,t=2,p=1$YWFhYWFhYWFhYWFhYWFhYQ$qJDuO5/0aMQHsfZK2dMNgJsEuXRTyT5xhI8Sx5Z03Ag";
        let (hash_str, found) = match users.users.iter().find(|u| u.name == name) {
            Some(u) => (u.password_hash.as_str(), Some(u.clone())),
            None => (DECOY, None),
        };
        let parsed = PasswordHash::new(hash_str).ok()?;
        let ok = Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok();
        if ok { found } else { None }
    }

    /// Re-read the file from disk. Useful after the CLI has mutated it
    /// while the server was running.
    #[allow(dead_code)]
    pub fn reload_from_disk(&self) -> Result<(), String> {
        let raw = fs::read_to_string(&self.path)
            .map_err(|e| format!("read {}: {e}", self.path.display()))?;
        let next: UsersFile = serde_json::from_str(&raw)
            .map_err(|e| format!("parse {}: {e}", self.path.display()))?;
        let mut file = self.inner.write().map_err(|_| "user store lock poisoned")?;
        *file = next;
        Ok(())
    }

    fn save(path: &Path, file: &UsersFile) -> Result<(), String> {
        let serialised = serde_json::to_string_pretty(file)
            .map_err(|e| format!("serialise users: {e}"))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serialised.as_bytes())
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        // Restrict to owner-readable (best-effort on Unix).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
        }
        fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        Ok(())
    }
}

/// Authorisation check: does `scope` permit access to a request `path`?
///
/// Rules:
///   - `Scope::Root`                  → every path.
///   - `Scope::Database("default")`   → /db/default/* AND /collections/*
///                                       AND /algebra/* AND /compose/*
///   - `Scope::Database(name)`        → /db/{name}/*  only
///   - `/db` (DB management)          → root only
///   - `/health`                      → not checked here (whitelisted upstream)
pub fn is_authorized(scope: &Scope, path: &str) -> bool {
    if matches!(scope, Scope::Root) {
        return true;
    }
    let db_name = match scope {
        Scope::Database(n) => n,
        Scope::Root => unreachable!(),
    };

    // /db/{name}/...
    if let Some(rest) = path.strip_prefix("/db/") {
        // The first segment up to the next '/' (or end) is the DB name.
        let segment_end = rest.find('/').unwrap_or(rest.len());
        let route_db = &rest[..segment_end];
        return route_db == db_name;
    }

    // /db itself (list + create) — root only.
    if path == "/db" {
        return false;
    }

    // Legacy default-DB routes — only if scoped to "default".
    if db_name == "default" {
        return path.starts_with("/collections")
            || path.starts_with("/algebra")
            || path.starts_with("/compose");
    }

    false
}

/* ─── helpers ─────────────────────────────────────────────────────────────── */

fn hash_password(password: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("argon2 hash: {e}"))
}

pub fn validate_username(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err("username must be 1..=64 chars".into());
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() {
        return Err("username must start with a letter".into());
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') {
            return Err(format!("username has invalid char {c:?} (allowed: a-zA-Z0-9 _ - .)"));
        }
    }
    Ok(())
}

/// Generate a URL-safe random password. Used when first-boot creates the
/// admin user — printed once to stderr in the engine logs and never stored
/// in plaintext.
pub fn generate_password(len: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789";
    let mut rng = OsRng;
    (0..len)
        .map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char)
        .collect()
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

    fn store() -> (tempfile::TempDir, UserStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = UserStore::load(dir.path()).unwrap();
        (dir, s)
    }

    #[test]
    fn create_and_verify() {
        let (_dir, s) = store();
        s.create("alice", "hunter22!", Scope::Root).unwrap();
        assert!(s.verify("alice", "hunter22!").is_some());
        assert!(s.verify("alice", "wrong").is_none());
        assert!(s.verify("nope", "anything").is_none());
    }

    #[test]
    fn duplicate_create_rejected() {
        let (_dir, s) = store();
        s.create("alice", "passw0rd!", Scope::Root).unwrap();
        assert!(s.create("alice", "passw0rd!", Scope::Root).is_err());
    }

    #[test]
    fn change_password() {
        let (_dir, s) = store();
        s.create("alice", "passw0rd!", Scope::Root).unwrap();
        s.set_password("alice", "new-passw0rd").unwrap();
        assert!(s.verify("alice", "passw0rd!").is_none());
        assert!(s.verify("alice", "new-passw0rd").is_some());
    }

    #[test]
    fn delete_user() {
        let (_dir, s) = store();
        s.create("alice", "passw0rd!", Scope::Root).unwrap();
        assert!(s.delete("alice").unwrap());
        assert!(!s.delete("alice").unwrap());
        assert!(s.verify("alice", "passw0rd!").is_none());
    }

    #[test]
    fn scope_parse() {
        assert_eq!(Scope::parse("root").unwrap(), Scope::Root);
        assert_eq!(Scope::parse("*").unwrap(), Scope::Root);
        assert_eq!(
            Scope::parse("memoria").unwrap(),
            Scope::Database("memoria".into())
        );
        assert!(Scope::parse("Memoria").is_err()); // uppercase
        assert!(Scope::parse("1abc").is_err()); // leading digit
    }

    #[test]
    fn auth_root_can_anything() {
        let s = Scope::Root;
        assert!(is_authorized(&s, "/db"));
        assert!(is_authorized(&s, "/db/anything"));
        assert!(is_authorized(&s, "/collections/foo/write"));
        assert!(is_authorized(&s, "/algebra/add"));
    }

    #[test]
    fn auth_db_scoped() {
        let s = Scope::Database("memoria".into());
        assert!(is_authorized(&s, "/db/memoria"));
        assert!(is_authorized(&s, "/db/memoria/collections/users/write"));
        assert!(!is_authorized(&s, "/db/navigator/collections/x"));
        assert!(!is_authorized(&s, "/db"));               // mgmt
        assert!(!is_authorized(&s, "/collections/x"));   // legacy = default
        assert!(!is_authorized(&s, "/algebra/add"));
    }

    #[test]
    fn auth_default_scoped_can_legacy() {
        let s = Scope::Database("default".into());
        assert!(is_authorized(&s, "/db/default"));
        assert!(is_authorized(&s, "/db/default/collections/x"));
        assert!(is_authorized(&s, "/collections/x"));        // legacy alias
        assert!(is_authorized(&s, "/algebra/add"));
        assert!(is_authorized(&s, "/compose/read"));
        assert!(!is_authorized(&s, "/db/memoria/collections/x"));
        assert!(!is_authorized(&s, "/db"));
    }

    #[test]
    fn timing_safe_unknown_user() {
        // Both calls should take roughly the same amount of work — hard to
        // assert timing in unit tests, but we can at least confirm both
        // return None deterministically.
        let (_dir, s) = store();
        s.create("alice", "passw0rd!", Scope::Root).unwrap();
        assert!(s.verify("alice", "wrong").is_none());
        assert!(s.verify("nonexistent", "wrong").is_none());
    }
}
