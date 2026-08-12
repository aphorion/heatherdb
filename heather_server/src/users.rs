//! User store — LMDB-backed.
//!
//! Lives at `$HEATHER_DATA_DIR/system/data/` (its own LMDB env, sibling
//! of `$ROOT/db/<name>/data/`). One named sub-DB called `users`; key is
//! the username (`Str`), value is bincode of `User`.
//!
//! Why LMDB:
//!   - Live consistency. The CLI mutates the same env the running server
//!     reads from — every `verify()` opens a fresh read transaction so
//!     `heather_server user create` is visible immediately, no restart.
//!   - One persistence layer. `backup`/`snapshot`/`restore` cover users
//!     for free now that they live in an env.
//!   - Multi-process safe by design — LMDB is built for it.
//!
//! No migration: pre-1.0, fresh-start only. Existing `users.json` files
//! are ignored — recreate the user via the CLI on first boot of an
//! upgraded engine.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use heed::types::{Bytes, Str};
use heed::{Database, Env, EnvOpenOptions};
use rand::Rng;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

const SYSTEM_SUBDIR: &str = "system/data";
const SYSTEM_DB_NAME: &str = "users";
/// Modest map size — a system with thousands of users is still tens of MB.
const SYSTEM_MAP_SIZE_MB: usize = 64;

/* ─── public types ────────────────────────────────────────────────────────── */

/// Where a user is allowed to operate.
///
/// Encoded with bincode's default enum layout (discriminant + payload) —
/// the serde-tagged JSON form was dropped when the user store moved into
/// LMDB; nothing reads these bytes by hand any more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scope {
    Root,
    Database(String),
}

impl Scope {
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("root") || s == "*" {
            Ok(Scope::Root)
        } else {
            heather_db::validate_db_name(s).map_err(|e| format!("invalid scope '{s}': {e}"))?;
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

/* ─── store ───────────────────────────────────────────────────────────────── */

/// LMDB-backed user store. Multi-process safe; CLI and server can share one
/// data dir without restart-to-see-changes weirdness.
pub struct UserStore {
    env: Env,
    users: Database<Str, Bytes>,
    /// Where the system env lives. Useful for diagnostics ("which file did
    /// this user store load?").
    env_path: PathBuf,
}

impl UserStore {
    /// Open (or create) the user store under `data_dir`.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        let env_path = data_dir.join(SYSTEM_SUBDIR);
        fs::create_dir_all(&env_path).map_err(|e| format!("create {}: {e}", env_path.display()))?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * SYSTEM_MAP_SIZE_MB)
                .max_dbs(2)
                .open(&env_path)
                .map_err(|e| format!("open env {}: {e}", env_path.display()))?
        };

        let mut wtxn = env.write_txn().map_err(|e| e.to_string())?;
        let users: Database<Str, Bytes> = env
            .create_database(&mut wtxn, Some(SYSTEM_DB_NAME))
            .map_err(|e| format!("create users db: {e}"))?;
        wtxn.commit().map_err(|e| e.to_string())?;

        Ok(Self {
            env,
            users,
            env_path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.env_path
    }

    pub fn is_empty(&self) -> bool {
        match self.env.read_txn() {
            Ok(rtxn) => self.users.is_empty(&rtxn).unwrap_or(true),
            Err(_) => true,
        }
    }

    pub fn list(&self) -> Vec<User> {
        let rtxn = match self.env.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        if let Ok(iter) = self.users.iter(&rtxn) {
            for entry in iter.flatten() {
                if let Ok(u) = bincode::deserialize::<User>(entry.1) {
                    out.push(u);
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
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

        let mut wtxn = self.env.write_txn().map_err(|e| e.to_string())?;
        if self
            .users
            .get(&wtxn, name)
            .map_err(|e| e.to_string())?
            .is_some()
        {
            return Err(format!("user already exists: {name}"));
        }
        self.put_user(&mut wtxn, &user)?;
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(user)
    }

    /// Update a user's password.
    pub fn set_password(&self, name: &str, new_password: &str) -> Result<(), String> {
        if new_password.len() < 8 {
            return Err("password must be at least 8 characters".into());
        }
        let hash = hash_password(new_password)?;

        let mut wtxn = self.env.write_txn().map_err(|e| e.to_string())?;
        let raw = self
            .users
            .get(&wtxn, name)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("user not found: {name}"))?;
        let mut user: User =
            bincode::deserialize(raw).map_err(|e| format!("deserialise user: {e}"))?;
        user.password_hash = hash;
        self.put_user(&mut wtxn, &user)?;
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Delete a user. Returns `true` if a user was actually removed.
    pub fn delete(&self, name: &str) -> Result<bool, String> {
        let mut wtxn = self.env.write_txn().map_err(|e| e.to_string())?;
        let removed = self
            .users
            .delete(&mut wtxn, name)
            .map_err(|e| e.to_string())?;
        wtxn.commit().map_err(|e| e.to_string())?;
        Ok(removed)
    }

    /// Verify a username + password. Always runs an Argon2 verify (against
    /// a decoy hash when the user doesn't exist) so timing doesn't leak
    /// whether the username was valid.
    pub fn verify(&self, name: &str, password: &str) -> Option<User> {
        // Decoy hash so the timing of "user not found" matches "wrong password".
        const DECOY: &str = "$argon2id$v=19$m=19456,t=2,p=1$YWFhYWFhYWFhYWFhYWFhYQ$qJDuO5/0aMQHsfZK2dMNgJsEuXRTyT5xhI8Sx5Z03Ag";

        // Fresh read txn — picks up whatever the CLI wrote since the last call.
        let rtxn = self.env.read_txn().ok()?;
        let (hash_str, found) = match self.users.get(&rtxn, name).ok().flatten() {
            Some(raw) => match bincode::deserialize::<User>(raw) {
                Ok(u) => (u.password_hash.clone(), Some(u)),
                Err(_) => (DECOY.to_string(), None),
            },
            None => (DECOY.to_string(), None),
        };
        drop(rtxn);

        let parsed = PasswordHash::new(&hash_str).ok()?;
        let ok = Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok();
        if ok { found } else { None }
    }

    /* ─── internals ───────────────────────────────────────────────────── */

    fn put_user(&self, wtxn: &mut heed::RwTxn, user: &User) -> Result<(), String> {
        let bytes = bincode::serialize(user).map_err(|e| format!("serialise user: {e}"))?;
        self.users
            .put(wtxn, &user.name, &bytes)
            .map_err(|e| format!("write user: {e}"))?;
        Ok(())
    }
}

/* ─── auth-route classifier (unchanged from JSON era) ─────────────────────── */

/// Authorisation check: does `scope` permit access to a request `path`?
pub fn is_authorized(scope: &Scope, path: &str) -> bool {
    if matches!(scope, Scope::Root) {
        return true;
    }
    let db_name = match scope {
        Scope::Database(n) => n,
        Scope::Root => unreachable!(),
    };

    if let Some(rest) = path.strip_prefix("/db/") {
        let segment_end = rest.find('/').unwrap_or(rest.len());
        let route_db = &rest[..segment_end];
        return route_db == db_name;
    }

    if path == "/db" {
        return false;
    }

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
            return Err(format!(
                "username has invalid char {c:?} (allowed: a-zA-Z0-9 _ - .)"
            ));
        }
    }
    Ok(())
}

/// Generate a URL-safe random password.
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
    fn live_visibility_across_handles() {
        // Simulates the CLI-while-server-runs case: open two handles on
        // the same dir, write through one, see it from the other.
        let dir = tempfile::tempdir().unwrap();
        let server = UserStore::load(dir.path()).unwrap();
        let cli = UserStore::load(dir.path()).unwrap();

        // CLI creates a user. Server (no restart) should see it.
        cli.create("alice", "passw0rd!", Scope::Root).unwrap();
        assert!(server.verify("alice", "passw0rd!").is_some());

        // CLI rotates the password. Server picks up the new one
        // immediately.
        cli.set_password("alice", "new-passw0rd").unwrap();
        assert!(server.verify("alice", "passw0rd!").is_none());
        assert!(server.verify("alice", "new-passw0rd").is_some());

        // CLI deletes. Server stops accepting.
        cli.delete("alice").unwrap();
        assert!(server.verify("alice", "new-passw0rd").is_none());
    }

    #[test]
    fn scope_parse() {
        assert_eq!(Scope::parse("root").unwrap(), Scope::Root);
        assert_eq!(Scope::parse("*").unwrap(), Scope::Root);
        assert_eq!(
            Scope::parse("memoria").unwrap(),
            Scope::Database("memoria".into())
        );
        assert!(Scope::parse("Memoria").is_err());
        assert!(Scope::parse("1abc").is_err());
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
        assert!(!is_authorized(&s, "/db"));
        assert!(!is_authorized(&s, "/collections/x"));
        assert!(!is_authorized(&s, "/algebra/add"));
    }

    #[test]
    fn auth_default_scoped_can_legacy() {
        let s = Scope::Database("default".into());
        assert!(is_authorized(&s, "/db/default"));
        assert!(is_authorized(&s, "/db/default/collections/x"));
        assert!(is_authorized(&s, "/collections/x"));
        assert!(is_authorized(&s, "/algebra/add"));
        assert!(is_authorized(&s, "/compose/read"));
        assert!(!is_authorized(&s, "/db/memoria/collections/x"));
        assert!(!is_authorized(&s, "/db"));
    }
}
