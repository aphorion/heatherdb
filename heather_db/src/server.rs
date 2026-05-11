//! `Server` — the multi-tenant top of the engine.
//!
//! A `Server` owns N `Arc<Hive>` instances, one per **database**. Each
//! database is a sibling directory under `$ROOT/db/<name>/` with its own
//! `db.toml` and LMDB env. Cross-database isolation is provided by the OS
//! (separate envs, separate file handles) — no key-prefix multiplexing
//! tricks at the engine level.
//!
//! ```no_run
//! use std::path::Path;
//! use heather_db::{Server, DbConfig};
//!
//! // Open or create the server data root.
//! let server = Server::open(Path::new("/var/lib/heatherdb"), 128).unwrap();
//!
//! // Existing collection routes hit `default` by default.
//! let default = server.database("default").unwrap();
//! let col = default.get_or_create_collection("vectors").unwrap();
//!
//! // Create a second database with a different dimension.
//! let memoria_cfg = DbConfig::new("memoria", 384).unwrap();
//! let memoria = server.create_database(memoria_cfg).unwrap();
//! let agents = memoria.get_or_create_collection("agents").unwrap();
//! # let _ = (col, agents);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::config::EAMConfig;
use crate::db_config::{validate_db_name, DbConfig};
use crate::error::{HeatherError, Result};
use crate::hive::Hive;

/// Default database name. Auto-created at first boot if no databases exist.
/// The legacy `/collections/...` HTTP routes are aliases for
/// `/db/default/collections/...`.
pub const DEFAULT_DB: &str = "default";

/// File written at the data-dir root so future engine versions can spot
/// the layout and refuse to mount it with an incompatible reader.
const SERVER_TAG: &str = "server.toml";

/// Multi-database engine root.
///
/// Cheap to clone (`Arc<RwLock<...>>` internally). Pass `Arc<Server>` into
/// HTTP state for axum handlers.
pub struct Server {
    root: PathBuf,
    /// Default dimension to use when lazily creating the default DB on a
    /// fresh boot. Once the default DB exists on disk, this value is
    /// ignored — the persisted `db.toml` wins.
    default_dimension: usize,
    /// `name → Arc<Hive>`. Read-heavy: handlers acquire a read lock,
    /// look up the Arc, and drop the lock before doing any work.
    inner: RwLock<HashMap<String, Arc<Hive>>>,
}

impl Server {
    /// Open (or create) a server rooted at `root`.
    ///
    /// `default_dimension` is only used when creating the default DB on a
    /// fresh boot — a persisted `db.toml` will override it on subsequent
    /// runs. Set this to whatever your "out of the box" workload uses
    /// (commonly 128).
    pub fn open(root: &Path, default_dimension: usize) -> Result<Self> {
        std::fs::create_dir_all(root)
            .map_err(|e| HeatherError::Storage(format!("create root {}: {e}", root.display())))?;

        // Touch the server tag file so future tooling can identify the layout.
        let tag = root.join(SERVER_TAG);
        if !tag.exists() {
            std::fs::write(
                &tag,
                format!(
                    "# heatherdb server data root\nlayout = \"v0.2\"\ncreated_at = {}\n",
                    now_secs()
                ),
            )
            .map_err(|e| HeatherError::Storage(format!("write {}: {e}", tag.display())))?;
        }

        let server = Server {
            root: root.to_path_buf(),
            default_dimension,
            inner: RwLock::new(HashMap::new()),
        };

        // Discover existing databases on disk and pre-open their hives.
        // (Pre-open keeps the first request to each DB fast and surfaces
        // any disk-corruption error at boot rather than on the request path.)
        let db_root = server.db_root();
        std::fs::create_dir_all(&db_root).map_err(|e| {
            HeatherError::Storage(format!("create {}: {e}", db_root.display()))
        })?;

        for entry in std::fs::read_dir(&db_root)
            .map_err(|e| HeatherError::Storage(format!("read {}: {e}", db_root.display())))?
        {
            let entry = entry.map_err(|e| HeatherError::Storage(e.to_string()))?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = match entry.file_name().to_str() {
                Some(s) if !s.starts_with('_') => s.to_string(),
                _ => continue, // skip hidden / system dirs
            };
            // Skip directories that don't actually contain a db.toml — they
            // could be in-progress create_database calls that crashed.
            let cfg_path = entry.path().join("db.toml");
            if !cfg_path.exists() {
                continue;
            }
            let cfg = DbConfig::load(&cfg_path)?;
            let hive = open_hive_for(&server.root, &cfg)?;
            server
                .inner
                .write()
                .map_err(|_| HeatherError::LockPoisoned)?
                .insert(name, Arc::new(hive));
        }

        // If nothing exists yet, lazily create `default` so the legacy
        // /collections/... routes work out of the box.
        let needs_default = server
            .inner
            .read()
            .map_err(|_| HeatherError::LockPoisoned)?
            .is_empty();
        if needs_default {
            let cfg = DbConfig::new(DEFAULT_DB, default_dimension)?;
            server.create_database(cfg)?;
        }

        Ok(server)
    }

    /// Path to the directory holding all per-DB subdirectories.
    fn db_root(&self) -> PathBuf {
        self.root.join("db")
    }

    /// Filesystem path for a given database (whether it exists or not).
    pub fn database_path(&self, name: &str) -> PathBuf {
        self.db_root().join(name)
    }

    /// Names of all open databases.
    pub fn databases(&self) -> Result<Vec<String>> {
        let map = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        let mut out: Vec<String> = map.keys().cloned().collect();
        out.sort();
        Ok(out)
    }

    /// Look up a database by name. Returns `None` if not found.
    pub fn database(&self, name: &str) -> Option<Arc<Hive>> {
        self.inner
            .read()
            .ok()
            .and_then(|map| map.get(name).cloned())
    }

    /// Look up the persisted config for a database.
    pub fn database_config(&self, name: &str) -> Result<DbConfig> {
        let cfg_path = self.database_path(name).join("db.toml");
        if !cfg_path.exists() {
            return Err(HeatherError::InvalidInput(format!(
                "database not found: {name}"
            )));
        }
        DbConfig::load(&cfg_path)
    }

    /// Create a new database. Persists `db.toml` first (the idempotency
    /// anchor) before opening the LMDB env, so a crash mid-create can be
    /// resumed by a re-call with the same name.
    pub fn create_database(&self, cfg: DbConfig) -> Result<Arc<Hive>> {
        validate_db_name(&cfg.name)?;
        cfg.eam.validate()?;

        // Refuse if already in the in-memory map.
        {
            let map = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
            if map.contains_key(&cfg.name) {
                return Err(HeatherError::InvalidInput(format!(
                    "database already exists: {}",
                    cfg.name
                )));
            }
        }

        let dir = self.database_path(&cfg.name);
        let data_dir = dir.join("data");
        std::fs::create_dir_all(&data_dir).map_err(|e| {
            HeatherError::Storage(format!("create {}: {e}", data_dir.display()))
        })?;

        let cfg_path = dir.join("db.toml");
        cfg.save(&cfg_path)?;

        let hive = open_hive_for(&self.root, &cfg)?;
        let arc = Arc::new(hive);

        let mut map = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
        // Re-check under the write lock in case of a concurrent create.
        if map.contains_key(&cfg.name) {
            return Err(HeatherError::InvalidInput(format!(
                "database already exists: {}",
                cfg.name
            )));
        }
        map.insert(cfg.name.clone(), arc.clone());
        Ok(arc)
    }

    /// Drop a database. The directory is moved to `$ROOT/_trash/` rather
    /// than deleted — operators can recover by moving it back. Trash GC
    /// is the operator's call (no auto-cleanup).
    ///
    /// Refuses to drop the default database to avoid the foot-gun where
    /// legacy routes silently start 404'ing.
    pub fn drop_database(&self, name: &str) -> Result<bool> {
        if name == DEFAULT_DB {
            return Err(HeatherError::InvalidInput(
                "cannot drop the default database (rename or restart with empty data dir)".into(),
            ));
        }

        let arc = {
            let mut map = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
            match map.remove(name) {
                Some(a) => a,
                None => return Ok(false),
            }
        };

        // Drop our reference so the LMDB env can close. There may still be
        // request-scoped clones holding the Hive — we don't block on them
        // here; the OS will release the env once the last ref is dropped,
        // which happens at most a few seconds later.
        drop(arc);

        // Move to trash atomically (rename within the same filesystem).
        let dir = self.database_path(name);
        if !dir.exists() {
            return Ok(true);
        }
        let trash_root = self.root.join("_trash");
        std::fs::create_dir_all(&trash_root).map_err(|e| {
            HeatherError::Storage(format!("create {}: {e}", trash_root.display()))
        })?;
        let trashed = trash_root.join(format!("{name}-{}", now_secs()));
        std::fs::rename(&dir, &trashed).map_err(|e| {
            HeatherError::Storage(format!(
                "move {} → {}: {e}",
                dir.display(),
                trashed.display()
            ))
        })?;
        Ok(true)
    }

    /// Default dimension that was used to seed the default DB on this run.
    /// Mostly informational — once `default` exists on disk, the persisted
    /// value wins.
    pub fn default_dimension(&self) -> usize {
        self.default_dimension
    }

    /// Root data directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Open a `Hive` for a given DbConfig at `$root/db/<name>/data/`.
fn open_hive_for(root: &Path, cfg: &DbConfig) -> Result<Hive> {
    let data_dir = root.join("db").join(&cfg.name).join("data");
    std::fs::create_dir_all(&data_dir).map_err(|e| {
        HeatherError::Storage(format!("create {}: {e}", data_dir.display()))
    })?;
    Hive::open(&data_dir, cfg.eam.clone(), cfg.map_size_mb)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Convenience: rebuild a `DbConfig` for an existing legacy default DB
/// from explicit knobs. Useful for ops scripts, not used internally.
pub fn default_db_config(dimension: usize, map_size_mb: usize) -> Result<DbConfig> {
    let mut cfg = DbConfig::new(DEFAULT_DB, dimension)?;
    cfg.eam = EAMConfig::new(dimension)?;
    cfg.map_size_mb = map_size_mb;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_open_creates_default() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        assert_eq!(server.databases().unwrap(), vec![DEFAULT_DB.to_string()]);
        let cfg = server.database_config(DEFAULT_DB).unwrap();
        assert_eq!(cfg.dimension(), 64);
    }

    #[test]
    fn create_and_drop() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        let cfg = DbConfig::new("memoria", 384).unwrap();
        server.create_database(cfg).unwrap();

        let mut names = server.databases().unwrap();
        names.sort();
        assert_eq!(names, vec!["default".to_string(), "memoria".to_string()]);

        let dropped = server.drop_database("memoria").unwrap();
        assert!(dropped);
        assert!(server.database("memoria").is_none());
    }

    #[test]
    fn cannot_drop_default() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        assert!(server.drop_database(DEFAULT_DB).is_err());
    }

    #[test]
    fn restart_recovers_databases() {
        let dir = tempfile::tempdir().unwrap();
        {
            let server = Server::open(dir.path(), 64).unwrap();
            let cfg = DbConfig::new("memoria", 384).unwrap();
            server.create_database(cfg).unwrap();
        }
        let server = Server::open(dir.path(), 64).unwrap();
        let mut names = server.databases().unwrap();
        names.sort();
        assert_eq!(names, vec!["default".to_string(), "memoria".to_string()]);
        assert_eq!(server.database_config("memoria").unwrap().dimension(), 384);
    }

    #[test]
    fn distinct_dimensions_per_database() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        server
            .create_database(DbConfig::new("a128", 128).unwrap())
            .unwrap();
        server
            .create_database(DbConfig::new("a384", 384).unwrap())
            .unwrap();
        assert_eq!(server.database_config("a128").unwrap().dimension(), 128);
        assert_eq!(server.database_config("a384").unwrap().dimension(), 384);
        assert_eq!(server.database_config(DEFAULT_DB).unwrap().dimension(), 64);
    }

    #[test]
    fn duplicate_create_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 64).unwrap();
        server
            .create_database(DbConfig::new("a", 64).unwrap())
            .unwrap();
        // `Hive` doesn't impl Debug, so we can't use `.unwrap_err()` —
        // pattern-match the Result directly.
        match server.create_database(DbConfig::new("a", 128).unwrap()) {
            Err(e) => assert!(format!("{e}").contains("already exists")),
            Ok(_) => panic!("expected duplicate-create to fail"),
        }
    }

    #[test]
    fn cross_db_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::open(dir.path(), 32).unwrap();
        server
            .create_database(DbConfig::new("a", 32).unwrap())
            .unwrap();
        server
            .create_database(DbConfig::new("b", 32).unwrap())
            .unwrap();

        let a = server.database("a").unwrap();
        let b = server.database("b").unwrap();

        // Same collection name, different DBs — should be independent.
        let _ = a.get_or_create_collection("shared").unwrap();
        let cols_a = a.list_collections().unwrap();
        let cols_b = b.list_collections().unwrap();
        assert!(cols_a.contains(&"shared".to_string()));
        assert!(!cols_b.contains(&"shared".to_string()));
    }
}
