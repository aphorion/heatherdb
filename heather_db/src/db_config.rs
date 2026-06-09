//! Per-database persisted configuration (`db.toml`).
//!
//! Lives at `$ROOT/db/<name>/db.toml`. Created by `Server::create_database`,
//! read at boot. Treated as immutable after first write — to change a
//! database's dimension or EAM knobs, create a new database and migrate.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::EAMConfig;
use crate::error::{HeatherError, Result};

/// Default LMDB map size for a fresh database (4 GiB virtual). LMDB never
/// commits more than what's actually written; the map size is a ceiling.
pub const DEFAULT_MAP_SIZE_MB: usize = 4096;

/// What a database needs to know about itself across restarts.
///
/// Persisted as TOML for human-readability — `cat db.toml` should explain
/// the database without grepping the engine source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    /// Database name (matches the directory name under `$ROOT/db/`).
    pub name: String,

    /// Unix timestamp when the database was created.
    pub created_at: u64,

    /// EAM knobs — the source of truth for `dimension` is `eam.d`.
    pub eam: EAMConfig,

    /// LMDB map size ceiling for this database's env.
    #[serde(default = "default_map_size")]
    pub map_size_mb: usize,
}

fn default_map_size() -> usize {
    DEFAULT_MAP_SIZE_MB
}

impl DbConfig {
    /// Build a fresh `DbConfig` with default EAM knobs and the given dimension.
    pub fn new(name: impl Into<String>, dimension: usize) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            eam: EAMConfig::new(dimension)?,
            map_size_mb: DEFAULT_MAP_SIZE_MB,
        })
    }

    pub fn with_map_size(mut self, mb: usize) -> Self {
        self.map_size_mb = mb;
        self
    }

    pub fn with_eam(mut self, eam: EAMConfig) -> Self {
        self.eam = eam;
        self
    }

    /// Read a `db.toml` from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| HeatherError::Storage(format!("read {}: {e}", path.display())))?;
        toml::from_str(&text)
            .map_err(|e| HeatherError::InvalidConfig(format!("parse {}: {e}", path.display())))
    }

    /// Atomically write a `db.toml`.
    ///
    /// Atomic = write to a sibling tempfile then rename, so a crash mid-write
    /// can't leave a half-baked config that fails to parse on next boot.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self)
            .map_err(|e| HeatherError::Serialization(format!("toml encode: {e}")))?;
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, text.as_bytes())
            .map_err(|e| HeatherError::Storage(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| HeatherError::Storage(format!("rename {}: {e}", path.display())))?;
        Ok(())
    }

    pub fn dimension(&self) -> usize {
        self.eam.d
    }
}

/// Validate a database name. Reserved for path segments + LMDB env names.
///
/// Rules (mirrors PostgreSQL identifier conventions, slightly tighter):
///   - 1..=63 ASCII chars
///   - leading char: a..z
///   - rest: a..z, 0..9, `_`, `-`
///   - must not start with `_` (reserved for system DBs)
pub fn validate_db_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 63 {
        return Err(HeatherError::InvalidInput(format!(
            "database name length must be 1..=63 (got {})",
            name.len()
        )));
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_lowercase() {
        return Err(HeatherError::InvalidInput(
            "database name must start with a lowercase letter".into(),
        ));
    }
    for c in chars {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-') {
            return Err(HeatherError::InvalidInput(format!(
                "database name contains invalid char {c:?} (allowed: a-z 0-9 _ -)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names() {
        assert!(validate_db_name("default").is_ok());
        assert!(validate_db_name("memoria").is_ok());
        assert!(validate_db_name("a").is_ok());
        assert!(validate_db_name("a1_b-c").is_ok());
    }

    #[test]
    fn invalid_names() {
        assert!(validate_db_name("").is_err());
        assert!(validate_db_name("Memoria").is_err()); // uppercase
        assert!(validate_db_name("1memoria").is_err()); // leading digit
        assert!(validate_db_name("_system").is_err()); // leading underscore
        assert!(validate_db_name("foo bar").is_err()); // space
        assert!(validate_db_name("foo.bar").is_err()); // dot
        assert!(validate_db_name(&"a".repeat(64)).is_err()); // too long
    }

    #[test]
    fn config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.toml");
        let cfg = DbConfig::new("memoria", 384).unwrap();
        cfg.save(&path).unwrap();
        let loaded = DbConfig::load(&path).unwrap();
        assert_eq!(loaded.name, "memoria");
        assert_eq!(loaded.dimension(), 384);
        assert_eq!(loaded.map_size_mb, DEFAULT_MAP_SIZE_MB);
    }
}
