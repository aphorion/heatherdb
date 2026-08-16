//! Per-database persisted configuration (`db.toml`).
//!
//! Lives at `$ROOT/db/<name>/db.toml`. Created by `Server::create_database`,
//! read at boot. Treated as immutable after first write — to change a
//! database's dimension or EAM knobs, create a new database and migrate.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::audit::AuditConfig;
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

    /// Idle-time consolidation ("dreaming"). Default: disabled — a database
    /// dreams only when an operator opts it in via `db.toml`.
    #[serde(default)]
    pub dream: DreamConfig,

    /// Access logging. Default: **enabled** — unlike `dream`, this is a
    /// security control, so it is on unless an operator explicitly turns it
    /// off. Turning it off silently stops recording who read what; see
    /// `docs/api.md` for the privacy implications of leaving it on.
    #[serde(default)]
    pub audit: AuditConfig,
}

fn default_map_size() -> usize {
    DEFAULT_MAP_SIZE_MB
}

/// Settings for idle-time consolidation ("dreaming"). When `enabled`, the
/// server's dream loop reprocesses a database's collections **in place** while
/// it is idle: it replays each collection's stored attractors back through the
/// EAM's own write rule (so attractors sharpen and novelty/overload splits
/// reorganize), then merges to dedup and enforce capacity. No external input,
/// no second collection — a collection simply wakes up better-organized.
///
/// `passes` > 1 extends this to multi-pass novelty organization: each
/// successive pass operates at a coarser operand granularity (clusters of
/// locations rather than single locations), growing higher-order structure
/// alongside the fine attractors. Pass 0 (replay + merge) is the proven,
/// safe default; coarser passes are the experimental frontier.
///
/// `mode = ladder` switches from in-place reorganization to the consolidation
/// ladder: each named collection of `(situation, procedure)` episodes is
/// consolidated — via the gated two-field write — into a `<name>__L1`
/// collection of `(context prototype, law)`, then `__L1` into `__L2`, and so
/// on for `passes` levels. This is what derives schemas (and laws-about-laws)
/// from raw episodes; it writes new collections rather than reorganizing in
/// place.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Opt-in switch. Off by default.
    #[serde(default)]
    pub enabled: bool,
    /// Collections to dream over. Empty = every collection in the database.
    #[serde(default)]
    pub collections: Vec<String>,
    /// Seconds of inactivity before a consolidation pass may run.
    #[serde(default = "default_idle_secs")]
    pub idle_secs: u64,
    /// Replay mode: granularity passes per dream. Ladder mode: levels to climb.
    #[serde(default = "default_passes")]
    pub passes: usize,
    /// `replay` (in-place reorganization, default) or `ladder` (consolidate
    /// episodes into a stack of derived collections).
    #[serde(default)]
    pub mode: DreamMode,
    /// Ladder mode only: minimum counter-coherence to join an existing family
    /// rather than spawn a new one (the gate threshold).
    #[serde(default = "default_tau_cohere")]
    pub tau_cohere: f64,
    /// Ladder mode only: minimum address similarity for a candidate family
    /// (scopes each rung to its own level of the hierarchy).
    #[serde(default = "default_dream_tau_split")]
    pub tau_split: f64,
}

/// What a dream pass does to a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DreamMode {
    /// Reorganize the collection in place (replay + merge). Default.
    #[default]
    Replay,
    /// Consolidate episodes into a stack of derived collections (the ladder).
    Ladder,
}

fn default_idle_secs() -> u64 {
    60
}

fn default_passes() -> usize {
    1
}

fn default_tau_cohere() -> f64 {
    0.0
}

fn default_dream_tau_split() -> f64 {
    0.0
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            collections: Vec::new(),
            idle_secs: default_idle_secs(),
            passes: default_passes(),
            mode: DreamMode::Replay,
            tau_cohere: default_tau_cohere(),
            tau_split: default_dream_tau_split(),
        }
    }
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
            dream: DreamConfig::default(),
            audit: AuditConfig::default(),
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
