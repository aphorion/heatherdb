use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::audit::{AuditConfig, AuditLog, AuditQuery, AuditRecord, UsageSummary};
use crate::collection::{Collection, EAMStats};
use crate::config::EAMConfig;
use crate::error::{HeatherError, Result};
use crate::store::Store;

/// Aggregate stats across all collections in a Hive.
#[derive(Debug, Clone)]
pub struct HiveStats {
    pub num_collections: usize,
    pub collections: Vec<CollectionInfo>,
}

#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub name: String,
    pub stats: EAMStats,
}

/// Top-level manager for multiple EAM collections sharing a single LMDB environment.
pub struct Hive {
    store: Arc<Store>,
    config: EAMConfig,
    audit: AuditLog,
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl Hive {
    /// Open (or create) a Hive at the given path, with default audit settings
    /// (access logging **on** — see [`crate::audit`]).
    pub fn open(path: &Path, config: EAMConfig, map_size_mb: usize) -> Result<Self> {
        Self::open_with_audit(path, config, map_size_mb, AuditConfig::default())
    }

    /// Open (or create) a Hive with explicit audit settings. `Server` passes
    /// the `[audit]` section of the database's `db.toml` here.
    pub fn open_with_audit(
        path: &Path,
        config: EAMConfig,
        map_size_mb: usize,
        audit: AuditConfig,
    ) -> Result<Self> {
        config.validate()?;
        let store = Arc::new(Store::open(path, map_size_mb)?);
        let audit = AuditLog::open(&store, audit)?;

        Ok(Hive {
            store,
            config,
            audit,
            collections: RwLock::new(HashMap::new()),
        })
    }

    /// Create a new collection. Returns it if it already exists.
    pub fn create_collection(&self, name: &str) -> Result<Arc<Collection>> {
        // Check in-memory cache first
        {
            let cache = self
                .collections
                .read()
                .map_err(|_| HeatherError::LockPoisoned)?;
            if let Some(col) = cache.get(name) {
                return Ok(col.clone());
            }
        }

        let mut cache = self
            .collections
            .write()
            .map_err(|_| HeatherError::LockPoisoned)?;

        // Double-check after acquiring write lock
        if let Some(col) = cache.get(name) {
            return Ok(col.clone());
        }

        // Check if it exists on disk but isn't loaded
        let collection_id = {
            let mut txn = self.store.write_txn()?;
            let id = self.store.create_collection(&mut txn, name)?;
            txn.commit()?;
            id
        };

        // Check if it has data on disk (was previously created)
        let existing_locations = self.store.load_all_locations(collection_id)?;

        let col = if existing_locations.is_empty() {
            // Brand new collection
            Collection::new(
                collection_id,
                name.to_string(),
                self.store.clone(),
                &self.config,
            )?
        } else {
            // Exists on disk, load it
            Collection::load(
                collection_id,
                name.to_string(),
                self.store.clone(),
                &self.config,
            )?
        };

        let col = Arc::new(col);
        cache.insert(name.to_string(), col.clone());
        Ok(col)
    }

    /// Get an existing collection. Returns None if it doesn't exist.
    /// Lazily loads from disk if it exists in the registry but not in memory.
    pub fn get_collection(&self, name: &str) -> Result<Option<Arc<Collection>>> {
        // Check in-memory cache
        {
            let cache = self
                .collections
                .read()
                .map_err(|_| HeatherError::LockPoisoned)?;
            if let Some(col) = cache.get(name) {
                return Ok(Some(col.clone()));
            }
        }

        // Check if it exists on disk
        let collection_id = match self.store.get_collection_id(name)? {
            Some(id) => id,
            None => return Ok(None),
        };

        // Lazy load from disk
        let mut cache = self
            .collections
            .write()
            .map_err(|_| HeatherError::LockPoisoned)?;

        // Double-check
        if let Some(col) = cache.get(name) {
            return Ok(Some(col.clone()));
        }

        let col = Arc::new(Collection::load(
            collection_id,
            name.to_string(),
            self.store.clone(),
            &self.config,
        )?);
        cache.insert(name.to_string(), col.clone());
        Ok(Some(col))
    }

    /// Get or create a collection (MongoDB-style auto-create on first access).
    pub fn get_or_create_collection(&self, name: &str) -> Result<Arc<Collection>> {
        // Fast path: check in-memory cache with read lock
        {
            let cache = self
                .collections
                .read()
                .map_err(|_| HeatherError::LockPoisoned)?;
            if let Some(col) = cache.get(name) {
                return Ok(col.clone());
            }
        }

        // Slow path: create or load
        self.create_collection(name)
    }

    /// Drop a collection, removing all its data from disk and memory.
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        let collection_id = match self.store.get_collection_id(name)? {
            Some(id) => id,
            None => return Ok(false),
        };

        // Remove from disk
        let mut txn = self.store.write_txn()?;
        self.store
            .delete_collection(&mut txn, name, collection_id)?;
        txn.commit()?;

        // Remove from memory
        let mut cache = self
            .collections
            .write()
            .map_err(|_| HeatherError::LockPoisoned)?;
        cache.remove(name);

        Ok(true)
    }

    /// List all collection names from the registry.
    pub fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self
            .store
            .list_collections()?
            .into_iter()
            .map(|(name, _)| name)
            .collect())
    }

    /// Get aggregate stats across all loaded collections.
    pub fn stats(&self) -> Result<HiveStats> {
        let names = self.list_collections()?;
        let mut collections = Vec::new();

        for name in &names {
            if let Some(col) = self.get_collection(name)? {
                collections.push(CollectionInfo {
                    name: name.clone(),
                    stats: col.stats()?,
                });
            }
        }

        Ok(HiveStats {
            num_collections: names.len(),
            collections,
        })
    }

    /// Get a reference to the shared config.
    pub fn config(&self) -> &EAMConfig {
        &self.config
    }

    /* ─── access log ──────────────────────────────────────────────────────
     *
     * The audit log is database-scoped, not collection-scoped: dropping a
     * collection deliberately does NOT erase the record of who read it.
     */

    /// The database's access log.
    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    /// Stage one audit record. Returns `true` when the caller should schedule
    /// a flush (the buffer hit its threshold). Never touches disk — see the
    /// [`crate::audit`] module docs for the write-path design.
    pub fn record_audit(&self, rec: AuditRecord) -> bool {
        self.audit.record(rec)
    }

    /// Persist staged audit records and enforce the retention bound.
    /// Blocking: call from a blocking context, not an async reactor thread.
    pub fn flush_audit(&self) -> Result<usize> {
        self.audit.flush(&self.store)
    }

    /// Query the access log, newest-first. Flushes first so records staged in
    /// memory are visible — an audit query that can't see the last second of
    /// activity is worse than a slightly slower audit query.
    pub fn query_audit(&self, q: &AuditQuery) -> Result<Vec<AuditRecord>> {
        self.audit.flush(&self.store)?;
        self.audit.query(&self.store, q)
    }

    /// Drop every audit record older than `cutoff_ms`. The periodic flush
    /// already applies the configured `retention_days`; this is the manual
    /// lever for an operator honouring a shorter erasure request.
    pub fn prune_audit_before(&self, cutoff_ms: u64) -> Result<usize> {
        let mut txn = self.store.write_txn()?;
        let n = self.store.prune_audit(&mut txn, u64::MAX, cutoff_ms)?;
        txn.commit()?;
        Ok(n)
    }

    /// Roll the access log up over a time window for the usage panels.
    pub fn usage_summary(&self, window_secs: u64, limit: usize) -> Result<UsageSummary> {
        self.audit.flush(&self.store)?;
        self.audit.usage(&self.store, window_secs, limit)
    }

    /// Close the underlying environment, blocking until it is released.
    ///
    /// Collections are dropped first because each holds its own `Arc<Store>`;
    /// the environment can only be closed once this Hive owns the last
    /// reference. If a request-scoped clone is still alive elsewhere the
    /// store outlives this call and closing is left to `Drop`.
    pub fn close_and_wait(self) {
        let Hive {
            store, collections, ..
        } = self;
        drop(collections);
        if let Ok(store) = Arc::try_unwrap(store) {
            store.close_and_wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::ReadStrategy;
    use crate::vec_ops;
    use tempfile::TempDir;

    fn test_config() -> EAMConfig {
        let mut config = EAMConfig::new(16).unwrap();
        config.l_0 = 50;
        config.k = 5;
        config
    }

    #[test]
    fn test_create_and_list() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();

        hive.create_collection("users").unwrap();
        hive.create_collection("products").unwrap();

        let mut names = hive.list_collections().unwrap();
        names.sort();
        assert_eq!(names, vec!["products", "users"]);
    }

    #[test]
    fn test_get_or_create() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();

        let col1 = hive.get_or_create_collection("auto").unwrap();
        let col2 = hive.get_or_create_collection("auto").unwrap();

        assert_eq!(col1.collection_id(), col2.collection_id());
    }

    #[test]
    fn test_write_read_across_collections() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();

        let col_a = hive.get_or_create_collection("a").unwrap();
        let col_b = hive.get_or_create_collection("b").unwrap();

        let pattern_a = vec_ops::normalize(&[1.0; 16]);
        let pattern_b = vec_ops::normalize(&[-1.0; 16]);

        for _ in 0..20 {
            col_a.write(&pattern_a).unwrap();
            col_b.write(&pattern_b).unwrap();
        }

        let result_a = col_a.read(&pattern_a, ReadStrategy::HopfieldIter).unwrap();
        let result_b = col_b.read(&pattern_b, ReadStrategy::HopfieldIter).unwrap();

        let sim_a = vec_ops::cosine_similarity(&result_a, &pattern_a);
        let sim_b = vec_ops::cosine_similarity(&result_b, &pattern_b);

        assert!(sim_a > 0.5, "collection A similarity {sim_a} too low");
        assert!(sim_b > 0.5, "collection B similarity {sim_b} too low");
    }

    #[test]
    fn test_drop_collection() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();

        hive.create_collection("doomed").unwrap();
        assert_eq!(hive.list_collections().unwrap().len(), 1);

        let dropped = hive.drop_collection("doomed").unwrap();
        assert!(dropped);
        assert_eq!(hive.list_collections().unwrap().len(), 0);

        // Dropping again returns false
        assert!(!hive.drop_collection("doomed").unwrap());
    }

    #[test]
    fn test_persistence_across_reopen() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let pattern = vec_ops::normalize(&[1.0; 16]);

        // Write data, then drop the Hive
        {
            let hive = Hive::open(dir.path(), config.clone(), 256).unwrap();
            let col = hive.get_or_create_collection("persistent").unwrap();
            for _ in 0..10 {
                col.write(&pattern).unwrap();
            }
            col.flush().unwrap();
        }

        // Reopen and verify data survived
        {
            let hive = Hive::open(dir.path(), config, 256).unwrap();
            let names = hive.list_collections().unwrap();
            assert_eq!(names, vec!["persistent"]);

            let col = hive.get_collection("persistent").unwrap().unwrap();
            let stats = col.stats().unwrap();
            assert!(stats.total_writes > 0.0);
            assert!(stats.num_locations >= 50);
        }
    }

    #[test]
    fn test_hive_stats() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();

        hive.create_collection("a").unwrap();
        hive.create_collection("b").unwrap();

        let stats = hive.stats().unwrap();
        assert_eq!(stats.num_collections, 2);
        assert_eq!(stats.collections.len(), 2);
    }
}
