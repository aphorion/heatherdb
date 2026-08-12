use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

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
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl Hive {
    /// Open (or create) a Hive at the given path.
    pub fn open(path: &Path, config: EAMConfig, map_size_mb: usize) -> Result<Self> {
        config.validate()?;
        let store = Arc::new(Store::open(path, map_size_mb)?);

        Ok(Hive {
            store,
            config,
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
