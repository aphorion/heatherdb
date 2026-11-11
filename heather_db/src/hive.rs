use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::audit::{AuditConfig, AuditLog, AuditQuery, AuditRecord};
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
    /// Selective attention shared by every collection, when the database
    /// opted in. See [`crate::attention`].
    attention: Option<Arc<crate::attention::SharedAttention>>,
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
        let store = Arc::new(Store::open_with_precision(
            path,
            map_size_mb,
            config.precision,
        )?);
        let audit = AuditLog::open(&store, audit)?;

        let attention = match &config.selective_attention {
            None => None,
            Some(att) => {
                let mut state: crate::attention::AttentionState = match store
                    .load_metadata(crate::attention::STATE_SCOPE, crate::attention::STATE_KEY)?
                {
                    Some(bytes) => bincode::deserialize(&bytes)?,
                    None => crate::attention::AttentionState::default(),
                };
                if let Some(bytes) = store
                    .load_metadata(crate::attention::STATE_SCOPE, crate::attention::SCALES_KEY)?
                {
                    state.scales = bincode::deserialize(&bytes)?;
                }
                if let Some(sym) = &att.symbols {
                    let records =
                        store.load_metadata_prefix(crate::attention::STATE_SCOPE, "sym/")?;
                    state.symbols = crate::symbols::SymbolTable::from_records(sym, &records)?;
                }
                Some(Arc::new(crate::attention::SharedAttention::new(
                    att.clone(),
                    config.d,
                    state,
                )))
            }
        };

        Ok(Hive {
            store,
            config,
            audit,
            collections: RwLock::new(HashMap::new()),
            attention,
        })
    }

    /// Decodes every stored record of every collection (see
    /// [`crate::verify`]). Reads only.
    pub fn verify(&self) -> Result<Vec<crate::verify::CollectionReport>> {
        let mut collections = self.store.list_collections()?;
        collections.sort();
        collections
            .iter()
            .map(|(name, id)| self.store.verify_collection(name, *id))
            .collect()
    }

    /// Records reads skipped since the database was opened because they did
    /// not decode (see [`Store::tolerate`]).
    pub fn skipped_records(&self) -> u64 {
        self.store.skipped_records()
    }

    /// Whether commits are synced to disk. See [`Store::set_sync`].
    pub fn syncs(&self) -> Result<bool> {
        self.store.syncs()
    }

    /// Turns per-commit syncing off or back on (syncing what was written
    /// meanwhile). See [`Store::set_sync`].
    pub fn set_sync(&self, sync: bool) -> Result<()> {
        self.store.set_sync(sync)
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
            .with_attention(self.attention.clone())
        } else {
            // Exists on disk, load it
            Collection::load(
                collection_id,
                name.to_string(),
                self.store.clone(),
                &self.config,
            )?
            .with_attention(self.attention.clone())
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

        let col = Arc::new(
            Collection::load(
                collection_id,
                name.to_string(),
                self.store.clone(),
                &self.config,
            )?
            .with_attention(self.attention.clone()),
        );
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
    /// The database's selective attention, when it opted in.
    pub fn attention(&self) -> Option<&Arc<crate::attention::SharedAttention>> {
        self.attention.as_ref()
    }

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

    fn attention_config(d: usize) -> EAMConfig {
        let mut config = EAMConfig::new(d).unwrap();
        config.competitive = false;
        config.selective_attention = Some(crate::attention::AttentionConfig {
            time: true,
            k: 50,
            warmup: 16,
            feature_rate: 0.01,
            time_rate: 0.002,
            heads: [("y".to_string(), vec![d - 1])].into_iter().collect(),
            symbols: None,
            columns: None,
        });
        config.validate().unwrap();
        config
    }

    /// Rows whose last field (the outcome) is decided by field 0.
    fn labelled_rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut r: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
                r[d - 1] = if r[0] > 0.0 { 1.0 } else { 0.0 };
                r
            })
            .collect()
    }

    #[test]
    fn attention_is_learned_by_writes_and_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let d = 6;
        let rows = labelled_rows(600, d, 1);
        let learned = {
            let hive = Hive::open(dir.path(), attention_config(d), 256).unwrap();
            let col = hive.get_or_create_collection("customers").unwrap();
            for chunk in rows.chunks(50) {
                col.write_batch(chunk).unwrap();
            }
            let state = hive.attention().unwrap().state.lock().unwrap().clone();
            let head = &state.heads["y"];
            assert!(
                head.updates > 500,
                "every write past warm-up teaches the head"
            );
            let a = head.alpha().unwrap();
            let noise = (1..d - 1).map(|j| a[j]).fold(0.0, f64::max);
            assert!(
                a[0] > noise,
                "field 0 decides the outcome: {} vs {}",
                a[0],
                noise
            );
            assert!(state.time.updates > 0);
            state
        };
        let hive = Hive::open(dir.path(), attention_config(d), 256).unwrap();
        assert_eq!(*hive.attention().unwrap().state.lock().unwrap(), learned);

        let col = hive.get_or_create_collection("customers").unwrap();
        let mut q = labelled_rows(1, d, 99).remove(0);
        let truth = q[d - 1];
        q[d - 1] = 0.5; // ignored: the head fills it in
        let r = col.attention_recall(&q, "y").unwrap();
        assert!((r.result[d - 1] - truth).abs() < 0.5);
        let total: f64 = r.contributors.iter().map(|c| c.weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(col.attention_next().unwrap().is_some());
        assert!(col.attention_recall(&q, "nope").is_err());
    }

    #[test]
    fn a_rejected_batch_leaves_attention_unchanged() {
        let dir = TempDir::new().unwrap();
        let d = 6;
        let hive = Hive::open(dir.path(), attention_config(d), 256).unwrap();
        let col = hive.get_or_create_collection("c").unwrap();
        col.write_batch(&labelled_rows(100, d, 2)).unwrap();
        let before = hive.attention().unwrap().state.lock().unwrap().clone();
        let mut bad = labelled_rows(3, d, 3);
        bad[2].push(0.0); // wrong dimension: the whole batch is refused
        assert!(col.write_batch(&bad).is_err());
        assert_eq!(*hive.attention().unwrap().state.lock().unwrap(), before);
    }

    #[test]
    fn databases_without_attention_learn_nothing() {
        let dir = TempDir::new().unwrap();
        let hive = Hive::open(dir.path(), test_config(), 256).unwrap();
        assert!(hive.attention().is_none());
        let col = hive.get_or_create_collection("c").unwrap();
        let v = vec![0.25; 16];
        col.write_batch(std::slice::from_ref(&v)).unwrap();
        assert!(col.attention_recall(&v, "y").is_err());
    }

    #[test]
    fn attention_needs_a_plain_database() {
        let mut config = attention_config(6);
        config.competitive = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn novelty_ranks_an_outlier_and_rebuilds_after_a_write() {
        use rand::{Rng, SeedableRng};
        let dir = TempDir::new().unwrap();
        let mut config = EAMConfig::new(3).unwrap();
        config.competitive = false;
        let hive = Hive::open(dir.path(), config, 256).unwrap();
        let col = hive.get_or_create_collection("normal").unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let rows: Vec<Vec<f64>> = (0..400)
            .map(|_| (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        col.write_batch(&rows).unwrap();
        let inside = vec![0.1, 0.0, -0.1];
        let outside = vec![6.0, 6.0, 6.0];
        let before = col.novelty(&[inside.clone(), outside.clone()], 20).unwrap();
        assert!(before[0].novelty < 1.5, "inside: {}", before[0].novelty);
        assert!(before[1].novelty > 3.0, "outside: {}", before[1].novelty);
        assert_eq!(before[1].nearest.len(), 20);

        // Normal activity now includes that region: the cache must rebuild.
        let near: Vec<Vec<f64>> = (0..60)
            .map(|_| (0..3).map(|_| 6.0 + rng.gen_range(-0.5..0.5)).collect())
            .collect();
        col.write_batch(&near).unwrap();
        let after = col.novelty(std::slice::from_ref(&outside), 20).unwrap();
        assert!(
            after[0].novelty < 1.5,
            "after normal writes there: {}",
            after[0].novelty
        );
        assert!(col.novelty(std::slice::from_ref(&outside), 0).is_err());
    }

    fn symbol_config(
        d: usize,
        fields: &[&str],
        heads: &[(&str, Vec<usize>)],
        time: bool,
    ) -> EAMConfig {
        let mut config = EAMConfig::new(d).unwrap();
        config.competitive = false;
        config.selective_attention = Some(crate::attention::AttentionConfig {
            time,
            k: 50,
            warmup: 16,
            feature_rate: 0.01,
            time_rate: 0.01,
            heads: heads
                .iter()
                .map(|(n, d)| (n.to_string(), d.clone()))
                .collect(),
            symbols: Some(crate::symbols::SymbolConfig {
                fields: fields.iter().map(|f| f.to_string()).collect(),
                dims: 4,
                rate: 0.05,
                negatives: 5,
            }),
            columns: None,
        });
        config.validate().unwrap();
        config
    }

    /// Rows: a noise number, an outcome slot, and a symbol `colour` with 200
    /// rare values (about 7 rows each) in two hidden groups that decide the
    /// outcome. A recall finds few rows with its own value, so it has to lean
    /// on similar values; random starting codes cannot know which those are,
    /// and the head's error must teach them.
    #[test]
    fn a_head_learns_symbol_codes_and_they_persist() {
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeMap;
        let dir = TempDir::new().unwrap();
        // d = 2 numeric (noise, outcome) + 1 symbol slot.
        let config = symbol_config(3, &["colour"], &[("y", vec![1])], false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let row = |rng: &mut rand::rngs::StdRng| {
            let c = rng.gen_range(0..200u32);
            let y = if c < 100 { 1.0 } else { 0.0 };
            (
                vec![rng.gen_range(-1.0..1.0), y],
                BTreeMap::from([("colour".to_string(), format!("c{c}"))]),
            )
        };
        let learned = {
            let hive = Hive::open(dir.path(), config.clone(), 256).unwrap();
            let col = hive.get_or_create_collection("rows").unwrap();
            for _ in 0..30 {
                let (nums, vals): (Vec<Vec<f64>>, Vec<BTreeMap<String, String>>) =
                    (0..50).map(|_| row(&mut rng)).unzip();
                col.write_batch_symbols(&nums, &vals).unwrap();
            }
            // A raw vector write cannot say which values it means.
            assert!(col.write_batch(&[vec![0.0, 1.0, 0.0]]).is_err());
            // A bad batch adds no value.
            let before = hive
                .attention()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .symbols
                .clone();
            let bad = vec![BTreeMap::from([("nope".to_string(), "x".to_string())])];
            assert!(col.write_batch_symbols(&[vec![0.0, 1.0]], &bad).is_err());
            assert_eq!(
                hive.attention().unwrap().state.lock().unwrap().symbols,
                before
            );

            // Codes learned the hidden groups: the groups separate.
            let st = hive.attention().unwrap().state.lock().unwrap();
            let sym = config
                .selective_attention
                .as_ref()
                .unwrap()
                .symbols
                .as_ref()
                .unwrap();
            let learned_code = |c: u32| {
                st.symbols
                    .code(sym, 0, st.symbols.fields[0].ids[&format!("c{c}")])
                    .to_vec()
            };
            let initial = |c: u32| crate::symbols::initial_code("colour", &format!("c{c}"), 4);
            // Separation: squared gap between the groups' mean codes over the
            // mean squared spread within a group.
            let separation = |code: &dyn Fn(u32) -> Vec<f64>| {
                let mean = |r: std::ops::Range<u32>| {
                    let mut m = vec![0.0; 4];
                    for c in r.clone() {
                        for (a, x) in m.iter_mut().zip(code(c)) {
                            *a += x / 100.0;
                        }
                    }
                    m
                };
                let (ma, mb) = (mean(0..100), mean(100..200));
                let gap: f64 = ma.iter().zip(&mb).map(|(a, b)| (a - b).powi(2)).sum();
                let spread: f64 = (0..200u32)
                    .map(|c| {
                        let m = if c < 100 { &ma } else { &mb };
                        code(c)
                            .iter()
                            .zip(m)
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    / 200.0;
                gap / spread
            };
            let (before, after) = (separation(&initial), separation(&learned_code));
            assert!(
                after > 3.0 * before,
                "separation before {before} after {after}"
            );
            st.clone()
        };
        let hive = Hive::open(dir.path(), config, 256).unwrap();
        assert_eq!(*hive.attention().unwrap().state.lock().unwrap(), learned);
        let col = hive.get_or_create_collection("rows").unwrap();
        let mut right = 0;
        for c in 0..200u32 {
            let vals = BTreeMap::from([("colour".to_string(), format!("c{c}"))]);
            let r = col
                .attention_recall_symbols(&[0.0, 0.5], "y", &vals)
                .unwrap();
            assert_eq!(r.result.len(), 2, "the result is the numeric part");
            if (r.result[1] > 0.5) == (c < 100) {
                right += 1;
            }
        }
        assert!(right >= 190, "{right}/200");
    }

    /// Per-user collections of an `item` symbol where item 2k is followed
    /// by 2k+1: next-value learning, shared across collections, ranks the
    /// follower first.
    #[test]
    fn next_values_rank_the_learned_follower() {
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeMap;
        let dir = TempDir::new().unwrap();
        let config = symbol_config(1, &["item"], &[], true);
        let hive = Hive::open(dir.path(), config, 256).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for u in 0..2000 {
            let col = hive.get_or_create_collection(&format!("user_{u}")).unwrap();
            let mut vals = Vec::new();
            for _ in 0..6 {
                let a = 2 * rng.gen_range(0..20u32);
                for x in [a, a + 1] {
                    vals.push(BTreeMap::from([("item".to_string(), format!("i{x}"))]));
                }
            }
            let nums = vec![Vec::<f64>::new(); vals.len()];
            col.write_batch_symbols(&nums, &vals).unwrap();
        }
        let mut right = 0;
        for k in 0..20u32 {
            let col = hive
                .get_or_create_collection(&format!("probe_{k}"))
                .unwrap();
            for x in [2 * ((k + 7) % 20), 2 * k] {
                let vals = vec![BTreeMap::from([("item".to_string(), format!("i{x}"))])];
                col.write_batch_symbols(&[Vec::<f64>::new()], &vals)
                    .unwrap();
            }
            let top = col.attention_next_values("item", 1, true).unwrap();
            if top[0].0 == format!("i{}", 2 * k + 1) {
                right += 1;
            }
        }
        assert!(right >= 16, "{right}/20");
        assert!(
            hive.get_or_create_collection("user_0")
                .unwrap()
                .attention_next()
                .unwrap()
                .is_none()
        );
    }

    /// A database declared by columns: rows written by name, a category that
    /// decides the outcome, a heavy-tailed number that does not, and the
    /// outcome predicted by name. Scales persist and a bad row changes nothing.
    #[test]
    fn a_column_database_predicts_by_name() {
        use crate::columns::{Cell, Column, Row, plan};
        use rand::{Rng, SeedableRng};
        let dir = TempDir::new().unwrap();
        let cols = vec![
            Column::category("role"),
            Column::number("amount"),
            Column::number("granted"),
        ];
        let (d, att) = plan(cols, &["granted".into()], false).unwrap();
        let mut config = EAMConfig::new(d).unwrap();
        config.competitive = false;
        config.selective_attention = Some(att);
        config.validate().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let row = |rng: &mut rand::rngs::StdRng| -> Row {
            let r = rng.gen_range(0..40u32);
            Row::from([
                ("role".into(), Cell::Text(format!("r{r}"))),
                (
                    "amount".into(),
                    Cell::Number(10f64.powf(rng.gen_range(0.0..6.0))),
                ),
                (
                    "granted".into(),
                    Cell::Number(if r < 20 { 1.0 } else { 0.0 }),
                ),
            ])
        };
        let ask = |r: u32| {
            Row::from([
                ("role".into(), Cell::Text(format!("r{r}"))),
                ("amount".into(), Cell::Number(500.0)),
            ])
        };
        let (scales, probe) = {
            let hive = Hive::open(dir.path(), config.clone(), 256).unwrap();
            let col = hive.get_or_create_collection("requests").unwrap();
            for _ in 0..30 {
                let batch: Vec<Row> = (0..50).map(|_| row(&mut rng)).collect();
                col.write_rows(&batch).unwrap();
            }
            let scales = hive
                .attention()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .scales
                .clone();
            assert!(scales["amount"].scale(1000.0).abs() < 1.0);
            assert!(
                !scales.contains_key("granted"),
                "a predicted column keeps its units"
            );
            let mut bad = row(&mut rng);
            bad.remove("role");
            assert!(col.write_rows(&[row(&mut rng), bad]).is_err());
            assert_eq!(
                hive.attention().unwrap().state.lock().unwrap().scales,
                scales,
                "a failed write leaves the statistics as they were"
            );
            let mut right = 0;
            for r in 0..40u32 {
                let (p, _) = col.predict(&ask(r), "granted").unwrap();
                if (p > 0.5) == (r < 20) {
                    right += 1;
                }
            }
            assert!(right >= 36, "{right}/40");
            assert!(
                col.predict(&ask(0), "role").is_err(),
                "only declared columns"
            );
            assert!(
                col.write_batch(&[vec![0.0; d]]).is_err(),
                "rows by name only"
            );

            // Novelty by row: an ordinary request fits, a strange one stands out.
            let usual = Row::from([
                ("role".into(), Cell::Text("r3".into())),
                ("amount".into(), Cell::Number(300.0)),
            ]);
            let odd = Row::from([
                ("role".into(), Cell::Text("never-seen".into())),
                ("amount".into(), Cell::Number(1e12)),
            ]);
            let nov = col.novelty_rows(&[usual.clone(), odd.clone()], 20).unwrap();
            assert!(nov[0].novelty < 1.5, "usual: {}", nov[0].novelty);
            assert!(
                nov[1].novelty > 2.0 * nov[0].novelty,
                "odd: {}",
                nov[1].novelty
            );
            assert_eq!(nov[0].nearest.len(), 20);
            let mut partial = usual.clone();
            partial.remove("amount");
            assert!(
                col.novelty_rows(&[partial], 20).is_err(),
                "every feature column"
            );
            assert!(
                col.novelty(&[vec![0.0; d]], 20).is_err(),
                "vectors mean nothing here"
            );
            // A write moves what is learned: the cached neighbourhood follows.
            col.write_rows(&[row(&mut rng)]).unwrap();
            let again = col.novelty_rows(&[odd], 20).unwrap();
            assert!(again[0].novelty > 2.0 * nov[0].novelty);
            let now = hive
                .attention()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .scales
                .clone();
            (now, col.predict(&ask(3), "granted").unwrap().0)
        };
        let hive = Hive::open(dir.path(), config, 256).unwrap();
        assert_eq!(
            hive.attention().unwrap().state.lock().unwrap().scales,
            scales
        );
        let col = hive.get_or_create_collection("requests").unwrap();
        assert_eq!(col.predict(&ask(3), "granted").unwrap().0, probe);
    }
}
