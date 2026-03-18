use std::sync::{Arc, RwLock};

use crate::config::EAMConfig;
use crate::error::{HeatherError, Result};
use crate::location::{HardLocation, LocationId};
use crate::merge;
use crate::read;
pub use crate::read::{ActivatedLocation, ReadTrace};
use crate::store::Store;
use crate::vec_ops;
use crate::write;

/// Statistics about the current state of an EAM collection.
#[derive(Debug, Clone)]
pub struct EAMStats {
    pub num_locations: usize,
    pub total_writes: f64,
    pub current_eta: f64,
    pub avg_write_count: f64,
    pub max_write_count: f64,
}

/// Summary of a single hard location for UI display.
#[derive(Debug, Clone)]
pub struct LocationSummary {
    pub id: usize,
    pub write_count: f64,
    pub avg_counter_magnitude: f64,
}

/// Read strategy selection.
#[derive(Debug, Clone, Copy)]
pub enum ReadStrategy {
    /// Hopfield iterative (best quality, default)
    HopfieldIter,
    /// Hopfield single-step (faster, lower quality)
    HopfieldSS,
}

pub(crate) struct EAMInner {
    pub(crate) config: EAMConfig,
    pub(crate) locations: Vec<HardLocation>,
    pub(crate) next_id: u64,
    pub(crate) eta: f64,
    pub(crate) doc_next_id: u64,
    /// Cached landmark indices for graph search entry points
    pub(crate) landmarks: Vec<usize>,
    /// Flat LocationId → index lookup array for O(1) graph traversal
    pub(crate) id_lookup: Vec<u32>,
}

impl EAMInner {
    /// Full rebuild of landmarks + id_lookup. Use after merge/load_snapshot.
    pub(crate) fn rebuild_graph_cache(&mut self) {
        self.id_lookup = read::build_id_lookup(&self.locations);
        self.landmarks = read::select_landmarks(&self.locations, self.config.num_landmarks);
    }

    /// Incremental update after appending new locations.
    /// O(new_count) instead of O(L). Landmarks are not rebuilt —
    /// existing landmarks are stable enough entry points for graph search.
    pub(crate) fn extend_id_lookup(&mut self, start_idx: usize) {
        for i in start_idx..self.locations.len() {
            let id = self.locations[i].id.0 as usize;
            if id >= self.id_lookup.len() {
                self.id_lookup.resize(id + 1, u32::MAX);
            }
            self.id_lookup[id] = i as u32;
        }
    }
}

/// A single EAM collection, scoped to a collection_id within a shared Store.
pub struct Collection {
    collection_id: u32,
    name: String,
    inner: RwLock<EAMInner>,
    store: Arc<Store>,
}

impl Collection {
    /// Create a new collection, initializing with L_0 random locations.
    pub fn new(
        collection_id: u32,
        name: String,
        store: Arc<Store>,
        config: &EAMConfig,
    ) -> Result<Self> {
        let mut rng = rand::thread_rng();

        let mut locations = Vec::with_capacity(config.l_0);
        for i in 0..config.l_0 {
            let addr = vec_ops::random_unit_vector(config.d, &mut rng);
            locations.push(HardLocation::new(LocationId(i as u64), addr));
        }

        let next_id = config.l_0 as u64;
        let eta = config.eta_0;

        // Persist initial state
        let mut txn = store.write_txn()?;
        for loc in &locations {
            store.put_location(&mut txn, collection_id, loc)?;
        }
        store.put_metadata(
            &mut txn,
            collection_id,
            "next_id",
            &bincode::serialize(&next_id)?,
        )?;
        store.put_metadata(
            &mut txn,
            collection_id,
            "eta",
            &bincode::serialize(&eta)?,
        )?;
        store.put_metadata(
            &mut txn,
            collection_id,
            "config_d",
            &bincode::serialize(&config.d)?,
        )?;
        txn.commit()?;

        let id_lookup = read::build_id_lookup(&locations);
        let landmarks = read::select_landmarks(&locations, config.num_landmarks);

        Ok(Collection {
            collection_id,
            name,
            inner: RwLock::new(EAMInner {
                config: config.clone(),
                locations,
                next_id,
                eta,
                doc_next_id: 0,
                landmarks,
                id_lookup,
            }),
            store,
        })
    }

    /// Load an existing collection from disk.
    pub fn load(
        collection_id: u32,
        name: String,
        store: Arc<Store>,
        config: &EAMConfig,
    ) -> Result<Self> {
        let locations = store.load_all_locations(collection_id)?;

        let next_id: u64 = store
            .load_metadata(collection_id, "next_id")?
            .map(|data| bincode::deserialize(&data))
            .transpose()?
            .unwrap_or(locations.len() as u64);

        let eta: f64 = store
            .load_metadata(collection_id, "eta")?
            .map(|data| bincode::deserialize(&data))
            .transpose()?
            .unwrap_or(config.eta_0);

        let doc_next_id: u64 = store
            .load_metadata(collection_id, "doc_next_id")?
            .map(|data| bincode::deserialize(&data))
            .transpose()?
            .unwrap_or(0);

        // Verify dimensionality
        if let Some(first) = locations.first() {
            if first.address.len() != config.d {
                return Err(HeatherError::DimensionMismatch {
                    expected: config.d,
                    got: first.address.len(),
                });
            }
        }

        let id_lookup = read::build_id_lookup(&locations);
        let landmarks = read::select_landmarks(&locations, config.num_landmarks);

        Ok(Collection {
            collection_id,
            name,
            inner: RwLock::new(EAMInner {
                config: config.clone(),
                locations,
                next_id,
                eta,
                doc_next_id,
                landmarks,
                id_lookup,
            }),
            store,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn collection_id(&self) -> u32 {
        self.collection_id
    }

    /// Write a pattern into the collection.
    pub fn write(&self, input: &[f64]) -> Result<()> {
        vec_ops::validate_vector(input)?;

        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;

        if input.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: input.len(),
            });
        }

        if inner.locations.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }

        let mut rng = rand::thread_rng();
        let EAMInner {
            ref config,
            ref mut locations,
            ref mut next_id,
            ref mut eta,
            ..
        } = *inner;
        let result = write::adaptive_write(input, locations, config, *eta, next_id, &mut rng);

        inner.eta = result.eta;

        let new_locs = result.new_locations;

        // Persist everything in a single atomic transaction
        let mut txn = self.store.write_txn()?;

        for &idx in &result.modified_indices {
            self.store
                .put_location(&mut txn, self.collection_id, &inner.locations[idx])?;
        }

        for loc in &new_locs {
            self.store
                .put_location(&mut txn, self.collection_id, loc)?;
        }

        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "next_id",
            &bincode::serialize(&inner.next_id)?,
        )?;
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "eta",
            &bincode::serialize(&inner.eta)?,
        )?;

        txn.commit()?;

        if !new_locs.is_empty() {
            let start = inner.locations.len();
            inner.locations.extend(new_locs);
            inner.extend_id_lookup(start);
        }

        Ok(())
    }

    /// Read (reconstruct) a pattern from the collection.
    /// Automatically uses graph-accelerated activation when the neighbor graph is ready.
    pub fn read(&self, query: &[f64], strategy: ReadStrategy) -> Result<Vec<f64>> {
        vec_ops::validate_vector(query)?;

        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        if query.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: query.len(),
            });
        }

        if inner.locations.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }

        let k = inner.config.k.min(inner.locations.len());
        let (indices, _sims) = read::activate_auto(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
        );

        match strategy {
            ReadStrategy::HopfieldIter => {
                read::hopfield_iter_from(query, &inner.locations, &inner.config, &indices)
            }
            ReadStrategy::HopfieldSS => {
                let sims: Vec<f64> = indices
                    .iter()
                    .map(|&i| vec_ops::cosine_similarity(query, &inner.locations[i].address))
                    .collect();
                let alpha = vec_ops::softmax(&sims, inner.config.beta);
                let patterns: Vec<Vec<f64>> = indices
                    .iter()
                    .map(|&i| inner.locations[i].normalized_pattern())
                    .collect();
                let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
                Ok(vec_ops::normalize(&vec_ops::weighted_sum(
                    &pattern_refs,
                    &alpha,
                )))
            }
        }
    }

    /// Run KNN merge to consolidate similar locations.
    /// Migrates document index posting lists from removed locations to survivors.
    pub fn merge(&self) -> Result<usize> {
        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
        let EAMInner {
            ref config,
            ref mut locations,
            ..
        } = *inner;
        let result = merge::knn_merge(locations, config);

        if !result.removed_ids.is_empty() {
            let mut txn = self.store.write_txn()?;

            // Migrate posting lists: removed → survivor
            for &(removed_id, survivor_id) in &result.merge_map {
                let removed_docs = self.store.get_doc_ids_for_location_txn(
                    &txn, self.collection_id, removed_id,
                )?;
                if !removed_docs.is_empty() {
                    let mut survivor_docs = self.store.get_doc_ids_for_location_txn(
                        &txn, self.collection_id, survivor_id,
                    )?;
                    survivor_docs.extend(removed_docs);
                    self.store.put_doc_index_entry(
                        &mut txn, self.collection_id, survivor_id, &survivor_docs,
                    )?;
                }
                self.store.delete_doc_index_entry(
                    &mut txn, self.collection_id, removed_id,
                )?;
            }

            for &id in &result.removed_ids {
                self.store
                    .delete_location(&mut txn, self.collection_id, LocationId(id))?;
            }
            txn.commit()?;
        }

        // Rebuild graph cache after topology change
        if !result.removed_ids.is_empty() {
            inner.rebuild_graph_cache();
        }

        Ok(result.merge_count)
    }

    /// Flush all locations to persistent storage.
    pub fn flush(&self) -> Result<()> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        let mut txn = self.store.write_txn()?;
        for loc in &inner.locations {
            self.store
                .put_location(&mut txn, self.collection_id, loc)?;
        }
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "next_id",
            &bincode::serialize(&inner.next_id)?,
        )?;
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "eta",
            &bincode::serialize(&inner.eta)?,
        )?;
        txn.commit()?;

        Ok(())
    }

    /// Write a pattern with associated metadata. Returns the document ID.
    /// Inlines the EAM write to capture activated locations for the document index.
    pub fn write_with_metadata(&self, input: &[f64], metadata: &[u8]) -> Result<u64> {
        vec_ops::validate_vector(input)?;

        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;

        if input.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: input.len(),
            });
        }

        if inner.locations.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }

        // Run adaptive write (same as write())
        let mut rng = rand::thread_rng();
        let EAMInner {
            ref config,
            ref mut locations,
            ref mut next_id,
            ref mut eta,
            ..
        } = *inner;
        let result = write::adaptive_write(input, locations, config, *eta, next_id, &mut rng);
        inner.eta = result.eta;

        // Capture activated location IDs before extending
        let activated_loc_ids: Vec<u64> = result
            .modified_indices
            .iter()
            .map(|&idx| inner.locations[idx].id.0)
            .collect();

        let new_locs = result.new_locations;

        // Assign document ID
        let doc_id = inner.doc_next_id;
        inner.doc_next_id += 1;

        // Persist everything in a single atomic transaction
        let mut txn = self.store.write_txn()?;

        // Persist modified + new locations
        for &idx in &result.modified_indices {
            self.store
                .put_location(&mut txn, self.collection_id, &inner.locations[idx])?;
        }
        for loc in &new_locs {
            self.store
                .put_location(&mut txn, self.collection_id, loc)?;
        }

        // Persist document (vector + metadata)
        let doc_data = bincode::serialize(&(input.to_vec(), metadata.to_vec()))?;
        self.store
            .put_document(&mut txn, self.collection_id, doc_id, &doc_data)?;

        // Update document index: append doc_id to each activated location's posting list
        for &loc_id in &activated_loc_ids {
            self.store
                .append_doc_to_location(&mut txn, self.collection_id, loc_id, doc_id)?;
        }

        // Persist counters
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "next_id",
            &bincode::serialize(&inner.next_id)?,
        )?;
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "eta",
            &bincode::serialize(&inner.eta)?,
        )?;
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "doc_next_id",
            &bincode::serialize(&inner.doc_next_id)?,
        )?;

        txn.commit()?;

        if !new_locs.is_empty() {
            let start = inner.locations.len();
            inner.locations.extend(new_locs);
            inner.extend_id_lookup(start);
        }

        Ok(doc_id)
    }

    /// Retrieve a document's metadata and vector by its ID.
    pub fn get_document(&self, write_index: u64) -> Result<Option<(Vec<f64>, Vec<u8>)>> {
        match self.store.get_document(self.collection_id, write_index)? {
            Some(data) => {
                let (vec, meta): (Vec<f64>, Vec<u8>) = bincode::deserialize(&data)?;
                Ok(Some((vec, meta)))
            }
            None => Ok(None),
        }
    }

    /// List all documents in this collection.
    pub fn list_documents(&self) -> Result<Vec<(u64, Vec<f64>, Vec<u8>)>> {
        let raw = self.store.list_documents(self.collection_id)?;
        let mut docs = Vec::with_capacity(raw.len());
        for (id, data) in raw {
            let (vec, meta): (Vec<f64>, Vec<u8>) = bincode::deserialize(&data)?;
            docs.push((id, vec, meta));
        }
        Ok(docs)
    }

    /// Query documents by cosine similarity to a query vector, returning top-n.
    /// Uses the hard-location posting list index: activates query against locations,
    /// gathers candidate doc IDs from posting lists, then does exact cosine on candidates.
    /// Returns `(doc_id, similarity, metadata_bytes)` sorted descending by similarity.
    pub fn query_documents(&self, query: &[f64], n: usize) -> Result<Vec<(u64, f64, Vec<u8>)>> {
        vec_ops::validate_vector(query)?;

        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        if query.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: query.len(),
            });
        }

        if inner.locations.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Activate query against hard locations (graph-accelerated when ready)
        let k = inner.config.k.min(inner.locations.len());
        let (indices, _sims) = read::activate_auto(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
        );

        // Step 2: Gather candidate doc IDs from posting lists (deduplicated)
        let mut candidate_ids = std::collections::HashSet::new();
        let rtxn = self.store.read_txn()?;
        for &idx in &indices {
            let loc_id = inner.locations[idx].id.0;
            let doc_ids = self.store.get_doc_ids_for_location_txn(&rtxn, self.collection_id, loc_id)?;
            candidate_ids.extend(doc_ids);
        }

        // Step 3: Fetch candidates and compute exact cosine similarity
        let mut scored: Vec<(u64, f64, Vec<u8>)> = Vec::with_capacity(candidate_ids.len());
        for doc_id in candidate_ids {
            if let Some(data) = self.store.get_document_txn(&rtxn, self.collection_id, doc_id)? {
                let (vec, meta): (Vec<f64>, Vec<u8>) = bincode::deserialize(&data)?;
                let sim = vec_ops::cosine_similarity(query, &vec);
                scored.push((doc_id, sim, meta));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    /// Get current collection statistics.
    pub fn stats(&self) -> Result<EAMStats> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        let total_writes: f64 = inner.locations.iter().map(|l| l.write_count).sum();
        let max_write_count = inner
            .locations
            .iter()
            .map(|l| l.write_count)
            .fold(0.0_f64, f64::max);
        let avg_write_count = if inner.locations.is_empty() {
            0.0
        } else {
            total_writes / inner.locations.len() as f64
        };

        Ok(EAMStats {
            num_locations: inner.locations.len(),
            total_writes,
            current_eta: inner.eta,
            avg_write_count,
            max_write_count,
        })
    }

    /// Get a clone of the config.
    pub fn config(&self) -> Result<EAMConfig> {
        Ok(self
            .inner
            .read()
            .map_err(|_| HeatherError::LockPoisoned)?
            .config
            .clone())
    }

    /// Get the current number of locations.
    pub fn num_locations(&self) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .map_err(|_| HeatherError::LockPoisoned)?
            .locations
            .len())
    }

    /// Get summary of all hard locations for UI display.
    pub fn locations_summary(&self) -> Result<Vec<LocationSummary>> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Ok(inner
            .locations
            .iter()
            .enumerate()
            .map(|(i, loc)| {
                let avg_mag = if loc.counter.is_empty() {
                    0.0
                } else {
                    let sum: f64 = loc.counter.iter().map(|c| c.abs()).sum();
                    sum / loc.counter.len() as f64
                };
                LocationSummary {
                    id: i,
                    write_count: loc.write_count,
                    avg_counter_magnitude: avg_mag,
                }
            })
            .collect())
    }

    /// Compute the emergent fingerprint of this collection.
    ///
    /// 1. Computes the weighted centroid of all hard location patterns
    ///    (each location's normalized counter, weighted by its write count).
    /// 2. Refines via Hopfield iterative read, letting EAM's structure
    ///    sharpen the centroid into a true attractor.
    ///
    /// Returns None if no writes have occurred.
    pub fn fingerprint(&self) -> Result<Option<Vec<f64>>> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        let total_writes: f64 = inner.locations.iter().map(|l| l.write_count).sum();
        if total_writes < 1e-10 {
            return Ok(None);
        }

        let d = inner.config.d;

        // Step 1: Weighted centroid of hard location patterns
        let mut centroid = vec![0.0_f64; d];
        for loc in &inner.locations {
            if loc.write_count < 1e-10 {
                continue;
            }
            let pattern = loc.normalized_pattern();
            let weight = loc.write_count;
            for (c, &p) in centroid.iter_mut().zip(pattern.iter()) {
                *c += weight * p;
            }
        }

        // Normalize centroid to unit vector
        centroid = vec_ops::normalize(&centroid);

        // Step 2: Refine via Hopfield iterative read
        let refined = read::hopfield_iter(&centroid, &inner.locations, &inner.config)?;

        Ok(Some(refined))
    }

    /// Extract an in-memory snapshot of this collection's EAM state.
    /// The snapshot is fully decoupled from persistence.
    pub fn snapshot(&self) -> Result<(Vec<HardLocation>, EAMConfig)> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Ok((inner.locations.clone(), inner.config.clone()))
    }

    /// Replace this collection's in-memory EAM state from a snapshot.
    /// Flushes the new state to persistent storage atomically.
    /// Resets next_id to max(location_ids) + 1.
    pub fn load_snapshot(
        &self,
        locations: Vec<HardLocation>,
        config: EAMConfig,
    ) -> Result<()> {
        config.validate()?;

        for loc in &locations {
            if loc.address.len() != config.d {
                return Err(HeatherError::DimensionMismatch {
                    expected: config.d,
                    got: loc.address.len(),
                });
            }
        }

        let next_id = locations
            .iter()
            .map(|l| l.id.0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;

        // Persist atomically: clear old locations, write new ones
        let mut txn = self.store.write_txn()?;

        for loc in &inner.locations {
            self.store
                .delete_location(&mut txn, self.collection_id, loc.id)?;
        }

        for loc in &locations {
            self.store
                .put_location(&mut txn, self.collection_id, loc)?;
        }

        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "next_id",
            &bincode::serialize(&next_id)?,
        )?;
        self.store.put_metadata(
            &mut txn,
            self.collection_id,
            "config_d",
            &bincode::serialize(&config.d)?,
        )?;
        txn.commit()?;

        inner.locations = locations;
        inner.config = config;
        inner.next_id = next_id;
        inner.rebuild_graph_cache();

        Ok(())
    }

    /// Perform a traced read with full activation details.
    /// Uses graph-accelerated activation when available.
    pub fn analyze_read(&self, query: &[f64], strategy: ReadStrategy) -> Result<ReadTrace> {
        vec_ops::validate_vector(query)?;
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        if query.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: query.len(),
            });
        }

        if inner.locations.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }

        let k = inner.config.k.min(inner.locations.len());
        let (indices, sims) = read::activate_auto(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
        );

        match strategy {
            ReadStrategy::HopfieldIter => {
                // Run iterative read from the (possibly graph-found) activation set
                let patterns: Vec<Vec<f64>> = indices
                    .iter()
                    .map(|&i| inner.locations[i].unit_pattern())
                    .collect();
                let addresses: Vec<&[f64]> = indices
                    .iter()
                    .map(|&i| inner.locations[i].address.as_slice())
                    .collect();

                let mut xi = vec_ops::normalize(query);
                let mut converged = false;
                let mut iterations = 0;
                let mut final_weights = vec_ops::softmax(&sims, inner.config.beta);

                for t in 0..inner.config.t_max {
                    iterations = t + 1;
                    let step_sims: Vec<f64> = addresses
                        .iter()
                        .map(|addr| vec_ops::cosine_similarity(&xi, addr))
                        .collect();
                    let alpha = vec_ops::softmax(&step_sims, inner.config.beta);
                    final_weights = alpha.clone();
                    let pattern_refs: Vec<&[f64]> =
                        patterns.iter().map(|p| p.as_slice()).collect();
                    let xi_new = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));
                    let sim = vec_ops::cosine_similarity(&xi, &xi_new);
                    xi = xi_new;
                    if sim > 1.0 - inner.config.epsilon {
                        converged = true;
                        break;
                    }
                }

                let activated_locations = indices
                    .iter()
                    .zip(final_weights.iter())
                    .map(|(&idx, &weight)| {
                        let sim = vec_ops::cosine_similarity(&xi, &inner.locations[idx].address);
                        read::ActivatedLocation {
                            id: idx,
                            similarity: sim,
                            weight,
                        }
                    })
                    .collect();

                Ok(ReadTrace {
                    iterations,
                    converged,
                    activated_locations,
                    result: xi,
                })
            }
            ReadStrategy::HopfieldSS => {
                let alpha = vec_ops::softmax(&sims, inner.config.beta);
                let patterns: Vec<Vec<f64>> = indices
                    .iter()
                    .map(|&i| inner.locations[i].normalized_pattern())
                    .collect();
                let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
                let result = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));

                let activated_locations = indices
                    .iter()
                    .zip(sims.iter())
                    .zip(alpha.iter())
                    .map(|((&idx, &sim), &weight)| read::ActivatedLocation {
                        id: idx,
                        similarity: sim,
                        weight,
                    })
                    .collect();

                Ok(ReadTrace {
                    iterations: 1,
                    converged: true,
                    activated_locations,
                    result,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> EAMConfig {
        let mut config = EAMConfig::new(16).unwrap();
        config.l_0 = 50;
        config.k = 5;
        config
    }

    fn setup_store(dir: &TempDir) -> (Arc<Store>, u32) {
        let store = Arc::new(Store::open(dir.path(), 256).unwrap());
        let mut txn = store.write_txn().unwrap();
        let id = store.create_collection(&mut txn, "test").unwrap();
        txn.commit().unwrap();
        (store, id)
    }

    #[test]
    fn test_create_and_write() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();

        let mut rng = rand::thread_rng();
        let pattern = vec_ops::random_unit_vector(16, &mut rng);
        col.write(&pattern).unwrap();

        let stats = col.stats().unwrap();
        assert!(stats.total_writes > 0.0);
    }

    #[test]
    fn test_write_and_read() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();

        let pattern = vec_ops::normalize(&vec![1.0; 16]);

        for _ in 0..20 {
            col.write(&pattern).unwrap();
        }

        let result = col.read(&pattern, ReadStrategy::HopfieldIter).unwrap();
        let sim = vec_ops::cosine_similarity(&result, &pattern);
        assert!(sim > 0.5, "reconstruction similarity {sim} too low");
    }

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().unwrap();
        let config = test_config();

        let store = Arc::new(Store::open(dir.path(), 256).unwrap());
        let col_id = {
            let mut txn = store.write_txn().unwrap();
            let id = store.create_collection(&mut txn, "test").unwrap();
            txn.commit().unwrap();
            id
        };

        {
            let col = Collection::new(col_id, "test".into(), store.clone(), &config).unwrap();
            let pattern = vec_ops::normalize(&vec![1.0; 16]);
            for _ in 0..10 {
                col.write(&pattern).unwrap();
            }
            col.flush().unwrap();
        }

        {
            let col = Collection::load(col_id, "test".into(), store, &config).unwrap();
            let stats = col.stats().unwrap();
            assert!(stats.total_writes > 0.0);
            assert!(stats.num_locations >= 50);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();

        let bad_input = vec![1.0; 8];
        assert!(col.write(&bad_input).is_err());
    }

    #[test]
    fn test_nan_input_rejected() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();

        let bad_input = vec![f64::NAN; 16];
        assert!(col.write(&bad_input).is_err());

        let bad_query = vec![f64::INFINITY; 16];
        assert!(col.read(&bad_query, ReadStrategy::HopfieldIter).is_err());
    }

    #[test]
    fn test_concurrent_reads() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Arc::new(Collection::new(col_id, "test".into(), store, &config).unwrap());

        let pattern = vec_ops::normalize(&vec![1.0; 16]);
        for _ in 0..10 {
            col.write(&pattern).unwrap();
        }

        let mut handles = vec![];
        for _ in 0..4 {
            let col = Arc::clone(&col);
            let p = pattern.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..10 {
                    let _result = col.read(&p, ReadStrategy::HopfieldIter).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
