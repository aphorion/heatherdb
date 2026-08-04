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

/// A stored document as it comes back off the wire: id, vector, metadata blob.
pub type DocumentRecord = (u64, Vec<f64>, Vec<u8>);

/// Select the attention inverse-temperature β by minimizing the description
/// length of the key set. β is a Gaussian-kernel bandwidth (β = 1/σ² on the
/// unit sphere); this is leave-one-out KDE bandwidth selection. For each key
/// Kⱼ, the log predictive density under a kernel mixture of the *other* keys
/// is `logsumexp_{i≠j}(β·(sᵢⱼ−1)) + (d/2)·ln(β/2π) − ln(m−1)`. The total over
/// j is the negative description length; maximizing it picks β*. The
/// `(d/2)ln β` term rewards sharpness, the leave-one-out logsumexp penalizes
/// it (when β is large, excluded neighbors fall off the kernel) — the trade
/// is the interior optimum. Swept on a log grid; falls back to `default_beta`
/// when fewer than two keys make leave-one-out undefined.
fn select_beta_mdl(keys: &[Vec<f64>], d: usize, default_beta: f64) -> f64 {
    let m = keys.len();
    if m < 2 {
        return default_beta;
    }

    // Pairwise cosine similarities of the (unit-norm) keys.
    let sim = |i: usize, j: usize| vec_ops::dot(&keys[i], &keys[j]);

    // Negative description length (LOO log-likelihood up to β-independent
    // constants) at inverse-temperature β.
    let neg_dl = |beta: f64| -> f64 {
        let half_d_ln_beta = (d as f64) / 2.0 * beta.ln();
        let mut total = 0.0;
        for j in 0..m {
            // logsumexp over i≠j of β·(s_ij − 1)
            let mut max_l = f64::NEG_INFINITY;
            for i in 0..m {
                if i == j {
                    continue;
                }
                let l = beta * (sim(i, j) - 1.0);
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum = 0.0;
            for i in 0..m {
                if i == j {
                    continue;
                }
                sum += (beta * (sim(i, j) - 1.0) - max_l).exp();
            }
            total += max_l + sum.ln() + half_d_ln_beta;
        }
        total
    };

    // Log-grid sweep over plausible bandwidths.
    let (lo, hi, steps) = (0.1_f64, 500.0_f64, 60);
    let (ln_lo, ln_hi) = (lo.ln(), hi.ln());
    let mut best_beta = default_beta;
    let mut best_score = f64::NEG_INFINITY;
    for s in 0..=steps {
        let beta = (ln_lo + (ln_hi - ln_lo) * (s as f64) / (steps as f64)).exp();
        let score = neg_dl(beta);
        if score.is_finite() && score > best_score {
            best_score = score;
            best_beta = beta;
        }
    }
    best_beta
}

/// Statistics about the current state of an EAM collection.
#[derive(Debug, Clone)]
pub struct EAMStats {
    pub num_locations: usize,
    pub total_writes: f64,
    pub current_eta: f64,
    pub avg_write_count: f64,
    pub max_write_count: f64,
}

/// Outcome of a description-length-minimising compression pass.
#[derive(Debug, Clone)]
pub struct CompressResult {
    pub locations_before: usize,
    pub locations_after: usize,
    pub description_length_before: f64,
    pub description_length_after: f64,
    pub merges: usize,
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
    /// Contiguous [L × D] address matrix for cache-friendly brute-force activation.
    /// Row i = `locations[i].address`. Updated incrementally on writes.
    pub(crate) address_matrix: Vec<f64>,
}

impl EAMInner {
    /// Full rebuild of landmarks + id_lookup + address_matrix.
    pub(crate) fn rebuild_graph_cache(&mut self) {
        self.id_lookup = read::build_id_lookup(&self.locations);
        self.landmarks = read::select_landmarks(&self.locations, self.config.num_landmarks);
        self.rebuild_address_matrix();
    }

    /// Rebuild address matrix from locations.
    pub(crate) fn rebuild_address_matrix(&mut self) {
        let d = self.config.d;
        self.address_matrix = Vec::with_capacity(self.locations.len() * d);
        for loc in &self.locations {
            self.address_matrix.extend_from_slice(&loc.address);
        }
    }

    /// Sync specific rows of the address matrix after address migration.
    pub(crate) fn sync_addresses(&mut self, indices: &[usize]) {
        let d = self.config.d;
        for &idx in indices {
            let start = idx * d;
            self.address_matrix[start..start + d].copy_from_slice(&self.locations[idx].address);
        }
    }

    /// Append addresses for newly added locations.
    pub(crate) fn append_addresses(&mut self, start_idx: usize) {
        for i in start_idx..self.locations.len() {
            self.address_matrix
                .extend_from_slice(&self.locations[i].address);
        }
    }

    /// Incremental update after appending new locations.
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
        store.put_metadata(&mut txn, collection_id, "eta", &bincode::serialize(&eta)?)?;
        store.put_metadata(
            &mut txn,
            collection_id,
            "config_d",
            &bincode::serialize(&config.d)?,
        )?;
        txn.commit()?;

        let id_lookup = read::build_id_lookup(&locations);
        let landmarks = read::select_landmarks(&locations, config.num_landmarks);
        let d = config.d;
        let mut address_matrix = Vec::with_capacity(locations.len() * d);
        for loc in &locations {
            address_matrix.extend_from_slice(&loc.address);
        }

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
                address_matrix,
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
        if let Some(first) = locations.first()
            && first.address.len() != config.d
        {
            return Err(HeatherError::DimensionMismatch {
                expected: config.d,
                got: first.address.len(),
            });
        }

        let id_lookup = read::build_id_lookup(&locations);
        let landmarks = read::select_landmarks(&locations, config.num_landmarks);
        let d = config.d;
        let mut address_matrix = Vec::with_capacity(locations.len() * d);
        for loc in &locations {
            address_matrix.extend_from_slice(&loc.address);
        }

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
                address_matrix,
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

        // No EmptyMemory guard: a data-seeded index (l_0 == 0) starts empty and
        // is seeded by the first write (adaptive_write cold-start).

        let mut rng = rand::thread_rng();
        let result = {
            let EAMInner {
                ref config,
                ref mut locations,
                ref mut next_id,
                ref mut eta,
                ref landmarks,
                ref id_lookup,
                ..
            } = *inner;
            let r = write::adaptive_write(
                input, locations, config, *eta, next_id, &mut rng, landmarks, id_lookup,
            );
            *eta = r.eta;
            r
        };

        let new_locs = result.new_locations;

        // Persist everything in a single atomic transaction
        let mut txn = self.store.write_txn()?;

        for &idx in &result.modified_indices {
            self.store
                .put_location(&mut txn, self.collection_id, &inner.locations[idx])?;
        }

        for loc in &new_locs {
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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

        // Sync address matrix for migrated locations
        inner.sync_addresses(&result.modified_indices);

        if !new_locs.is_empty() {
            let start = inner.locations.len();
            inner.locations.extend(new_locs);
            inner.extend_id_lookup(start);
            inner.append_addresses(start);
        }

        Ok(())
    }

    /// Two-field write: route/migrate by `address`, accumulate `counter`.
    /// `Collection::write(x)` is the special case `write_two(x, x, default)`.
    /// With `opts.gate` set this is the consolidation primitive (route by
    /// context, learn a different vector, spawn on incoherence).
    pub fn write_two(
        &self,
        address: &[f64],
        counter: &[f64],
        opts: write::WriteOpts,
    ) -> Result<()> {
        vec_ops::validate_vector(address)?;
        vec_ops::validate_vector(counter)?;

        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;

        if address.len() != inner.config.d || counter.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: if address.len() != inner.config.d {
                    address.len()
                } else {
                    counter.len()
                },
            });
        }

        let mut rng = rand::thread_rng();
        let result = {
            let EAMInner {
                ref config,
                ref mut locations,
                ref mut next_id,
                ref mut eta,
                ref landmarks,
                ref id_lookup,
                ..
            } = *inner;
            let r = write::adaptive_write_two(
                address, counter, locations, config, *eta, next_id, &mut rng, landmarks, id_lookup,
                opts,
            );
            *eta = r.eta;
            r
        };

        let new_locs = result.new_locations;

        let mut txn = self.store.write_txn()?;
        for &idx in &result.modified_indices {
            self.store
                .put_location(&mut txn, self.collection_id, &inner.locations[idx])?;
        }
        for loc in &new_locs {
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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

        inner.sync_addresses(&result.modified_indices);
        if !new_locs.is_empty() {
            let start = inner.locations.len();
            inner.locations.extend(new_locs);
            inner.extend_id_lookup(start);
            inner.append_addresses(start);
        }

        Ok(())
    }

    /// Batch write multiple patterns under a single lock + transaction.
    pub fn write_batch(&self, inputs: &[impl AsRef<[f64]>]) -> Result<()> {
        for input in inputs {
            vec_ops::validate_vector(input.as_ref())?;
        }

        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;

        // No EmptyMemory guard: data-seeded index is seeded by the first write.

        let d = inner.config.d;
        for input in inputs {
            if input.as_ref().len() != d {
                return Err(HeatherError::DimensionMismatch {
                    expected: d,
                    got: input.as_ref().len(),
                });
            }
        }

        let mut rng = rand::thread_rng();
        let mut all_modified = std::collections::HashSet::new();
        let mut all_new_locs: Vec<HardLocation> = Vec::new();

        for input in inputs {
            let input = input.as_ref();
            let result = {
                let EAMInner {
                    ref config,
                    ref mut locations,
                    ref mut next_id,
                    ref mut eta,
                    ref landmarks,
                    ref id_lookup,
                    ..
                } = *inner;
                let r = write::adaptive_write(
                    input, locations, config, *eta, next_id, &mut rng, landmarks, id_lookup,
                );
                *eta = r.eta;
                r
            };

            // Sync addresses for migrated locations immediately
            inner.sync_addresses(&result.modified_indices);
            for &idx in &result.modified_indices {
                all_modified.insert(idx);
            }

            // Append new locations so subsequent writes see them
            if !result.new_locations.is_empty() {
                let start = inner.locations.len();
                let new_locs = result.new_locations;
                for loc in &new_locs {
                    all_new_locs.push(loc.clone());
                }
                inner.locations.extend(new_locs);
                inner.extend_id_lookup(start);
                inner.append_addresses(start);
            }
        }

        // Persist everything in a single atomic transaction
        let mut txn = self.store.write_txn()?;

        for &idx in &all_modified {
            if idx < inner.locations.len() {
                self.store
                    .put_location(&mut txn, self.collection_id, &inner.locations[idx])?;
            }
        }

        for loc in &all_new_locs {
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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
        let (indices, _sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            inner.config.d,
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

    /// Raw dot-product attention read over stored (address=K, counter=V)
    /// locations: `Σ softmax(Q·Kᵢ · scale) · Vᵢ` over the top-k activated
    /// locations. Unlike [`read`]'s HopfieldSS (cosine sims, normalized
    /// patterns, normalized output), this is bit-exact transformer attention
    /// when `scale = 1/√d` and every key participates — the read that lets a
    /// trained model's attention run on the EAM. With top-k < |locations| it
    /// is the sparse attention that, with elastic merging, gives unbounded
    /// context at bounded memory.
    pub fn read_attention(&self, query: &[f64], scale: f64) -> Result<Vec<f64>> {
        self.read_attention_ex(query, scale, None)
    }

    /// [`Self::read_attention`] with an optional location index to drop from the
    /// activated set — leave-one-out reads (a stored key queried against the
    /// rest of the codebook) for honest held-out evaluation. When `exclude`
    /// is set, one extra location is activated so `config.k` neighbours
    /// survive the drop.
    pub fn read_attention_ex(
        &self,
        query: &[f64],
        scale: f64,
        exclude: Option<usize>,
    ) -> Result<Vec<f64>> {
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

        let want = inner.config.k + exclude.is_some() as usize;
        let k = want.min(inner.locations.len());
        let (mut indices, _sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            inner.config.d,
        );
        if let Some(ex) = exclude {
            indices.retain(|&i| i != ex);
            indices.truncate(inner.config.k);
        }

        // Count-weighted raw-dot scores: a merged engram represents
        // write_count tokens, so it enters the softmax with that multiplicity
        // — αᵢ ∝ write_countᵢ · exp(scale · Q·Kᵢ), folded as the logit
        // scale·Q·Kᵢ + ln(write_countᵢ). With write_count == 1 everywhere this
        // is exactly softmax(Q·Kᵢ · scale); with merged engrams it reconstructs
        // the attention the un-merged tokens would have produced.
        let logits: Vec<f64> = indices
            .iter()
            .map(|&i| {
                scale * vec_ops::dot(query, &inner.locations[i].address)
                    + inner.locations[i].write_count.max(1e-12).ln()
            })
            .collect();
        let alpha = vec_ops::softmax(&logits, 1.0);
        // raw values (V = counter / write_count), NO output normalization
        let values: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| inner.locations[i].normalized_pattern())
            .collect();
        let value_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
        Ok(vec_ops::weighted_sum(&value_refs, &alpha))
    }

    /// Calibrate the attention temperature by minimizing the leave-one-out
    /// description length of the stored *values*. For each location j the value
    /// Vⱼ is reconstructed from its top-k key-neighbours at inverse-temperature
    /// β — `recon = Σ softmax(β·Kⱼ·Kᵢ)·Vᵢ` — and the held-out code length is
    /// `−ln(Vⱼ·recon)` (categorical NLL when V is a one-hot label; a proper
    /// Gaussian-residual code in general). The sum over j is convex-with-interior
    /// in β: too flat reconstructs the mean value, too sharp reconstructs only
    /// the nearest neighbour's value — both code the held-out value poorly. This
    /// is the dimension-free, value-aware temperature; unlike a key-only KDE
    /// bandwidth it does not degenerate when the ambient dimension is large.
    /// Returns `(β*, DL*)`. The neighbour lists are β-independent, so the sweep
    /// computes them once.
    pub fn calibrate_beta(&self, betas: &[f64]) -> Result<(f64, f64)> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        let n = inner.locations.len();
        if n < 2 {
            return Ok((inner.config.beta, 0.0));
        }
        let k = inner.config.k.min(n - 1);
        let vals: Vec<Vec<f64>> = inner
            .locations
            .iter()
            .map(|l| l.normalized_pattern())
            .collect();

        // top-k key-neighbours of each location (excluding itself), with the
        // raw dot logit Kⱼ·Kᵢ that read_attention scales by β.
        let neighbours: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|j| {
                let mut sims: Vec<(usize, f64)> = (0..n)
                    .filter(|&i| i != j)
                    .map(|i| {
                        (
                            i,
                            vec_ops::dot(&inner.locations[j].address, &inner.locations[i].address),
                        )
                    })
                    .collect();
                sims.sort_by(|a, b| b.1.total_cmp(&a.1));
                sims.truncate(k);
                sims
            })
            .collect();

        let mut best = (
            betas.first().copied().unwrap_or(inner.config.beta),
            f64::INFINITY,
        );
        for &beta in betas {
            let mut dl = 0.0;
            for (j, nb) in neighbours.iter().enumerate() {
                let logits: Vec<f64> = nb.iter().map(|&(_, s)| beta * s).collect();
                let alpha = vec_ops::softmax(&logits, 1.0);
                let refs: Vec<&[f64]> = nb.iter().map(|&(i, _)| vals[i].as_slice()).collect();
                let recon = vec_ops::weighted_sum(&refs, &alpha);
                let p = vec_ops::dot(&vals[j], &recon).max(1e-12);
                dl -= p.ln();
            }
            if dl < best.1 {
                best = (beta, dl);
            }
        }
        Ok(best)
    }

    /// Self-calibrating attention read. The softmax inverse-temperature is a
    /// Gaussian-kernel bandwidth: for unit-norm keys `‖Q−Kᵢ‖² = 2−2·Q·Kᵢ`, so
    /// `exp(β·Q·Kᵢ) ∝ exp(−‖Q−Kᵢ‖²/2σ²)` with `β = 1/σ²`. Instead of fixing the
    /// temperature, β is chosen to minimize the description length of the
    /// activated key set — leave-one-out KDE predictive coding: each key is
    /// reconstructed from the others, summed log-loss plus the `d/2·ln(2π/β)`
    /// code for the bandwidth itself. This has an interior optimum (too sharp
    /// → each key only predicts itself → infinite loss off-codebook; too flat
    /// → mean, no resolution). The read then runs at β*, so confidence tracks
    /// the codebook's own geometry rather than a hard-wired knob: a query
    /// equidistant to well-separated keys reads soft (MaxEnt abstention), a
    /// query inside a tight cluster reads sharp. Returns `(value, β*)`.
    pub fn read_attention_mdl(&self, query: &[f64]) -> Result<(Vec<f64>, f64)> {
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
        let (indices, _sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            inner.config.d,
        );

        // Unit-norm activated keys — kernel geometry needs the cosine sphere.
        let keys: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| vec_ops::normalize(&inner.locations[i].address))
            .collect();
        let beta = select_beta_mdl(&keys, inner.config.d, inner.config.beta);

        // Read at β*: count-weighted raw-dot attention, identical form to
        // `read_attention` but with the self-selected temperature.
        let logits: Vec<f64> = indices
            .iter()
            .map(|&i| {
                beta * vec_ops::dot(query, &inner.locations[i].address)
                    + inner.locations[i].write_count.max(1e-12).ln()
            })
            .collect();
        let alpha = vec_ops::softmax(&logits, 1.0);
        let values: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| inner.locations[i].normalized_pattern())
            .collect();
        let value_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
        Ok((vec_ops::weighted_sum(&value_refs, &alpha), beta))
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
                    &txn,
                    self.collection_id,
                    removed_id,
                )?;
                if !removed_docs.is_empty() {
                    let mut survivor_docs = self.store.get_doc_ids_for_location_txn(
                        &txn,
                        self.collection_id,
                        survivor_id,
                    )?;
                    survivor_docs.extend(removed_docs);
                    self.store.put_doc_index_entry(
                        &mut txn,
                        self.collection_id,
                        survivor_id,
                        &survivor_docs,
                    )?;
                }
                self.store
                    .delete_doc_index_entry(&mut txn, self.collection_id, removed_id)?;
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

    /// Description length (bits) of the collection under a minimum-description-length
    /// reading: `L·kappa` to specify the L locations (the model) plus `W·log2(L)` to
    /// encode each of the W writes as which location it selects (the data). This is
    /// the engine's native entropy readout; compression lowers it.
    pub fn description_length(&self, kappa: f64) -> Result<f64> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        let l = inner.locations.len().max(1) as f64;
        let w: f64 = inner
            .locations
            .iter()
            .map(|x| x.write_count)
            .sum::<f64>()
            .max(1.0);
        Ok(l * kappa + w * l.log2())
    }

    /// Minimise description length: greedily merge the closest two locations while it
    /// pays — model saving (`kappa` + cheaper encoding) exceeds the merge's data-fit
    /// cost (`lambda` × the variance increase) — and stop at the minimum. The engine
    /// auto-calibrates how many locations its data actually warrants. Compression is
    /// the native objective; the location count is the emergent result.
    pub fn compress(&self, kappa: f64, lambda: f64) -> Result<CompressResult> {
        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
        let w: f64 = inner
            .locations
            .iter()
            .map(|x| x.write_count)
            .sum::<f64>()
            .max(1.0);
        let locations_before = inner.locations.len();
        let dl_before =
            (locations_before.max(1) as f64) * kappa + w * (locations_before.max(1) as f64).log2();

        let mut removed_ids: Vec<u64> = Vec::new();
        let mut parent: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();

        loop {
            let n = inner.locations.len();
            if n <= 2 {
                break;
            }
            // closest pair by cosine (addresses are unit norm)
            let (mut bi, mut bj, mut bc) = (0usize, 1usize, -2.0f64);
            for i in 0..n {
                for j in (i + 1)..n {
                    let c = vec_ops::dot(&inner.locations[i].address, &inner.locations[j].address);
                    if c > bc {
                        bc = c;
                        bi = i;
                        bj = j;
                    }
                }
            }
            let wi = inner.locations[bi].write_count.max(1e-9);
            let wj = inner.locations[bj].write_count.max(1e-9);
            let inertia = (wi * wj / (wi + wj)) * 2.0 * (1.0 - bc).max(0.0);
            let model_saving = kappa + w * ((n as f64).log2() - ((n - 1) as f64).log2());
            if model_saving <= inertia * lambda {
                break; // MDL minimum reached — auto-calibrated location count
            }
            // merge bj into bi (bi survives), write-weighted
            let removed_id = inner.locations[bj].id.0;
            let survivor_id = inner.locations[bi].id.0;
            let na: Vec<f64> = inner.locations[bi]
                .address
                .iter()
                .zip(inner.locations[bj].address.iter())
                .map(|(ai, aj)| wi * ai + wj * aj)
                .collect();
            let na = vec_ops::normalize(&na);
            let cj = inner.locations[bj].counter.clone();
            let nj = inner.locations[bj].neighbors.clone();
            {
                let li = &mut inner.locations[bi];
                for (c, add) in li.counter.iter_mut().zip(cj.iter()) {
                    *c += add;
                }
                li.address = na;
                li.write_count = wi + wj;
                li.neighbors.extend(nj);
            }
            inner.locations.remove(bj);
            parent.insert(removed_id, survivor_id);
            removed_ids.push(removed_id);
        }

        let locations_after = inner.locations.len();
        let dl_after =
            (locations_after.max(1) as f64) * kappa + w * (locations_after.max(1) as f64).log2();

        if !removed_ids.is_empty() {
            let mut txn = self.store.write_txn()?;
            for &removed in &removed_ids {
                let mut final_surv = removed; // resolve through the merge chain
                while let Some(&s) = parent.get(&final_surv) {
                    final_surv = s;
                }
                let removed_docs =
                    self.store
                        .get_doc_ids_for_location_txn(&txn, self.collection_id, removed)?;
                if !removed_docs.is_empty() {
                    let mut sd = self.store.get_doc_ids_for_location_txn(
                        &txn,
                        self.collection_id,
                        final_surv,
                    )?;
                    sd.extend(removed_docs);
                    self.store.put_doc_index_entry(
                        &mut txn,
                        self.collection_id,
                        final_surv,
                        &sd,
                    )?;
                }
                self.store
                    .delete_doc_index_entry(&mut txn, self.collection_id, removed)?;
                self.store
                    .delete_location(&mut txn, self.collection_id, LocationId(removed))?;
            }
            for loc in &inner.locations {
                self.store.put_location(&mut txn, self.collection_id, loc)?;
            }
            txn.commit()?;
            inner.rebuild_graph_cache();
        }

        Ok(CompressResult {
            locations_before,
            locations_after,
            description_length_before: dl_before,
            description_length_after: dl_after,
            merges: removed_ids.len(),
        })
    }

    /// Flush all locations to persistent storage.
    pub fn flush(&self) -> Result<()> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        let mut txn = self.store.write_txn()?;
        for loc in &inner.locations {
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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

        // No EmptyMemory guard: a data-seeded index (l_0 == 0) starts empty and
        // is seeded by the first write (adaptive_write cold-start).

        // Run adaptive write (same as write())
        let mut rng = rand::thread_rng();
        let result = {
            let EAMInner {
                ref config,
                ref mut locations,
                ref mut next_id,
                ref mut eta,
                ref landmarks,
                ref id_lookup,
                ..
            } = *inner;
            let r = write::adaptive_write(
                input, locations, config, *eta, next_id, &mut rng, landmarks, id_lookup,
            );
            *eta = r.eta;
            r
        };

        // Capture the location IDs this write touched, before extending.
        //
        // Both halves are load-bearing. `modified_indices` covers locations
        // the write joined; `new_locations` covers locations the write
        // *created* — a novelty/overload split, and every write into an empty
        // collection, which has nothing to join. Indexing only the former
        // leaves such documents in no posting list at all, so
        // `query_documents` (which gathers candidates from posting lists)
        // can never return them: stored, counted by `stats`, unreachable.
        let new_locs = result.new_locations;
        let activated_loc_ids: Vec<u64> = result
            .modified_indices
            .iter()
            .map(|&idx| inner.locations[idx].id.0)
            .chain(new_locs.iter().map(|loc| loc.id.0))
            .collect();

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
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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

        // Sync address matrix for migrated locations
        inner.sync_addresses(&result.modified_indices);

        if !new_locs.is_empty() {
            let start = inner.locations.len();
            inner.locations.extend(new_locs);
            inner.extend_id_lookup(start);
            inner.append_addresses(start);
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

    /// The ids of this collection's hard locations, in storage order.
    pub fn location_ids(&self) -> Result<Vec<u64>> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Ok(inner.locations.iter().map(|l| l.id.0).collect())
    }

    /// The document ids indexed under one hard location — the posting list
    /// `query_documents` gathers its candidates from.
    pub fn doc_ids_for_location(&self, location_id: u64) -> Result<Vec<u64>> {
        self.store
            .get_doc_ids_for_location(self.collection_id, location_id)
    }

    /// Delete a document: remove it from the document store and from every
    /// posting list that references it. Returns true when it existed.
    ///
    /// **This is a retrieval and citation tombstone, not an erasure of the
    /// document's influence on the memory.** A write accumulates its pattern
    /// into one or more hard locations, and superposition cannot cleanly
    /// subtract one term from a merged engram — the arithmetic that added it
    /// is not invertible once other writes have landed on the same location.
    /// After this call the document can never be returned by
    /// `query_documents`, fetched by id, or named as a contributor; its
    /// residual contribution to a location's address and counter decays only
    /// through subsequent writes and consolidation.
    ///
    /// Callers with a legal erasure obligation must treat that distinction as
    /// load-bearing: use a separate collection (or database) for material
    /// that may need to be destroyed, so the unit of erasure is one the
    /// substrate can actually drop.
    pub fn delete_document(&self, write_index: u64) -> Result<bool> {
        let _guard = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
        let mut txn = self.store.write_txn()?;
        let existed =
            self.store
                .delete_document_everywhere(&mut txn, self.collection_id, write_index)?;
        txn.commit()?;
        Ok(existed)
    }

    /// List all documents in this collection.
    pub fn list_documents(&self) -> Result<Vec<DocumentRecord>> {
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
        let (indices, _sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            inner.config.d,
        );

        // Step 2: Gather candidate doc IDs from posting lists (deduplicated)
        let mut candidate_ids = std::collections::HashSet::new();
        let rtxn = self.store.read_txn()?;
        for &idx in &indices {
            let loc_id = inner.locations[idx].id.0;
            let doc_ids =
                self.store
                    .get_doc_ids_for_location_txn(&rtxn, self.collection_id, loc_id)?;
            candidate_ids.extend(doc_ids);
        }

        // Step 3: Fetch candidates and compute exact cosine similarity
        let mut scored: Vec<(u64, f64, Vec<u8>)> = Vec::with_capacity(candidate_ids.len());
        for doc_id in candidate_ids {
            if let Some(data) = self
                .store
                .get_document_txn(&rtxn, self.collection_id, doc_id)?
            {
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
    pub fn load_snapshot(&self, locations: Vec<HardLocation>, config: EAMConfig) -> Result<()> {
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
            self.store.put_location(&mut txn, self.collection_id, loc)?;
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
        let (indices, sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            inner.config.d,
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
                    let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
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

    /// read_attention must equal a hand-computed softmax(Q·Kᵢ·scale)·Vᵢ over
    /// the stored locations — i.e. it is bit-exact transformer attention, not
    /// the cosine/normalized HopfieldSS read.
    #[test]
    fn read_attention_is_exact_softmax_dot() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let mut config = test_config();
        config.l_0 = 0; // start empty: written pairs are the only locations
        config.k = 64; // every key participates (full attention)
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();
        let d = 16;
        let mut rng = rand::thread_rng();
        let opts = crate::write::WriteOpts::default();
        for _ in 0..5 {
            let key = vec_ops::random_unit_vector(d, &mut rng);
            let val = vec_ops::random_unit_vector(d, &mut rng);
            col.write_two(&key, &val, opts).unwrap();
        }
        let query = vec_ops::random_unit_vector(d, &mut rng);
        let scale = 1.0 / (d as f64).sqrt();

        let manual = {
            let inner = col.inner.read().unwrap();
            // count-weighted logits: scale·Q·Kᵢ + ln(write_countᵢ)
            let logits: Vec<f64> = inner
                .locations
                .iter()
                .map(|l| scale * vec_ops::dot(&query, &l.address) + l.write_count.max(1e-12).ln())
                .collect();
            let alpha = vec_ops::softmax(&logits, 1.0);
            let vals: Vec<Vec<f64>> = inner
                .locations
                .iter()
                .map(|l| l.normalized_pattern())
                .collect();
            let refs: Vec<&[f64]> = vals.iter().map(|v| v.as_slice()).collect();
            vec_ops::weighted_sum(&refs, &alpha)
        };

        let got = col.read_attention(&query, scale).unwrap();
        assert_eq!(got.len(), d);
        for (a, b) in got.iter().zip(&manual) {
            assert!(
                (a - b).abs() < 1e-9,
                "read_attention {a} != manual softmax-dot {b}"
            );
        }
    }

    /// End-to-end: a non-competitive collection stores K/V verbatim (norm
    /// preserved), so write_two + read_attention reproduces exact dot
    /// attention over the RAW written keys. Non-unit keys make this fail if
    /// the write had normalized the address.
    #[test]
    fn non_competitive_write_then_read_attention_is_exact() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let mut config = test_config();
        config.l_0 = 0;
        config.k = 64;
        config.competitive = false; // verbatim streaming append
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();
        let d = 16;
        let mut rng = rand::thread_rng();
        let opts = crate::write::WriteOpts::default();

        let mut keys = Vec::new();
        let mut vals = Vec::new();
        for i in 0..5 {
            // non-unit key (norm varies) — would mismatch if the write normalized
            let key: Vec<f64> = vec_ops::random_unit_vector(d, &mut rng)
                .iter()
                .map(|x| x * (1.0 + i as f64 * 0.7))
                .collect();
            let val = vec_ops::random_unit_vector(d, &mut rng);
            col.write_two(&key, &val, opts).unwrap();
            keys.push(key);
            vals.push(val);
        }
        let query: Vec<f64> = vec_ops::random_unit_vector(d, &mut rng);
        let scale = 1.0 / (d as f64).sqrt();

        // manual softmax-dot attention over the RAW written keys/values
        let sims: Vec<f64> = keys.iter().map(|kk| vec_ops::dot(&query, kk)).collect();
        let alpha = vec_ops::softmax(&sims, scale);
        let refs: Vec<&[f64]> = vals.iter().map(|v| v.as_slice()).collect();
        let manual = vec_ops::weighted_sum(&refs, &alpha);

        let got = col.read_attention(&query, scale).unwrap();
        for (a, b) in got.iter().zip(&manual) {
            assert!(
                (a - b).abs() < 1e-9,
                "read_attention {a} != raw-K attention {b}"
            );
        }
    }

    /// MDL-selected temperature calibrates confidence to codebook geometry:
    /// β* is broad (soft read) when keys are well-separated and an ambiguous
    /// query has no honest winner, and sharp when keys cluster tightly. The
    /// soft read is MaxEnt abstention — it refuses to amplify a hair-thin
    /// similarity gap into a confident pick the way a hard-wired high β does.
    /// The very first document written into an empty collection must be
    /// retrievable. It creates a location rather than joining one, so a
    /// document index that only records `modified_indices` leaves it in no
    /// posting list: stored, counted by `stats`, and permanently invisible to
    /// `query_documents`. Regression test for exactly that.
    #[test]
    fn first_document_in_an_empty_collection_is_retrievable() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let d = 16;
        let mut config = EAMConfig::new(d).unwrap();
        config.k = 1;
        config.gamma = 0.0;
        let col = Collection::new(col_id, "docs".into(), store, &config).unwrap();
        let v = basis_vec(d, 0);

        let id = col.write_with_metadata(&v, br#"{"kind":"first"}"#).unwrap();
        assert_eq!(
            col.num_locations().unwrap(),
            1,
            "the write created a location rather than joining one"
        );

        let got = col.query_documents(&v, 10).unwrap();
        assert_eq!(got.len(), 1, "the first document must be queryable");
        assert_eq!(got[0].0, id);
    }

    /// A split creates a new location too. A document that lands only on a
    /// newly-spawned location must still be indexed.
    #[test]
    fn documents_on_newly_spawned_locations_are_indexed() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let d = 16;
        let mut config = EAMConfig::new(d).unwrap();
        config.k = 1;
        config.gamma = 0.0;
        let col = Collection::new(col_id, "docs".into(), store, &config).unwrap();

        // Mutually orthogonal patterns: each is novel enough to spawn its own
        // location rather than join an existing one.
        let vs = [
            basis_vec(d, 0),
            basis_vec(d, 4),
            basis_vec(d, 8),
            basis_vec(d, 12),
        ];
        let mut ids = Vec::new();
        for (i, v) in vs.iter().enumerate() {
            ids.push(
                col.write_with_metadata(v, format!(r#"{{"i":{i}}}"#).as_bytes())
                    .unwrap(),
            );
        }

        // Every document must be findable by its own vector.
        for (v, id) in vs.iter().zip(&ids) {
            let got = col.query_documents(v, 10).unwrap();
            assert!(
                got.iter().any(|(g, _, _)| g == id),
                "document {id} written to a spawned location is unreachable"
            );
        }
    }

    /// Deletion removes a document from retrieval and from every posting
    /// list, leaves its neighbours alone, and is idempotent.
    #[test]
    fn delete_document_removes_it_from_retrieval_and_postings() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let d = 16;
        let mut config = EAMConfig::new(d).unwrap();
        config.k = 1;
        config.gamma = 0.0;
        let col = Collection::new(col_id, "docs".into(), store, &config).unwrap();

        let a = basis_vec(d, 0);
        let b = basis_vec(d, 8);
        let victim = col.write_with_metadata(&a, br#"{"n":"remove"}"#).unwrap();
        let survivor = col.write_with_metadata(&b, br#"{"n":"keep"}"#).unwrap();

        assert!(
            col.delete_document(victim).unwrap(),
            "an existing document reports deleted"
        );

        // gone from query results ...
        let after = col.query_documents(&a, 10).unwrap();
        assert!(
            !after.iter().any(|(id, _, _)| *id == victim),
            "deleted document still returned by query_documents"
        );
        // ... and from the document store ...
        assert!(col.get_document(victim).unwrap().is_none());
        // ... while its neighbour survives.
        assert!(
            col.query_documents(&b, 10)
                .unwrap()
                .iter()
                .any(|(id, _, _)| *id == survivor)
        );

        // no posting list anywhere still names it
        for loc_id in col.location_ids().unwrap() {
            assert!(
                !col.doc_ids_for_location(loc_id).unwrap().contains(&victim),
                "location {loc_id} still references the deleted document"
            );
        }

        // idempotent: a replayed tombstone must not error
        assert!(!col.delete_document(victim).unwrap());
        assert!(!col.delete_document(999_999).unwrap());
    }

    fn basis_vec(d: usize, i: usize) -> Vec<f64> {
        let mut v = vec![0.0; d];
        v[i] = 1.0;
        v
    }

    fn entropy(alpha: &[f64]) -> f64 {
        -alpha
            .iter()
            .filter(|&&a| a > 0.0)
            .map(|&a| a * a.ln())
            .sum::<f64>()
    }

    /// Attention weights at a fixed β for raw-dot logits over `keys`.
    fn alpha_at(query: &[f64], keys: &[Vec<f64>], beta: f64) -> Vec<f64> {
        let logits: Vec<f64> = keys
            .iter()
            .map(|kk| beta * vec_ops::dot(query, kk))
            .collect();
        vec_ops::softmax(&logits, 1.0)
    }

    fn write_keys(col: &Collection, keys: &[Vec<f64>]) {
        let opts = crate::write::WriteOpts::default();
        for (i, k) in keys.iter().enumerate() {
            // distinct value per key (a fresh basis dir) so reads are separable
            let val = basis_vec(keys[0].len(), i);
            col.write_two(k, &val, opts).unwrap();
        }
    }

    fn separated_collection(dir: &TempDir, d: usize) -> (Collection, Vec<Vec<f64>>) {
        let (store, col_id) = setup_store(dir);
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 0;
        config.k = 64;
        config.competitive = false;
        let col = Collection::new(col_id, "sep".into(), store, &config).unwrap();
        // two orthonormal keys — maximally separated on the sphere
        let keys = vec![basis_vec(d, 0), basis_vec(d, 1)];
        write_keys(&col, &keys);
        (col, keys)
    }

    fn clustered_collection(dir: &TempDir, d: usize) -> (Collection, Vec<Vec<f64>>) {
        let (store, col_id) = setup_store(dir);
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 0;
        config.k = 64;
        config.competitive = false;
        let col = Collection::new(col_id, "tight".into(), store, &config).unwrap();
        // keys tightly packed around e0 (pairwise cosine ≈ 0.997)
        let keys: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let mut v = basis_vec(d, 0);
                v[i + 1] = 0.05;
                vec_ops::normalize(&v)
            })
            .collect();
        write_keys(&col, &keys);
        (col, keys)
    }

    #[test]
    fn mdl_temperature_calibrates_to_geometry() {
        let d = 16;
        let dir_s = TempDir::new().unwrap();
        let dir_t = TempDir::new().unwrap();
        let (sep, sep_keys) = separated_collection(&dir_s, d);
        let (tight, _tight_keys) = clustered_collection(&dir_t, d);

        // Ambiguous query: nearly equidistant to the two separated keys
        // (sims 0.51 vs 0.49 → a 0.02 hair, no honest winner).
        let query = vec_ops::normalize(
            &sep_keys[0]
                .iter()
                .zip(&sep_keys[1])
                .map(|(a, b)| 0.51 * a + 0.49 * b)
                .collect::<Vec<_>>(),
        );

        let (_v_sep, beta_sep) = sep.read_attention_mdl(&query).unwrap();
        let (_v_tight, beta_tight) = tight.read_attention_mdl(&query).unwrap();

        // Calibration: a tight codebook earns a sharp temperature; a spread
        // one does not. This is the whole claim — β tracks resolution.
        assert!(
            beta_tight > beta_sep,
            "tight codebook should select sharper β: tight={beta_tight} sep={beta_sep}"
        );

        // The MDL read over the separated keys abstains: its attention entropy
        // is far higher than a hard-wired sharp read on the same hair-thin gap.
        let h_mdl = entropy(&alpha_at(&query, &sep_keys, beta_sep));
        let h_hard = entropy(&alpha_at(&query, &sep_keys, 100.0));
        assert!(
            h_mdl > h_hard + 0.3,
            "MDL read should abstain (high entropy) vs hard-wired sharp: \
             h_mdl={h_mdl} (β*={beta_sep}) h_hard={h_hard}"
        );
        // ...and stay near the 50/50 ceiling (ln 2 ≈ 0.693) rather than collapse.
        assert!(
            h_mdl > 0.6,
            "MDL read should be near-maximal entropy: {h_mdl}"
        );
    }

    fn empty_seeded_collection(dir: &TempDir) -> Collection {
        let (store, col_id) = setup_store(dir);
        let mut config = test_config();
        config.l_0 = 0; // data-seeded: start empty so written families are the only locations
        Collection::new(col_id, "test".into(), store, &config).unwrap()
    }

    /// The gate's defining behavior: same context (address), different law
    /// (counter) must land in separate families — the grid case that defeats
    /// address-only routing.
    #[test]
    fn write_two_gate_separates_by_counter() {
        let dir = TempDir::new().unwrap();
        let col = empty_seeded_collection(&dir);
        let d = 16;
        let context = vec_ops::normalize(&vec![1.0; d]); // shared address
        let mut law_x = vec![0.0; d];
        law_x[0] = 1.0;
        let mut law_y = vec![0.0; d];
        law_y[8] = 1.0; // orthogonal law
        let opts = crate::write::WriteOpts {
            gate: true,
            tau_cohere: 0.2,
            ..Default::default()
        };

        col.write_two(&context, &law_x, opts).unwrap();
        col.write_two(&context, &law_y, opts).unwrap();
        col.write_two(&context, &law_x, opts).unwrap();
        col.write_two(&context, &law_y, opts).unwrap();

        let (locs, _) = col.snapshot().unwrap();
        assert_eq!(
            locs.len(),
            2,
            "gate should split same-context different-law into two families"
        );
        let got_x = locs
            .iter()
            .any(|l| vec_ops::cosine_similarity(&l.counter, &law_x) > 0.9);
        let got_y = locs
            .iter()
            .any(|l| vec_ops::cosine_similarity(&l.counter, &law_y) > 0.9);
        assert!(got_x && got_y, "each family's counter holds its own law");
    }

    /// Without the gate, the shared address collapses everything into one
    /// location — the contrast that makes the gate necessary for consolidation.
    #[test]
    fn write_two_no_gate_collapses_shared_context() {
        let dir = TempDir::new().unwrap();
        let col = empty_seeded_collection(&dir);
        let d = 16;
        let context = vec_ops::normalize(&vec![1.0; d]);
        let mut law_x = vec![0.0; d];
        law_x[0] = 1.0;
        let mut law_y = vec![0.0; d];
        law_y[8] = 1.0;
        let opts = crate::write::WriteOpts {
            gate: false,
            tau_cohere: 0.2,
            ..Default::default()
        };

        col.write_two(&context, &law_x, opts).unwrap();
        col.write_two(&context, &law_y, opts).unwrap();
        col.write_two(&context, &law_x, opts).unwrap();

        let (locs, _) = col.snapshot().unwrap();
        assert_eq!(locs.len(), 1, "no gate: shared address keeps one location");
    }

    #[test]
    fn test_write_and_read() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let config = test_config();
        let col = Collection::new(col_id, "test".into(), store, &config).unwrap();

        let pattern = vec_ops::normalize(&[1.0; 16]);

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
            let pattern = vec_ops::normalize(&[1.0; 16]);
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

        let pattern = vec_ops::normalize(&[1.0; 16]);
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

    /// Compression minimises description length and auto-calibrates the location
    /// count to the data's true structure: near-duplicates merge, distinct
    /// clusters survive, and reads still resolve.
    #[test]
    fn compress_recovers_true_structure() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let mut config = test_config();
        config.l_0 = 0;
        let d = config.d;
        let col = Collection::new(col_id, "t".into(), store, &config).unwrap();

        // 3 distinct clusters, 2 near-duplicate locations each (6 total)
        let mut locs = Vec::new();
        for k in 0..3 {
            let mut proto = vec![0.0; d];
            proto[k * 4] = 1.0;
            for j in 0..2 {
                let mut a = proto.clone();
                a[1] += 0.01 * (j as f64); // tiny perturbation → near-duplicate
                let a = vec_ops::normalize(&a);
                let mut loc = HardLocation::new(LocationId((k * 2 + j) as u64), a.clone());
                loc.counter = a;
                loc.write_count = 1.0;
                locs.push(loc);
            }
        }
        col.load_snapshot(locs, config.clone()).unwrap();

        let dl_before = col.description_length(d as f64).unwrap();
        let r = col.compress(d as f64, 30.0).unwrap();

        assert_eq!(r.locations_before, 6);
        assert_eq!(
            r.locations_after, 3,
            "merges duplicates, keeps the 3 true clusters"
        );
        assert!(
            r.description_length_after < dl_before,
            "description length must drop"
        );
        assert_eq!(
            col.num_locations().unwrap(),
            3,
            "persisted location count matches the compressed result"
        );

        // a query near cluster 0 still resolves there
        let mut q = vec![0.0; d];
        q[0] = 1.0;
        let res = col.read(&q, ReadStrategy::HopfieldSS).unwrap();
        assert!(vec_ops::dot(&vec_ops::normalize(&res), &q) > 0.5);
    }
}
