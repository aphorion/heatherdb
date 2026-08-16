use std::sync::{Arc, RwLock};

use rayon::prelude::*;
use realfft::num_complex::Complex;

use crate::config::EAMConfig;
use crate::error::{HeatherError, Result};
use crate::location::{HardLocation, LocationId};
use crate::merge;
use crate::read;
pub use crate::read::{ActivatedLocation, AttentionContributor, AttentionTrace, ReadTrace};
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

/// Softest cleanup temperature [`Collection::query_documents_multi_role`] will
/// run at; explicit overrides below it are raised to it.
///
/// See [`RoleCleanup::Beta`] for the measured curve this number comes from.
pub const MIN_CLEANUP_BETA: f64 = 30.0;

/// How the filler recovered at a role is cleaned up before it is scored.
///
/// Unbinding recovers the filler plus crosstalk from every other role in the
/// bundle, so the raw recovered vector understates the match. Cleanup denoises
/// it against the collection's codebook — a Hopfield/attention read at inverse
/// temperature β — before the cosine is taken.
///
/// The default is [`Mdl`](RoleCleanup::Mdl): correctness is the default, speed
/// is opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RoleCleanup {
    /// Clean up at the MDL-calibrated temperature the engine selects for
    /// itself, the same `select_beta_mdl` bandwidth
    /// [`Collection::read_attention_mdl`] reads at. The temperature then
    /// tracks the codebook's own geometry instead of a hard-wired knob.
    #[default]
    Mdl,

    /// Clean up at an explicit inverse temperature.
    ///
    /// # The sharpness cliff
    ///
    /// Cleanup quality is brutally sensitive to β, and the failure is quiet —
    /// a soft β returns a plausible-looking ordering that is simply wrong.
    /// Measured role score against a 0.095 ceiling:
    ///
    /// | β    | score |
    /// |------|-------|
    /// | 10   | 0.057 |
    /// | 30   | 0.077 |
    /// | ≥100 | 0.091 |
    ///
    /// β=10 recovers 60% of the available signal; β=30 recovers 81%; the curve
    /// is flat from 100 up. **Values below [`MIN_CLEANUP_BETA`] are clamped to
    /// it**, and the temperature actually used is reported back in
    /// [`MultiRoleResults::cleanup_beta`] so the substitution is visible.
    ///
    /// Clamping rather than warning is deliberate. `heather_db` has no log
    /// sink, so a warning would go nowhere, whereas the returned β is in front
    /// of every caller. And β is a softmax temperature: it costs the same to
    /// evaluate at any value, so a soft β buys nothing and only degrades the
    /// ordering — unlike [`Off`](RoleCleanup::Off), which is a genuine
    /// accuracy-for-speed trade and is therefore honoured exactly as asked.
    Beta(f64),

    /// Score the raw recovered filler, crosstalk and all.
    ///
    /// Fast — no inverse transform and no attention read per (document, pair)
    /// — and it is the mode whose scores are identical to
    /// [`Collection::query_documents_scoped`]. It is measurably worse at
    /// ranking; choose it knowingly.
    Off,
}

/// One recalled document, with its per-criterion breakdown.
#[derive(Debug, Clone)]
pub struct MultiRoleHit {
    pub doc_id: u64,

    /// Plain full-bundle cosine against the recall query. **This is the only
    /// quantity the result ordering uses** — see
    /// [`Collection::query_documents_multi_role`].
    pub recall_score: f64,

    pub metadata: Vec<u8>,

    /// One similarity per requested `(role, filler)` pair, **in the caller's
    /// pair order**. Never aggregated, never weighted.
    pub role_scores: Vec<f64>,
}

/// The result of [`Collection::query_documents_multi_role`].
#[derive(Debug, Clone)]
pub struct MultiRoleResults {
    /// Cleanup temperature actually used: the MDL-selected β, the caller's
    /// override after clamping to [`MIN_CLEANUP_BETA`], or `None` when cleanup
    /// was off.
    pub cleanup_beta: Option<f64>,

    /// Descending by [`MultiRoleHit::recall_score`], truncated to `n`.
    pub hits: Vec<MultiRoleHit>,
}

/// Hopfield cleanup of `v` against the collection's codebook at inverse
/// temperature `beta`: `Σ softmax(β·v·Kᵢ + ln write_countᵢ)·Vᵢ` over the
/// top-k activated locations.
///
/// Same read rule as [`Collection::read_attention`] with `scale = beta`,
/// written against a borrowed `EAMInner` because the caller already holds the
/// read lock and re-entering it would risk deadlocking against a queued
/// writer.
fn cleanup_read(inner: &EAMInner, v: &[f64], beta: f64) -> Vec<f64> {
    let k = inner.config.k.min(inner.locations.len());
    let (indices, _sims) = read::activate_auto_full(
        v,
        &inner.locations,
        k,
        &inner.landmarks,
        &inner.id_lookup,
        &inner.address_matrix,
        inner.config.d,
    );
    let logits: Vec<f64> = indices
        .iter()
        .map(|&i| {
            beta * vec_ops::dot(v, &inner.locations[i].address)
                + inner.locations[i].write_count.max(0.0).ln()
        })
        .collect();
    let alpha = vec_ops::softmax(&logits, 1.0);
    let values: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| inner.locations[i].normalized_pattern())
        .collect();
    let value_refs: Vec<&[f64]> = values.iter().map(|x| x.as_slice()).collect();
    vec_ops::weighted_sum(&value_refs, &alpha)
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
    /// In-process mutation counter. Bumped on every write-locked mutation;
    /// lets snapshot→load_snapshot cycles detect concurrent writes. Not
    /// persisted — resets to 0 on load.
    pub(crate) version: u64,
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
                version: 0,
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
                version: 0,
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
        inner.version += 1;

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
        inner.version += 1;

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
        Ok(self.read_attention_traced(query, scale, exclude)?.result)
    }

    /// [`read_attention_ex`](Self::read_attention_ex) returning the contributors alongside the value.
    ///
    /// The value vector is a single weighted sum, so on its own it cannot say
    /// which stored locations produced it. The trace names them: each
    /// contributor carries the raw dot product `Q·Kᵢ` and the post-softmax
    /// weight actually applied to `Vᵢ`, sorted descending by weight. This is
    /// location-level provenance — mapping a location back to the documents
    /// indexed under it is the caller's job.
    pub fn read_attention_traced(
        &self,
        query: &[f64],
        scale: f64,
        exclude: Option<usize>,
    ) -> Result<AttentionTrace> {
        vec_ops::validate_vector(query)?;

        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        Self::check_query(&inner, query)?;

        Self::attention_locked(&inner, query, scale, exclude, None)
    }

    /// [`read_attention`](Self::read_attention) with a *set* of locations barred from answering.
    ///
    /// The single-`exclude` form withholds one engram — a row's own — for
    /// leave-one-out reads. This withholds a family of them, for the case where
    /// rows come in groups of near-duplicates (a doping series, repeated
    /// measurements of one subject): asking what a row resembles is only
    /// honest when its own cousins are not allowed to answer for it.
    ///
    /// Routes through the same activation and attention as every other read —
    /// the barred set is expressed as the allowed complement, so results are
    /// bit-identical to a read over a memory that never held those locations'
    /// competitors.
    pub fn read_attention_excluding(
        &self,
        query: &[f64],
        scale: f64,
        barred: &[usize],
    ) -> Result<AttentionTrace> {
        vec_ops::validate_vector(query)?;
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Self::check_query(&inner, query)?;

        let barred: std::collections::HashSet<usize> = barred.iter().copied().collect();
        let allowed: Vec<usize> = (0..inner.locations.len())
            .filter(|i| !barred.contains(i))
            .collect();
        if allowed.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }
        Self::attention_locked(&inner, query, scale, None, Some(&allowed))
    }

    /// Per-query dimension / non-empty checks, factored out so the single-query
    /// and batched entry points make exactly the same checks in the same order.
    fn check_query(inner: &EAMInner, query: &[f64]) -> Result<()> {
        if query.len() != inner.config.d {
            return Err(HeatherError::DimensionMismatch {
                expected: inner.config.d,
                got: query.len(),
            });
        }
        if inner.locations.is_empty() {
            return Err(HeatherError::EmptyMemory);
        }
        Ok(())
    }

    /// Activated top-k indices for one query under an already-held read lock.
    /// `allowed`, when supplied, restricts activation to that candidate set
    /// (`None` = unrestricted). Sorted descending by similarity; `exclude` is
    /// dropped and the list re-truncated to `config.k`, so an excluded
    /// location costs nothing.
    fn activate_locked(
        inner: &EAMInner,
        query: &[f64],
        exclude: Option<usize>,
        allowed: Option<&[usize]>,
    ) -> Vec<usize> {
        let want = inner.config.k + exclude.is_some() as usize;
        let k = want.min(inner.locations.len());

        let (mut indices, _sims) = match allowed {
            Some(allowed) => read::activate_subset(query, &inner.locations, allowed, k),
            None => read::activate_auto_full(
                query,
                &inner.locations,
                k,
                &inner.landmarks,
                &inner.id_lookup,
                &inner.address_matrix,
                inner.config.d,
            ),
        };
        if let Some(ex) = exclude {
            indices.retain(|&i| i != ex);
            indices.truncate(inner.config.k);
        }
        indices
    }

    /// The whole attention read for one query, under an already-held read
    /// lock. This is the single definition of the read: both
    /// [`read_attention_traced`](Self::read_attention_traced) and
    /// [`read_attention_excluding`](Self::read_attention_excluding) call it,
    /// so neither can drift from the single-query numbers.
    fn attention_locked(
        inner: &EAMInner,
        query: &[f64],
        scale: f64,
        exclude: Option<usize>,
        allowed: Option<&[usize]>,
    ) -> Result<AttentionTrace> {
        let indices = Self::activate_locked(inner, query, exclude, allowed);

        // Count-weighted raw-dot scores: a merged engram represents
        // write_count tokens, so it enters the softmax with that multiplicity
        // — αᵢ ∝ write_countᵢ · exp(scale · Q·Kᵢ), folded as the logit
        // scale·Q·Kᵢ + ln(write_countᵢ). With write_count == 1 everywhere this
        // is exactly softmax(Q·Kᵢ · scale); with merged engrams it reconstructs
        // the attention the un-merged tokens would have produced.
        let dots: Vec<f64> = indices
            .iter()
            .map(|&i| vec_ops::dot(query, &inner.locations[i].address))
            .collect();
        let logits: Vec<f64> = indices
            .iter()
            .zip(&dots)
            .map(|(&i, &dot)| scale * dot + inner.locations[i].write_count.max(0.0).ln())
            .collect();
        let alpha = vec_ops::softmax(&logits, 1.0);
        // raw values (V = counter / write_count), NO output normalization
        let values: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| inner.locations[i].normalized_pattern())
            .collect();
        let value_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
        let result = vec_ops::weighted_sum(&value_refs, &alpha);

        let mut contributors: Vec<AttentionContributor> = indices
            .iter()
            .zip(&dots)
            .zip(&alpha)
            .map(|((&id, &similarity), &weight)| AttentionContributor {
                id,
                similarity,
                weight,
            })
            .collect();
        contributors.sort_by(|a, b| b.weight.total_cmp(&a.weight));

        Ok(AttentionTrace {
            result,
            contributors,
        })
    }

    /// Batched [`read_attention_ex`](Self::read_attention_ex): one read lock
    /// for the whole sweep, queries fanned out across rayon's pool.
    ///
    /// The per-query work is unchanged — the same activation, the same
    /// count-weighted softmax, the same weighted sum, all through
    /// `attention_locked`. What disappears is the per-query overhead paid
    /// outside it: re-acquiring the collection lock and re-reaching for the
    /// landmark / id-lookup / address-matrix caches once per query instead of
    /// once per sweep. Results are element-for-element identical to calling
    /// the single-query form in a loop, in input order.
    ///
    /// `excludes`, when given, must have one entry per query — the location to
    /// withhold from that query's activated set (`None` = withhold nothing).
    /// An empty batch returns an empty vec without touching the lock.
    pub fn read_attention_batch(
        &self,
        queries: &[Vec<f64>],
        scale: f64,
        excludes: Option<&[Option<usize>]>,
    ) -> Result<Vec<Vec<f64>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        if excludes.is_some_and(|ex| ex.len() != queries.len()) {
            return Err(HeatherError::InvalidInput(format!(
                "excludes length {} does not match queries length {}",
                excludes.map_or(0, <[Option<usize>]>::len),
                queries.len()
            )));
        }

        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;

        queries
            .par_iter()
            .enumerate()
            .map(|(qi, query)| {
                vec_ops::validate_vector(query)?;
                Self::check_query(&inner, query)?;
                let exclude = excludes.and_then(|ex| ex[qi]);
                Ok(Self::attention_locked(&inner, query, scale, exclude, None)?.result)
            })
            .collect()
    }

    /// The stored location that answers for this query: the index of the
    /// highest-similarity activated location, the same one that heads
    /// [`read_attention_traced`](Self::read_attention_traced)'s activation.
    pub fn top_location(&self, query: &[f64]) -> Result<usize> {
        vec_ops::validate_vector(query)?;
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Self::check_query(&inner, query)?;
        Self::top_location_locked(&inner, query)
    }

    /// Batched [`top_location`](Self::top_location) — one lock, one fan-out.
    /// Same indices, same order, as calling the single-query form per query.
    pub fn top_locations_batch(&self, queries: &[Vec<f64>]) -> Result<Vec<usize>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        queries
            .par_iter()
            .map(|query| {
                vec_ops::validate_vector(query)?;
                Self::check_query(&inner, query)?;
                Self::top_location_locked(&inner, query)
            })
            .collect()
    }

    fn top_location_locked(inner: &EAMInner, query: &[f64]) -> Result<usize> {
        let indices = Self::activate_locked(inner, query, None, None);
        indices.first().copied().ok_or(HeatherError::EmptyMemory)
    }

    /// Leave-one-out attention sweep: for each query, find the location that
    /// answers for it and read again with that location withheld.
    ///
    /// This is the engine half of held-out scoring — "what does the codebook
    /// say about this row when the row's own engram is not allowed to answer?"
    /// It is exactly `top_location` followed by
    /// [`read_attention_ex`](Self::read_attention_ex) with that index excluded,
    /// which is two lock acquisitions and two activations per row when done
    /// from outside; here it is one lock for the sweep. Comparing the returned
    /// value against the caller's own target — whatever "score" means to it —
    /// stays with the caller.
    pub fn read_attention_loo_batch(
        &self,
        queries: &[Vec<f64>],
        scale: f64,
    ) -> Result<Vec<Vec<f64>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        queries
            .par_iter()
            .map(|query| {
                vec_ops::validate_vector(query)?;
                Self::check_query(&inner, query)?;
                let own = Self::top_location_locked(&inner, query)?;
                Ok(Self::attention_locked(&inner, query, scale, Some(own), None)?.result)
            })
            .collect()
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
        let (trace, beta) = self.read_attention_mdl_traced(query)?;
        Ok((trace.result, beta))
    }

    /// [`read_attention_mdl`](Self::read_attention_mdl) returning the contributors alongside the value
    /// and the self-selected β.
    ///
    /// Same trace as [`read_attention_traced`](Self::read_attention_traced) —
    /// each contributor carries the raw dot product `Q·Kᵢ` and the post-softmax
    /// weight applied to `Vᵢ`, sorted descending by weight — except that the
    /// softmax runs at the MDL-selected temperature rather than a caller's.
    /// The weights are what an abstention gate reads: a codebook that cannot
    /// resolve the query spreads them, and the spread is visible here without
    /// any threshold being imposed on the caller.
    pub fn read_attention_mdl_traced(&self, query: &[f64]) -> Result<(AttentionTrace, f64)> {
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
        let dots: Vec<f64> = indices
            .iter()
            .map(|&i| vec_ops::dot(query, &inner.locations[i].address))
            .collect();
        let logits: Vec<f64> = indices
            .iter()
            .zip(&dots)
            .map(|(&i, &dot)| beta * dot + inner.locations[i].write_count.max(0.0).ln())
            .collect();
        let alpha = vec_ops::softmax(&logits, 1.0);
        let values: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| inner.locations[i].normalized_pattern())
            .collect();
        let value_refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
        let result = vec_ops::weighted_sum(&value_refs, &alpha);

        let mut contributors: Vec<AttentionContributor> = indices
            .iter()
            .zip(&dots)
            .zip(&alpha)
            .map(|((&id, &similarity), &weight)| AttentionContributor {
                id,
                similarity,
                weight,
            })
            .collect();
        contributors.sort_by(|a, b| b.weight.total_cmp(&a.weight));

        Ok((
            AttentionTrace {
                result,
                contributors,
            },
            beta,
        ))
    }

    /// Run KNN merge to consolidate similar locations.
    /// Migrates document index posting lists from removed locations to survivors.
    pub fn merge(&self) -> Result<usize> {
        let mut inner = self.inner.write().map_err(|_| HeatherError::LockPoisoned)?;
        inner.version += 1;
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
        inner.version += 1;

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
        self.query_documents_scoped(query, n, None)
    }

    /// [`Self::query_documents`] with the scoring rule made explicit.
    ///
    /// `unbind_role`, when supplied, switches to **role-scoped scoring**: each
    /// candidate is unbound by that role vector and the recovered filler
    /// compared to `query` by cosine, instead of the stored superposition
    /// being compared whole. This removes a confound in the plain path — a
    /// document's score for one role is otherwise divided by its own norm,
    /// which grows with every other role it carries, so richly structured
    /// documents rank lower for no reason related to the query (measured
    /// correlation between filled-slot count and score: −0.35 to −0.42).
    ///
    /// The recovered filler carries crosstalk from the other roles. Callers
    /// wanting that removed should follow with a cleanup read (`attention`, or
    /// `attention/mdl` to let the engine pick the temperature); a sharp
    /// cleanup recovers most of the loss, a soft one destroys ranking.
    ///
    /// Candidate *retrieval* is unchanged — the posting-list activation still
    /// uses `query` directly, so the role only re-scores the set the index
    /// already produced. A document whose role-A filler matches but whose
    /// bundle is far from `query` may therefore not be a candidate at all.
    pub fn query_documents_scoped(
        &self,
        query: &[f64],
        n: usize,
        unbind_role: Option<&[f64]>,
    ) -> Result<Vec<(u64, f64, Vec<u8>)>> {
        vec_ops::validate_vector(query)?;
        if let Some(role) = unbind_role {
            vec_ops::validate_vector(role)?;
            let d = self
                .inner
                .read()
                .map_err(|_| HeatherError::LockPoisoned)?
                .config
                .d;
            if role.len() != d {
                return Err(HeatherError::DimensionMismatch {
                    expected: d,
                    got: role.len(),
                });
            }
        }

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
                // Role-scoped when a role is supplied: unbind the stored
                // superposition by it first, so the score reflects only the
                // role the query constrained rather than being diluted by
                // every other role the document carries.
                let sim = match unbind_role {
                    Some(role) => vec_ops::role_scoped_similarity(&vec, role, query),
                    None => vec_ops::cosine_similarity(query, &vec),
                };
                scored.push((doc_id, sim, meta));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    /// Score every recalled document against **several** `(role, filler)`
    /// pairs at once, returning one similarity per pair per document.
    ///
    /// This is the batched form of [`query_documents_scoped`]. That call takes
    /// one role and reuses `query` as the filler, so a consumer scoring a
    /// document against six weighted criteria pays six full engine reads and
    /// gets back only `(id, score, payload)` — no vector, nothing to rescore
    /// locally. Here each criterion carries its own filler, the conflation of
    /// query vector and filler is gone, and the per-criterion breakdown comes
    /// back in one read. That breakdown is the point: it is both the terms the
    /// consumer's weighted score is built from and the "why this matched"
    /// explanation a UI can render.
    ///
    /// # Ranking is plain bundle similarity, and that is a real limitation
    ///
    /// Recall and ranking use `query` alone, by ordinary full-bundle cosine,
    /// exactly as [`query_documents`] does. The pairs only *describe* the
    /// documents that survived; they do not steer retrieval and they are not
    /// combined into anything.
    ///
    /// This engine deliberately **does not accept per-criterion weights and
    /// does not aggregate the per-role scores into a single number.**
    /// Weighting and qualification are business rules owned by the consuming
    /// service; an associative-memory similarity is not a business match
    /// score, and folding one into the other here would bury a policy decision
    /// inside the storage layer.
    ///
    /// The consequence is unavoidable and callers must plan for it: a document
    /// that would score well *once weighted* can be missed entirely, because
    /// nothing about the weighting reaches the recall stage — if its bundle is
    /// far from `query`, it is never a candidate and never appears in these
    /// results at all. **Over-fetch.** Ask for an `n` far larger than the
    /// number of results you intend to display, and let the consumer re-rank
    /// the wider set. There is no value of `n` that makes this exact.
    ///
    /// # Cleanup
    ///
    /// The filler recovered at a role carries crosstalk from every other role
    /// in the bundle. `cleanup` controls whether it is denoised against the
    /// collection's codebook before scoring; see [`RoleCleanup`], whose
    /// default is on. Cleanup costs an inverse transform and an attention read
    /// per (document, pair); with it off the score is computed entirely in the
    /// frequency domain.
    ///
    /// # Errors
    ///
    /// Every role and filler must match the collection dimension. An empty
    /// `pairs` slice is legal and degenerates to [`query_documents`] with an
    /// empty `role_scores` on each hit.
    ///
    /// [`query_documents`]: Collection::query_documents
    /// [`query_documents_scoped`]: Collection::query_documents_scoped
    pub fn query_documents_multi_role(
        &self,
        query: &[f64],
        n: usize,
        pairs: &[(&[f64], &[f64])],
        cleanup: RoleCleanup,
    ) -> Result<MultiRoleResults> {
        vec_ops::validate_vector(query)?;

        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        let d = inner.config.d;

        if query.len() != d {
            return Err(HeatherError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }
        for (role, filler) in pairs {
            vec_ops::validate_vector(role)?;
            vec_ops::validate_vector(filler)?;
            for v in [role, filler] {
                if v.len() != d {
                    return Err(HeatherError::DimensionMismatch {
                        expected: d,
                        got: v.len(),
                    });
                }
            }
        }

        if inner.locations.is_empty() {
            return Ok(MultiRoleResults {
                cleanup_beta: None,
                hits: Vec::new(),
            });
        }

        // Recall is identical to `query_documents`: activate `query` against
        // the hard locations and take the union of their posting lists. The
        // pairs play no part in choosing candidates — see the doc comment.
        let k = inner.config.k.min(inner.locations.len());
        let (indices, _sims) = read::activate_auto_full(
            query,
            &inner.locations,
            k,
            &inner.landmarks,
            &inner.id_lookup,
            &inner.address_matrix,
            d,
        );

        // The cleanup temperature is a property of the codebook's geometry,
        // not of any one recovered filler, so it is selected once per query
        // over the keys this query activated rather than per (document, pair).
        let cleanup_beta = match cleanup {
            RoleCleanup::Off => None,
            RoleCleanup::Mdl => {
                let keys: Vec<Vec<f64>> = indices
                    .iter()
                    .map(|&i| vec_ops::normalize(&inner.locations[i].address))
                    .collect();
                Some(select_beta_mdl(&keys, d, inner.config.beta))
            }
            RoleCleanup::Beta(beta) => Some(beta.max(MIN_CLEANUP_BETA)),
        };

        // Forward-transform each role and filler once for the whole query.
        // Per candidate this leaves one forward transform and, per pair, an
        // O(d) conjugate product — instead of two forward transforms per
        // (document, pair).
        let role_spectra: Vec<Vec<Complex<f64>>> = pairs
            .iter()
            .map(|(r, _)| vec_ops::rfft_forward(r))
            .collect();
        let filler_spectra: Vec<Vec<Complex<f64>>> = if cleanup_beta.is_some() {
            // The cleanup path compares in the time domain, so the filler
            // spectra would never be read.
            Vec::new()
        } else {
            pairs
                .iter()
                .map(|(_, f)| vec_ops::rfft_forward(f))
                .collect()
        };
        let filler_norms: Vec<f64> = pairs.iter().map(|(_, f)| vec_ops::l2_norm(f)).collect();

        let mut candidate_ids = std::collections::HashSet::new();
        let rtxn = self.store.read_txn()?;
        for &idx in &indices {
            let loc_id = inner.locations[idx].id.0;
            let doc_ids =
                self.store
                    .get_doc_ids_for_location_txn(&rtxn, self.collection_id, loc_id)?;
            candidate_ids.extend(doc_ids);
        }

        let mut hits: Vec<MultiRoleHit> = Vec::with_capacity(candidate_ids.len());
        for doc_id in candidate_ids {
            let Some(data) = self
                .store
                .get_document_txn(&rtxn, self.collection_id, doc_id)?
            else {
                continue;
            };
            let (vec, metadata): (Vec<f64>, Vec<u8>) = bincode::deserialize(&data)?;
            if vec.len() != d {
                continue;
            }

            let doc_spectrum = vec_ops::rfft_forward(&vec);
            let role_scores: Vec<f64> = pairs
                .iter()
                .enumerate()
                .map(|(i, (_, filler))| match cleanup_beta {
                    // Cleanup needs the recovered filler as a vector, so the
                    // inverse transform comes back.
                    Some(beta) => {
                        let recovered =
                            vec_ops::unbind_by_spectra(&doc_spectrum, &role_spectra[i], d);
                        let cleaned = cleanup_read(&inner, &recovered, beta);
                        vec_ops::cosine_similarity(&cleaned, filler)
                    }
                    None => vec_ops::role_scoped_similarity_spectra(
                        &doc_spectrum,
                        &role_spectra[i],
                        &filler_spectra[i],
                        filler_norms[i],
                        d,
                    ),
                })
                .collect();

            hits.push(MultiRoleHit {
                doc_id,
                recall_score: vec_ops::cosine_similarity(query, &vec),
                metadata,
                role_scores,
            });
        }

        hits.sort_by(|a, b| {
            b.recall_score
                .partial_cmp(&a.recall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(n);
        Ok(MultiRoleResults { cleanup_beta, hits })
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

    /// Like [`snapshot`](Self::snapshot), but also returns the mutation
    /// version, for handing to [`load_snapshot_checked`](Self::load_snapshot_checked).
    pub fn snapshot_versioned(&self) -> Result<(Vec<HardLocation>, EAMConfig, u64)> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Ok((inner.locations.clone(), inner.config.clone(), inner.version))
    }

    /// Current mutation version. Bumped on every write-locked mutation
    /// (write, merge, load_snapshot). In-process only — resets on load.
    pub fn version(&self) -> Result<u64> {
        let inner = self.inner.read().map_err(|_| HeatherError::LockPoisoned)?;
        Ok(inner.version)
    }

    /// Replace this collection's in-memory EAM state from a snapshot.
    /// Flushes the new state to persistent storage atomically.
    /// Resets next_id to max(location_ids) + 1.
    pub fn load_snapshot(&self, locations: Vec<HardLocation>, config: EAMConfig) -> Result<()> {
        self.load_snapshot_checked(locations, config, None)
    }

    /// [`load_snapshot`](Self::load_snapshot) with optimistic concurrency:
    /// when `expected_version` is `Some`, the replace is refused with
    /// [`HeatherError::Conflict`] if the collection was mutated since that
    /// version was observed (via [`snapshot_versioned`](Self::snapshot_versioned)
    /// or [`version`](Self::version)). Prevents a snapshot→compute→load
    /// cycle from silently discarding concurrent writes.
    pub fn load_snapshot_checked(
        &self,
        locations: Vec<HardLocation>,
        config: EAMConfig,
        expected_version: Option<u64>,
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
        if let Some(expected) = expected_version
            && inner.version != expected
        {
            return Err(HeatherError::Conflict(format!(
                "collection '{}' was modified concurrently (version {} != expected {}); retry",
                self.name, inner.version, expected
            )));
        }
        inner.version += 1;

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

    /// Naive circular convolution — the bind the test corpus is built with.
    /// Deliberately not the FFT path, so the read is checked against an
    /// independent definition of binding rather than against itself.
    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = a.len();
        let mut out = vec![0.0; d];
        for (k, o) in out.iter_mut().enumerate() {
            *o = (0..d).map(|i| a[i] * b[(k + d - i) % d]).sum();
        }
        vec_ops::normalize(&out)
    }

    /// End-to-end analogue of the `vec_ops` unit test: two documents agree on
    /// role A, one carries two further roles the query never mentions. Through
    /// the real posting-list read path, full-bundle scoring must penalise the
    /// richer document and role-scoped scoring must penalise it less.
    #[test]
    fn query_documents_scoped_reduces_the_richness_penalty() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let store = Arc::new(Store::open(dir.path(), 256).unwrap());
        let mut txn = store.write_txn().unwrap();
        let col_id = store.create_collection(&mut txn, "bids").unwrap();
        txn.commit().unwrap();

        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 8;
        config.k = 8; // every location activates, so both docs are candidates
        let col = Collection::new(col_id, "bids".into(), store, &config).unwrap();

        let mut rng = rand::thread_rng();
        let role_a = vec_ops::random_unit_vector(d, &mut rng);
        let role_b = vec_ops::random_unit_vector(d, &mut rng);
        let role_c = vec_ops::random_unit_vector(d, &mut rng);
        let filler = vec_ops::random_unit_vector(d, &mut rng);

        let sparse = convolve(&role_a, &filler);
        let bound_b = convolve(&role_b, &vec_ops::random_unit_vector(d, &mut rng));
        let bound_c = convolve(&role_c, &vec_ops::random_unit_vector(d, &mut rng));
        let rich = vec_ops::normalize(
            &sparse
                .iter()
                .zip(&bound_b)
                .zip(&bound_c)
                .map(|((x, y), z)| x + y + z)
                .collect::<Vec<f64>>(),
        );

        let sparse_id = col.write_with_metadata(&sparse, b"{}").unwrap();
        let rich_id = col.write_with_metadata(&rich, b"{}").unwrap();

        let score_of = |rs: &[(u64, f64, Vec<u8>)], id: u64| {
            rs.iter()
                .find(|(i, _, _)| *i == id)
                .unwrap_or_else(|| panic!("document {id} was not a candidate"))
                .1
        };

        // Full-bundle: the query is the whole bound pair.
        let bundle_query = convolve(&role_a, &filler);
        let plain = col.query_documents(&bundle_query, 10).unwrap();
        let plain_penalty = score_of(&plain, rich_id) / score_of(&plain, sparse_id);

        // Role-scoped: the query is the bare filler, plus the role to unbind by.
        let scoped = col
            .query_documents_scoped(&filler, 10, Some(&role_a))
            .unwrap();
        let scoped_penalty = score_of(&scoped, rich_id) / score_of(&scoped, sparse_id);

        assert!(
            scoped_penalty > plain_penalty,
            "role-scoped must penalise the richer document less: \
             scoped {scoped_penalty:.3} vs full-bundle {plain_penalty:.3}"
        );
    }

    #[test]
    fn query_documents_scoped_rejects_a_role_of_the_wrong_dimension() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let col = Collection::new(col_id, "test".into(), store, &test_config()).unwrap();
        let mut rng = rand::thread_rng();
        let query = vec_ops::random_unit_vector(16, &mut rng);

        let err = col
            .query_documents_scoped(&query, 5, Some(&[0.5; 8]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                HeatherError::DimensionMismatch {
                    expected: 16,
                    got: 8
                }
            ),
            "got {err:?}"
        );
    }

    // --- Multi-role scoped queries ---------------------------------------

    /// Four documents over three roles, each taking a different filler at each
    /// role from a shared pool, written through the real write path so recall
    /// runs over real posting lists. `k` is set above the location count so
    /// every document is a candidate — these tests are about the *scores*, and
    /// letting recall drop documents would make them intermittent.
    ///
    struct MultiRoleCorpus {
        col: Collection,
        roles: Vec<Vec<f64>>,
        fillers: Vec<Vec<f64>>,
        /// The bundle a caller looking for `roles[i] ↦ fillers[i]` would send.
        query: Vec<f64>,
    }

    fn multi_role_corpus(dir: &TempDir, d: usize) -> MultiRoleCorpus {
        let store = Arc::new(Store::open(dir.path(), 256).unwrap());
        let mut txn = store.write_txn().unwrap();
        let col_id = store.create_collection(&mut txn, "bids").unwrap();
        txn.commit().unwrap();

        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 8;
        config.k = 8;
        let col = Collection::new(col_id, "bids".into(), store, &config).unwrap();

        let mut rng = rand::thread_rng();
        let roles: Vec<Vec<f64>> = (0..3)
            .map(|_| vec_ops::random_unit_vector(d, &mut rng))
            .collect();
        let fillers: Vec<Vec<f64>> = (0..4)
            .map(|_| vec_ops::random_unit_vector(d, &mut rng))
            .collect();

        for j in 0..4 {
            let bundle = bundle_for(&roles, &fillers, j);
            col.write_with_metadata(&bundle, b"{}").unwrap();
        }

        let query = bundle_for(&roles, &fillers, 0);
        MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        }
    }

    /// Document `j`'s bundle: `roles[i]` bound to `fillers[(i + j) % 4]`.
    /// Document 0 is the one the corpus query is built from.
    fn bundle_for(roles: &[Vec<f64>], fillers: &[Vec<f64>], j: usize) -> Vec<f64> {
        let mut bundle = vec![0.0; roles[0].len()];
        for (i, role) in roles.iter().enumerate() {
            vec_ops::add_scaled(&mut bundle, &convolve(role, &fillers[(i + j) % 4]), 1.0);
        }
        vec_ops::normalize(&bundle)
    }

    fn pairs_of<'a>(roles: &'a [Vec<f64>], fillers: &'a [Vec<f64>]) -> Vec<(&'a [f64], &'a [f64])> {
        roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| (r.as_slice(), f.as_slice()))
            .collect()
    }

    /// **The load-bearing test.** The whole change is a batching optimisation,
    /// so a multi-role call must be observationally identical to N separate
    /// single-role calls: same documents, same scores, per pair.
    ///
    /// Compared with cleanup off, because that is the mode
    /// `query_documents_scoped` implements — it has no cleanup to compare to.
    /// Tolerance is 1e-12 rather than exact equality because the cleanup-off
    /// path scores in the frequency domain; the measured worst deviation here
    /// is well inside that (see `vec_ops::role_tests`).
    ///
    /// The test discriminates on every axis it is meant to: mis-ordering the
    /// pairs, applying cleanup, or getting the Parseval packing wrong each move
    /// the scores by ~1e-3 or more, a billion times the tolerance. The two
    /// spread assertions rule out the degenerate way this could pass — all
    /// scores collapsing to the same value would satisfy the comparison
    /// vacuously.
    #[test]
    fn multi_role_scores_match_n_separate_single_role_calls() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        } = multi_role_corpus(&dir, d);
        let pairs = pairs_of(&roles, &fillers);

        let multi = col
            .query_documents_multi_role(&query, 100, &pairs, RoleCleanup::Off)
            .unwrap();
        assert_eq!(multi.cleanup_beta, None, "cleanup was off");
        assert_eq!(multi.hits.len(), 4, "every document must be a candidate");

        for hit in &multi.hits {
            assert_eq!(hit.role_scores.len(), pairs.len());
        }
        // Not vacuous: every pair must separate the documents. A batching bug
        // that returned a constant — zeros, or the same score everywhere —
        // would satisfy the equality below trivially, and this rules it out.
        for i in 0..pairs.len() {
            let per_doc: Vec<f64> = multi.hits.iter().map(|h| h.role_scores[i]).collect();
            let spread = per_doc.iter().cloned().fold(f64::MIN, f64::max)
                - per_doc.iter().cloned().fold(f64::MAX, f64::min);
            assert!(
                spread > 0.2,
                "pair {i} must discriminate between documents: {per_doc:?}"
            );
        }

        for (i, (role, filler)) in pairs.iter().enumerate() {
            let single = col.query_documents_scoped(filler, 100, Some(role)).unwrap();
            let mut compared = 0;
            for hit in &multi.hits {
                let (_, single_score, _) = single
                    .iter()
                    .find(|(id, _, _)| *id == hit.doc_id)
                    .unwrap_or_else(|| panic!("doc {} missing from single-role call", hit.doc_id));
                let delta = (hit.role_scores[i] - single_score).abs();
                assert!(
                    delta < 1e-12,
                    "pair {i}, doc {}: batched {} vs single-role {single_score}, delta {delta:e}",
                    hit.doc_id,
                    hit.role_scores[i]
                );
                compared += 1;
            }
            assert_eq!(compared, 4, "pair {i}: nothing was actually compared");
        }
    }

    /// `role_scores[i]` must belong to `pairs[i]`. Reversing the pair list must
    /// reverse the breakdown and change nothing else — an implementation that
    /// scored by hash order, or that sorted pairs, would fail this.
    #[test]
    fn multi_role_preserves_pair_order() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        } = multi_role_corpus(&dir, d);
        let forward = pairs_of(&roles, &fillers);
        let reversed: Vec<(&[f64], &[f64])> = forward.iter().rev().copied().collect();

        let a = col
            .query_documents_multi_role(&query, 100, &forward, RoleCleanup::Off)
            .unwrap();
        let b = col
            .query_documents_multi_role(&query, 100, &reversed, RoleCleanup::Off)
            .unwrap();

        for (ha, hb) in a.hits.iter().zip(&b.hits) {
            assert_eq!(
                ha.doc_id, hb.doc_id,
                "ordering must not depend on the pairs"
            );
            let flipped: Vec<f64> = hb.role_scores.iter().rev().copied().collect();
            assert_eq!(ha.role_scores, flipped, "doc {}", ha.doc_id);
            // The reversal is only observable if the entries differ.
            assert_ne!(
                ha.role_scores[0], ha.role_scores[2],
                "doc {}: pairs are indistinguishable, the check is vacuous",
                ha.doc_id
            );
        }
    }

    /// Ranking is plain bundle similarity and nothing else — the pairs must not
    /// reach it. Same query with and without pairs must produce the same
    /// documents in the same order with the same scores.
    #[test]
    fn multi_role_ranks_by_bundle_similarity_only() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            ..
        } = multi_role_corpus(&dir, d);
        let pairs = pairs_of(&roles, &fillers);
        // Recall a *different* document's bundle than the pairs describe, so
        // the bundle ordering and the pair-0 ordering genuinely disagree — with
        // both pointing the same way the assertion below proves nothing.
        let query = bundle_for(&roles, &fillers, 3);

        let plain = col.query_documents(&query, 100).unwrap();
        let multi = col
            .query_documents_multi_role(&query, 100, &pairs, RoleCleanup::Off)
            .unwrap();

        assert_eq!(plain.len(), multi.hits.len());
        for ((id, sim, _), hit) in plain.iter().zip(&multi.hits) {
            assert_eq!(*id, hit.doc_id);
            assert_eq!(*sim, hit.recall_score);
        }
        // A pair whose score ordering contradicts the bundle ordering must not
        // move anything: prove the two orderings really do disagree somewhere.
        let by_pair_0: Vec<f64> = multi.hits.iter().map(|h| h.role_scores[0]).collect();
        assert!(
            by_pair_0.windows(2).any(|w| w[0] < w[1]),
            "pair 0 already agrees with the bundle order; the check is vacuous: {by_pair_0:?}"
        );
    }

    #[test]
    fn multi_role_handles_an_empty_pair_list() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus { col, query, .. } = multi_role_corpus(&dir, d);

        let out = col
            .query_documents_multi_role(&query, 100, &[], RoleCleanup::default())
            .unwrap();
        let plain = col.query_documents(&query, 100).unwrap();

        assert_eq!(out.hits.len(), plain.len());
        for (hit, (id, sim, _)) in out.hits.iter().zip(&plain) {
            assert_eq!(hit.doc_id, *id);
            assert_eq!(hit.recall_score, *sim);
            assert!(hit.role_scores.is_empty());
        }
    }

    #[test]
    fn multi_role_with_a_single_pair_equals_the_single_role_call() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        } = multi_role_corpus(&dir, d);
        let one: [(&[f64], &[f64]); 1] = [(roles[1].as_slice(), fillers[1].as_slice())];

        let multi = col
            .query_documents_multi_role(&query, 100, &one, RoleCleanup::Off)
            .unwrap();
        let single = col
            .query_documents_scoped(&fillers[1], 100, Some(&roles[1]))
            .unwrap();

        assert_eq!(multi.hits.len(), 4);
        for hit in &multi.hits {
            assert_eq!(hit.role_scores.len(), 1);
            let (_, expected, _) = single.iter().find(|(id, _, _)| *id == hit.doc_id).unwrap();
            assert!((hit.role_scores[0] - expected).abs() < 1e-12);
        }
    }

    /// Cleanup is on by default, off is an explicit choice, and the two do
    /// materially different work. The reported `cleanup_beta` is what makes the
    /// engine's choice of temperature visible to the caller.
    #[test]
    fn cleanup_is_on_by_default_and_reports_the_temperature_it_used() {
        assert_eq!(RoleCleanup::default(), RoleCleanup::Mdl);

        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        } = multi_role_corpus(&dir, d);
        let pairs = pairs_of(&roles, &fillers);

        let on = col
            .query_documents_multi_role(&query, 100, &pairs, RoleCleanup::default())
            .unwrap();
        let off = col
            .query_documents_multi_role(&query, 100, &pairs, RoleCleanup::Off)
            .unwrap();

        let beta = on.cleanup_beta.expect("the default must clean up");
        assert!(beta > 0.0, "MDL must select a positive temperature: {beta}");
        assert_eq!(off.cleanup_beta, None);

        // Both modes must still be usable reads — same documents, same order.
        let ids = |r: &MultiRoleResults| r.hits.iter().map(|h| h.doc_id).collect::<Vec<_>>();
        assert_eq!(ids(&on), ids(&off));

        let moved = on
            .hits
            .iter()
            .zip(&off.hits)
            .flat_map(|(a, b)| a.role_scores.iter().zip(&b.role_scores))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-3,
            "cleanup must actually change the scores, but the largest move was {moved:e}"
        );
    }

    /// A too-soft explicit beta is clamped to [`MIN_CLEANUP_BETA`] rather than
    /// silently degrading the ordering, and the clamp is reported.
    ///
    /// The second half is what stops this from being a test of nothing: if beta
    /// never reached the scoring, `Beta(10)` would equal `Beta(30)` for the
    /// wrong reason. A far sharper temperature must produce different scores.
    #[test]
    fn a_too_soft_cleanup_beta_is_clamped_and_the_clamp_is_reported() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            query,
        } = multi_role_corpus(&dir, d);
        let pairs = pairs_of(&roles, &fillers);
        let run = |c| {
            col.query_documents_multi_role(&query, 100, &pairs, c)
                .unwrap()
        };

        let soft = run(RoleCleanup::Beta(10.0));
        let floor = run(RoleCleanup::Beta(MIN_CLEANUP_BETA));
        let sharp = run(RoleCleanup::Beta(2000.0));

        assert_eq!(soft.cleanup_beta, Some(MIN_CLEANUP_BETA));
        assert_eq!(sharp.cleanup_beta, Some(2000.0));

        let scores = |r: &MultiRoleResults| {
            r.hits
                .iter()
                .flat_map(|h| h.role_scores.clone())
                .collect::<Vec<f64>>()
        };
        assert_eq!(
            scores(&soft),
            scores(&floor),
            "beta 10 must be run as beta {MIN_CLEANUP_BETA}"
        );

        let moved = scores(&floor)
            .iter()
            .zip(scores(&sharp))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            moved > 1e-6,
            "the cleanup temperature must reach the scoring, but beta {MIN_CLEANUP_BETA} \
             and beta 2000 agreed to {moved:e}"
        );
    }

    /// The mechanism behind the documented sharpness cliff: cleanup is a
    /// softmax over the codebook, so a sharper temperature resolves onto a
    /// single stored pattern while a soft one returns a blend. A soft beta does
    /// not fail loudly — it returns a plausible average — which is why the
    /// public API clamps instead of trusting the caller.
    #[test]
    fn sharper_cleanup_resolves_the_codebook_harder() {
        let d = 256;
        let dir = TempDir::new().unwrap();
        let MultiRoleCorpus {
            col,
            roles,
            fillers,
            ..
        } = multi_role_corpus(&dir, d);

        let inner = col.inner.read().unwrap();
        let doc = col.list_documents().unwrap()[0].1.clone();
        let recovered = vec_ops::unbind_by(&doc, &roles[0]);

        let nearest = |v: &[f64]| {
            inner
                .locations
                .iter()
                .map(|l| vec_ops::cosine_similarity(v, &l.normalized_pattern()))
                .fold(f64::MIN, f64::max)
        };
        let soft = nearest(&cleanup_read(&inner, &recovered, 1.0));
        let sharp = nearest(&cleanup_read(&inner, &recovered, 5000.0));

        assert!(
            sharp > soft + 1e-6,
            "a sharper cleanup must land closer to a stored pattern: \
             sharp {sharp:.4} vs soft {soft:.4}"
        );
        // Guard against the corpus being degenerate: the fillers must not all
        // be the same thing.
        assert!(
            vec_ops::cosine_similarity(&fillers[0], &fillers[1]).abs() < 0.5,
            "corpus fillers are not independent"
        );
    }

    #[test]
    fn multi_role_rejects_a_pair_of_the_wrong_dimension() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let col = Collection::new(col_id, "test".into(), store, &test_config()).unwrap();
        let mut rng = rand::thread_rng();
        let query = vec_ops::random_unit_vector(16, &mut rng);
        let good = vec_ops::random_unit_vector(16, &mut rng);

        for pairs in [
            vec![(&[0.5; 8][..], good.as_slice())],
            vec![(good.as_slice(), &[0.5; 8][..])],
        ] {
            let err = col
                .query_documents_multi_role(&query, 5, &pairs, RoleCleanup::Off)
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    HeatherError::DimensionMismatch {
                        expected: 16,
                        got: 8
                    }
                ),
                "got {err:?}"
            );
        }
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

    /// A collection with `n` written locations at dimension `d`, plus `m`
    /// random queries — enough locations that graph/SoA activation paths and
    /// the sequential/parallel split in `activate` both get exercised.
    fn batch_fixture(dir: &TempDir, d: usize, n: usize, m: usize) -> (Collection, Vec<Vec<f64>>) {
        let store = Arc::new(Store::open(dir.path(), 256).unwrap());
        let mut txn = store.write_txn().unwrap();
        let id = store.create_collection(&mut txn, "batch").unwrap();
        txn.commit().unwrap();

        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 64;
        config.k = 5;
        let col = Collection::new(id, "batch".into(), store, &config).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..n {
            col.write(&vec_ops::random_unit_vector(d, &mut rng))
                .unwrap();
        }
        let queries = (0..m)
            .map(|_| vec_ops::random_unit_vector(d, &mut rng))
            .collect();
        (col, queries)
    }

    /// The batch form is not "close to" the loop form, it *is* the loop form:
    /// every returned coordinate must compare bit-equal, with and without
    /// exclusions.
    #[test]
    fn read_attention_batch_matches_single_query_loop_exactly() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 40);
        let scale = 1.0 / (32f64).sqrt();

        let batched = col.read_attention_batch(&queries, scale, None).unwrap();
        assert_eq!(batched.len(), queries.len());
        for (q, got) in queries.iter().zip(&batched) {
            let want = col.read_attention_ex(q, scale, None).unwrap();
            assert_eq!(*got, want, "batched read diverged from single-query read");
        }

        // Exclusion path: withhold each query's own top location, plus a
        // location that is (almost certainly) not in the activated set, plus
        // no exclusion at all — all in one batch.
        let n = col.num_locations().unwrap();
        let excludes: Vec<Option<usize>> = queries
            .iter()
            .enumerate()
            .map(|(i, q)| match i % 3 {
                0 => None,
                1 => Some(col.top_location(q).unwrap()),
                _ => Some(n - 1 - (i % n)),
            })
            .collect();

        let batched = col
            .read_attention_batch(&queries, scale, Some(&excludes))
            .unwrap();
        for ((q, ex), got) in queries.iter().zip(&excludes).zip(&batched) {
            let want = col.read_attention_ex(q, scale, *ex).unwrap();
            assert_eq!(*got, want, "batched exclude read diverged");
        }
    }

    /// Excluding a location the query never activated must leave the read
    /// untouched — the same vector the unexcluded read returns.
    #[test]
    fn excluding_an_unactivated_location_changes_nothing() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 8);
        let scale = 0.7;

        for q in &queries {
            let activated = {
                let inner = col.inner.read().unwrap();
                Collection::activate_locked(&inner, q, None, None)
            };
            let n = col.num_locations().unwrap();
            let outsider = (0..n).find(|i| !activated.contains(i)).unwrap();

            let plain = col.read_attention_ex(q, scale, None).unwrap();
            let excluded = col.read_attention_ex(q, scale, Some(outsider)).unwrap();
            assert_eq!(plain, excluded);

            let batched = col
                .read_attention_batch(std::slice::from_ref(q), scale, Some(&[Some(outsider)]))
                .unwrap();
            assert_eq!(batched[0], plain);
        }
    }

    #[test]
    fn batch_edge_cases() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 4);
        let scale = 0.5;

        // Empty batch — no lock, no work, empty answer.
        assert!(
            col.read_attention_batch(&[], scale, None)
                .unwrap()
                .is_empty()
        );
        assert!(col.top_locations_batch(&[]).unwrap().is_empty());
        assert!(col.read_attention_loo_batch(&[], scale).unwrap().is_empty());

        // Single query — same as the single-query call.
        let one = std::slice::from_ref(&queries[0]);
        assert_eq!(
            col.read_attention_batch(one, scale, None).unwrap()[0],
            col.read_attention_ex(&queries[0], scale, None).unwrap()
        );
        assert_eq!(
            col.top_locations_batch(one).unwrap()[0],
            col.top_location(&queries[0]).unwrap()
        );

        // Mismatched excludes length is rejected, not silently zipped.
        assert!(matches!(
            col.read_attention_batch(&queries, scale, Some(&[None])),
            Err(HeatherError::InvalidInput(_))
        ));

        // Wrong dimension anywhere in the batch fails the batch.
        let bad = vec![vec![0.0; 8]];
        assert!(matches!(
            col.read_attention_batch(&bad, scale, None),
            Err(HeatherError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn top_locations_batch_matches_single_query_loop_exactly() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 40);

        let batched = col.top_locations_batch(&queries).unwrap();
        for (q, &got) in queries.iter().zip(&batched) {
            assert_eq!(got, col.top_location(q).unwrap());
        }
    }

    /// The leave-one-out sweep must equal the two-call sequence the caller
    /// would otherwise write by hand.
    #[test]
    fn loo_batch_matches_manual_two_call_sequence() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 24);
        let scale = 1.0 / (32f64).sqrt();

        let batched = col.read_attention_loo_batch(&queries, scale).unwrap();
        for (q, got) in queries.iter().zip(&batched) {
            let own = col.top_location(q).unwrap();
            let want = col.read_attention_ex(q, scale, Some(own)).unwrap();
            assert_eq!(*got, want);
        }
    }

    /// The trace must name the locations the value came from: the weights are
    /// the ones actually applied (they reproduce the value exactly), the
    /// similarities are raw dot products against the stored keys, and the list
    /// is ordered by weight so the top contributor is the first citation.
    #[test]
    fn attention_trace_names_its_contributors() {
        let dir = TempDir::new().unwrap();
        let (store, col_id) = setup_store(&dir);
        let mut config = test_config();
        config.l_0 = 0;
        config.k = 64; // every key participates
        config.competitive = false; // verbatim K→V store: one location per write
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

        let trace = col.read_attention_traced(&query, scale, None).unwrap();

        // The thin wrapper and the traced read agree on the value.
        let plain = col.read_attention(&query, scale).unwrap();
        assert_eq!(trace.result, plain);

        assert_eq!(trace.contributors.len(), 5, "every location participated");

        // Descending by weight, and the weights are a distribution.
        for w in trace.contributors.windows(2) {
            assert!(
                w[0].weight >= w[1].weight,
                "contributors must sort by weight"
            );
        }
        let total: f64 = trace.contributors.iter().map(|c| c.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "weights must sum to 1, got {total}"
        );

        let inner = col.inner.read().unwrap();
        for c in &trace.contributors {
            // `similarity` is the RAW dot product Q·Kᵢ — not a cosine, not βQ·Kᵢ.
            let raw = vec_ops::dot(&query, &inner.locations[c.id].address);
            assert!(
                (c.similarity - raw).abs() < 1e-12,
                "similarity {} != raw dot {raw}",
                c.similarity
            );
        }

        // The contributors fully account for the value: Σ weightᵢ · Vᵢ == result.
        let mut recon = vec![0.0; d];
        for c in &trace.contributors {
            let v = inner.locations[c.id].normalized_pattern();
            for (r, x) in recon.iter_mut().zip(&v) {
                *r += c.weight * x;
            }
        }
        for (a, b) in trace.result.iter().zip(&recon) {
            assert!((a - b).abs() < 1e-12, "result {a} != Σ weight·V {b}");
        }
    }

    /// `read_attention_excluding` over a barred set must equal the same read
    /// with the barred locations removed from the candidate pool entirely —
    /// not merely dropped post-hoc from an unrestricted top-k.
    #[test]
    fn read_attention_excluding_matches_a_memory_without_the_barred_set() {
        let dir = TempDir::new().unwrap();
        let (col, queries) = batch_fixture(&dir, 32, 120, 8);
        let scale = 0.6;

        for q in &queries {
            let n = col.num_locations().unwrap();
            // Bar a handful of locations, including at least one likely
            // activated (the query's own top match).
            let top = col.top_location(q).unwrap();
            let barred: Vec<usize> = std::iter::once(top).chain([1usize, 2, 3]).collect();

            let trace = col.read_attention_excluding(q, scale, &barred).unwrap();
            for c in &trace.contributors {
                assert!(
                    !barred.contains(&c.id),
                    "barred location {} answered anyway",
                    c.id
                );
            }
            assert!((n) > barred.len(), "sanity: barred is a strict subset");
        }

        // Barring everything is an empty memory, not a panic.
        let all: Vec<usize> = (0..col.num_locations().unwrap()).collect();
        assert!(matches!(
            col.read_attention_excluding(&queries[0], scale, &all),
            Err(HeatherError::EmptyMemory)
        ));
    }

    /// [`read_attention_mdl_traced`] must name its contributors the same way
    /// [`attention_trace_names_its_contributors`] checks for the fixed-β read:
    /// weights sum to 1, sorted descending, similarities are raw dot products,
    /// and the contributors fully reconstruct the returned value — all at the
    /// self-selected β rather than a caller-supplied one.
    #[test]
    fn mdl_attention_trace_names_its_contributors() {
        let d = 16;
        let dir = TempDir::new().unwrap();
        let (col, keys) = separated_collection(&dir, d);
        // Ambiguous query: nearly equidistant to both keys, so several
        // locations carry real weight and the sum has something to check.
        let query = vec_ops::normalize(
            &keys[0]
                .iter()
                .zip(&keys[1])
                .map(|(a, b)| 0.51 * a + 0.49 * b)
                .collect::<Vec<_>>(),
        );

        let (trace, beta) = col.read_attention_mdl_traced(&query).unwrap();
        let (plain, plain_beta) = col.read_attention_mdl(&query).unwrap();
        assert_eq!(trace.result, plain, "wrapper and traced read agree");
        assert_eq!(beta, plain_beta);
        assert!(beta.is_finite() && beta > 0.0, "β* must be usable: {beta}");

        assert_eq!(trace.contributors.len(), keys.len());
        for w in trace.contributors.windows(2) {
            assert!(
                w[0].weight >= w[1].weight,
                "contributors must sort by weight"
            );
        }
        let total: f64 = trace.contributors.iter().map(|c| c.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "weights must sum to 1, got {total}"
        );

        let inner = col.inner.read().unwrap();
        for c in &trace.contributors {
            // `similarity` is the RAW dot product Q·Kᵢ — not scaled by β*.
            let raw = vec_ops::dot(&query, &inner.locations[c.id].address);
            assert!(
                (c.similarity - raw).abs() < 1e-12,
                "similarity {} != raw dot {raw}",
                c.similarity
            );
        }

        // Σ weightᵢ · Vᵢ == result, at the self-selected temperature.
        let mut recon = vec![0.0; d];
        for c in &trace.contributors {
            let v = inner.locations[c.id].normalized_pattern();
            for (r, x) in recon.iter_mut().zip(&v) {
                *r += c.weight * x;
            }
        }
        for (a, b) in trace.result.iter().zip(&recon) {
            assert!((a - b).abs() < 1e-12, "result {a} != Σ weight·V {b}");
        }
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
                .map(|l| scale * vec_ops::dot(&query, &l.address) + l.write_count.max(0.0).ln())
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
