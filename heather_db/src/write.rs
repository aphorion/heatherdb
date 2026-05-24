use rand::Rng;

use crate::config::EAMConfig;
use crate::error::HeatherError;
use crate::location::{HardLocation, LocationId};
use crate::read;
use crate::vec_ops;

/// Result of a write operation, including any new locations created by splits.
pub struct WriteResult {
    /// Indices of locations that were modified (in the caller's location vec)
    pub modified_indices: Vec<usize>,
    /// New locations created by novelty or overload splits
    pub new_locations: Vec<HardLocation>,
    /// Updated learning rate after decay
    pub eta: f64,
}

/// Auxiliary input-space signals injected into a substrate write at the
/// existing parameter-update sites. Lets callers shape per-write learning
/// without forking the substrate's update rule. See RFC 0005.
///
/// All fields default to `None`. A write with `WriteAux::default()` is
/// bit-identical to the plain write path.
#[derive(Debug, Default, Clone)]
pub struct WriteAux {
    /// Additional input-space signal added to the counter update. Each
    /// activated location j accumulates `w_j · counter_delta` in addition
    /// to the standard `w_j · input` term.
    ///
    /// Use case: a downstream task supplies its own gradient on each
    /// location's counter (e.g. supervised target deltas, reconstruction
    /// residual from an external decoder).
    ///
    /// Length must equal `config.d` if `Some`.
    pub counter_delta: Option<Vec<f64>>,

    /// Additional input-space signal added to the address migration step.
    /// Each activated location j shifts by `lr_j · address_delta` in
    /// addition to the standard `lr_j · (input − address_j)` migration.
    /// Addresses are renormalised after.
    ///
    /// Use case: shape *which* attractor an input goes to (e.g. pull
    /// addresses toward task-specific regions).
    ///
    /// Length must equal `config.d` if `Some`.
    pub address_delta: Option<Vec<f64>>,
}

impl WriteAux {
    /// Returns `Ok(())` if every populated delta has the expected length.
    pub fn validate(&self, expected_d: usize) -> Result<(), HeatherError> {
        if let Some(c) = &self.counter_delta {
            if c.len() != expected_d {
                return Err(HeatherError::DimensionMismatch {
                    expected: expected_d,
                    got: c.len(),
                });
            }
        }
        if let Some(a) = &self.address_delta {
            if a.len() != expected_d {
                return Err(HeatherError::DimensionMismatch {
                    expected: expected_d,
                    got: a.len(),
                });
            }
        }
        Ok(())
    }

    /// True when no field is populated. The aux path can then take the
    /// fast path that's bit-identical to the plain write.
    pub fn is_empty(&self) -> bool {
        self.counter_delta.is_none() && self.address_delta.is_none()
    }
}

/// Per-location activation reported by a write. The location index refers
/// to the caller's location vec at the time of the write (before any
/// splits in this same write).
#[derive(Debug, Clone, Copy)]
pub struct WriteActivation {
    pub location_index: usize,
    /// Activation weight w_j ∈ [0, 1]. The same weight applied to the
    /// counter update.
    pub weight: f64,
}

/// Diagnostic information returned by `adaptive_write_with_aux`. Lets
/// callers compute their own external gradients (e.g. encoder chain rule)
/// without having to re-run the activation pass.
#[derive(Debug, Default)]
pub struct WriteDiagnostics {
    /// Activations recorded *before* this write's parameter updates were
    /// applied. Empty when no locations activated.
    pub activations: Vec<WriteActivation>,
    /// Weighted soft-prediction Σ_j w_j · address_j, in input space.
    /// Lets a caller compute their own residual without re-doing
    /// activation. Empty when no locations activated.
    pub prediction: Vec<f64>,
    /// `input − prediction`. Empty when no locations activated.
    pub residual: Vec<f64>,
}

/// Execute the three-phase adaptive write pipeline.
///
/// Phase 1 — Select: k-NN activation, weight computation, conscience winner.
/// Phase 2 — Update: counter accumulation + competitive address migration in one pass.
/// Phase 3 — Regulate: topology maintenance (novelty/overload split, local dedup).
///
/// Convenience wrapper around [`adaptive_write_with_aux`] for the
/// no-aux case. Diagnostics are discarded.
pub fn adaptive_write(
    input: &[f64],
    locations: &mut [HardLocation],
    config: &EAMConfig,
    eta: f64,
    next_id: &mut u64,
    rng: &mut impl Rng,
    landmarks: &[usize],
    id_lookup: &[u32],
) -> WriteResult {
    let (result, _diagnostics) = adaptive_write_with_aux(
        input, locations, config, eta, next_id, rng, landmarks, id_lookup, None,
    );
    result
}

/// As [`adaptive_write`], but accepts optional auxiliary input-space
/// signals that are folded into the counter and address updates at the
/// existing update sites, and returns diagnostics (activation pattern,
/// soft prediction, residual) suitable for downstream chain-rule
/// gradient computation.
///
/// When `aux` is `None` or all fields are `None`, the produced state
/// changes are bit-identical to [`adaptive_write`].
pub fn adaptive_write_with_aux(
    input: &[f64],
    locations: &mut [HardLocation],
    config: &EAMConfig,
    eta: f64,
    next_id: &mut u64,
    rng: &mut impl Rng,
    landmarks: &[usize],
    id_lookup: &[u32],
    aux: Option<&WriteAux>,
) -> (WriteResult, WriteDiagnostics) {
    let k = config.k.min(locations.len());

    // ── Phase 1: Select ─────────────────────────────────────────────
    // k-nearest via graph search (falls back to SoA/brute force when graph is young).
    let (indices, sims) = read::activate_auto(input, locations, k, landmarks, id_lookup);

    if indices.is_empty() {
        return (
            WriteResult {
                modified_indices: vec![],
                new_locations: vec![],
                eta,
            },
            WriteDiagnostics::default(),
        );
    }

    let max_sim = sims[0]; // sorted descending

    // Activation weights: w_j = max(S(x, a_j), 0) / max_j S(x, a_j)
    let weights: Vec<f64> = if max_sim > 1e-12 {
        sims.iter().map(|s| (s.max(0.0)) / max_sim).collect()
    } else {
        vec![1.0 / k as f64; sims.len()]
    };

    // Winner via conscience: S_eff = S(x, a_j) - γ · n_j / Σn_i
    let total_writes: f64 = indices.iter().map(|&i| locations[i].write_count).sum();
    let winner_local = if total_writes > 1e-12 {
        indices
            .iter()
            .enumerate()
            .map(|(local_idx, &global_idx)| {
                let s_eff = sims[local_idx]
                    - config.gamma * locations[global_idx].write_count / total_writes;
                (local_idx, s_eff)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    } else {
        0
    };
    let winner_global = indices[winner_local];

    // ── Phase 2: Update ─────────────────────────────────────────────
    // Single pass over activated set: counter accumulation + address migration.
    let eta_winner = eta / (1.0 + locations[winner_global].write_count / config.tau_damp);
    let eta_neighbor = eta_winner * 0.1;

    // Pre-compute weighted soft-prediction (Σ w_j · address_j) for the
    // diagnostics. Done before parameter updates so the returned residual
    // reflects the substrate's state at the moment of activation.
    let mut diagnostics = WriteDiagnostics::default();
    if aux.is_some() {
        let d = config.d;
        let mut prediction = vec![0.0_f64; d];
        let mut activations = Vec::with_capacity(indices.len());
        for (local_idx, &global_idx) in indices.iter().enumerate() {
            let w = weights[local_idx];
            vec_ops::add_scaled(&mut prediction, &locations[global_idx].address, w);
            activations.push(WriteActivation {
                location_index: global_idx,
                weight: w,
            });
        }
        let residual: Vec<f64> = input
            .iter()
            .zip(prediction.iter())
            .map(|(x, p)| x - p)
            .collect();
        diagnostics.activations = activations;
        diagnostics.prediction = prediction;
        diagnostics.residual = residual;
    }

    let counter_delta = aux.and_then(|a| a.counter_delta.as_deref());
    let address_delta = aux.and_then(|a| a.address_delta.as_deref());

    for (local_idx, &global_idx) in indices.iter().enumerate() {
        let loc = &mut locations[global_idx];

        // Counter accumulation: c_j += w_j · (x + counter_delta), n_j += w_j
        let w = weights[local_idx];
        vec_ops::add_scaled(&mut loc.counter, input, w);
        if let Some(cd) = counter_delta {
            vec_ops::add_scaled(&mut loc.counter, cd, w);
        }
        loc.write_count += w;

        // Address migration: winner at η_eff, neighbors at 10%
        let lr = if local_idx == winner_local {
            eta_winner
        } else {
            eta_neighbor
        };
        let diff: Vec<f64> = input
            .iter()
            .zip(loc.address.iter())
            .map(|(x, a)| x - a)
            .collect();
        vec_ops::add_scaled(&mut loc.address, &diff, lr);
        if let Some(ad) = address_delta {
            vec_ops::add_scaled(&mut loc.address, ad, lr);
        }
        loc.address = vec_ops::normalize(&loc.address);
    }

    // ── Neighbor graph update ─────────────────────────────────────
    // Each activated location learns about its co-activated peers.
    // Cost: O(k²) — negligible compared to the O(LD) activation step.
    // Adaptive capacity: max(D, k·ln(L)) — D for local Voronoi coverage,
    // k·ln(L) for global navigability in O(log L) hops.
    let nb_cap = config.adaptive_neighbor_cap(locations.len());
    if nb_cap > 0 {
        // Collect IDs of all activated locations
        let activated_ids: Vec<u64> = indices.iter().map(|&i| locations[i].id.0).collect();

        for (local_idx, &global_idx) in indices.iter().enumerate() {
            let loc = &mut locations[global_idx];
            let my_id = activated_ids[local_idx];

            for &peer_id in &activated_ids {
                if peer_id == my_id {
                    continue;
                }
                if !loc.neighbors.contains(&peer_id) {
                    loc.neighbors.push(peer_id);
                }
            }

            // Prune if over capacity: keep the most recently added (tail)
            if loc.neighbors.len() > nb_cap {
                let excess = loc.neighbors.len() - nb_cap;
                loc.neighbors.drain(0..excess);
            }
        }
    }

    // Decay learning rate
    let eta_new = (eta * config.lambda).max(config.eta_min);

    // ── Phase 3: Regulate ───────────────────────────────────────────
    // Topology maintenance: novelty split OR overload split, then local dedup.
    let mut new_locations = Vec::new();

    if max_sim < config.tau_split {
        // Novelty split: no location is close enough — spawn at input
        let id = LocationId(*next_id);
        *next_id += 1;
        let addr = vec_ops::normalize(input);
        let mut new_loc = HardLocation::new(id, addr);
        new_loc.counter = input.to_vec();
        new_loc.write_count = 1.0;
        // Novelty child: seed neighbors from the activated set
        if nb_cap > 0 {
            let activated_ids: Vec<u64> = indices.iter().map(|&i| locations[i].id.0).collect();
            new_loc.neighbors = activated_ids;
            if new_loc.neighbors.len() > nb_cap {
                new_loc.neighbors.truncate(nb_cap);
            }
        }
        new_locations.push(new_loc);
    }

    if locations[winner_global].write_count > config.tau_overload {
        // Overload split: winner is saturated — spawn perturbed neighbor
        let id = LocationId(*next_id);
        *next_id += 1;

        let perturbation = vec_ops::random_unit_vector(config.d, rng);
        let mut new_addr = locations[winner_global].address.clone();
        vec_ops::add_scaled(&mut new_addr, &perturbation, 0.1);
        let new_addr = vec_ops::normalize(&new_addr);

        let mut new_loc = HardLocation::new(id, new_addr);

        // Split counters: new location gets half
        new_loc.counter = locations[winner_global]
            .counter
            .iter()
            .map(|c| c * 0.5)
            .collect();
        new_loc.write_count = locations[winner_global].write_count * 0.5;

        // Winner keeps the other half
        for c in locations[winner_global].counter.iter_mut() {
            *c *= 0.5;
        }
        locations[winner_global].write_count *= 0.5;

        // Child inherits parent's neighbors; they become mutual neighbors
        if nb_cap > 0 {
            let parent_id = locations[winner_global].id.0;
            let child_id = id.0;
            new_loc.neighbors = locations[winner_global].neighbors.clone();
            // Add parent as neighbor of child
            if !new_loc.neighbors.contains(&parent_id) {
                new_loc.neighbors.push(parent_id);
            }
            if new_loc.neighbors.len() > nb_cap {
                new_loc.neighbors.truncate(nb_cap);
            }
            // Add child as neighbor of parent
            if !locations[winner_global].neighbors.contains(&child_id) {
                locations[winner_global].neighbors.push(child_id);
                if locations[winner_global].neighbors.len() > nb_cap {
                    locations[winner_global].neighbors.drain(0..1);
                }
            }
        }

        new_locations.push(new_loc);
    }

    // Local dedup: if both splits fired and produced near-identical locations,
    // merge them. Does not merge back into existing (overload children are
    // intentionally near the winner; novelty children are far from everything).
    if new_locations.len() > 1 {
        dedup_new_locations(&mut new_locations, config);
    }

    (
        WriteResult {
            modified_indices: indices,
            new_locations,
            eta: eta_new,
        },
        diagnostics,
    )
}

/// Dedup among newly created locations: if two are within tau_merge, merge.
fn dedup_new_locations(new_locations: &mut Vec<HardLocation>, config: &EAMConfig) {
    let mut absorbed = vec![false; new_locations.len()];

    for i in 0..new_locations.len() {
        if absorbed[i] {
            continue;
        }
        for j in (i + 1)..new_locations.len() {
            if absorbed[j] {
                continue;
            }
            let sim =
                vec_ops::cosine_similarity(&new_locations[i].address, &new_locations[j].address);
            if sim > config.tau_merge {
                let j_counter = new_locations[j].counter.clone();
                let j_wc = new_locations[j].write_count;
                let total_wc = new_locations[i].write_count + j_wc;
                if total_wc > 1e-12 {
                    let wi = new_locations[i].write_count / total_wc;
                    let wj = j_wc / total_wc;
                    let new_addr: Vec<f64> = new_locations[i]
                        .address
                        .iter()
                        .zip(new_locations[j].address.iter())
                        .map(|(a, b)| a * wi + b * wj)
                        .collect();
                    new_locations[i].address = vec_ops::normalize(&new_addr);
                }
                vec_ops::add_scaled(&mut new_locations[i].counter, &j_counter, 1.0);
                new_locations[i].write_count += j_wc;
                absorbed[j] = true;
            }
        }
    }

    if absorbed.iter().any(|&a| a) {
        let mut idx = 0;
        new_locations.retain(|_| {
            let keep = !absorbed[idx];
            idx += 1;
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_write() {
        let config = EAMConfig::new(3).unwrap();
        let mut locations = vec![
            HardLocation::new(LocationId(0), vec_ops::normalize(&[1.0, 0.0, 0.0])),
            HardLocation::new(LocationId(1), vec_ops::normalize(&[0.0, 1.0, 0.0])),
            HardLocation::new(LocationId(2), vec_ops::normalize(&[0.0, 0.0, 1.0])),
        ];
        let input = vec_ops::normalize(&[1.0, 0.1, 0.0]);
        let mut next_id = 3u64;
        let mut rng = rand::thread_rng();

        let result = adaptive_write(&input, &mut locations, &config, 0.01, &mut next_id, &mut rng, &[], &[]);
        assert!(!result.modified_indices.is_empty());
        // Location 0 should have received the most weight
        assert!(locations[0].write_count > locations[1].write_count);
    }

    #[test]
    fn test_novelty_split() {
        let mut config = EAMConfig::new(3).unwrap();
        config.tau_split = 0.99; // very high threshold -> always split
        let mut locations = vec![
            HardLocation::new(LocationId(0), vec_ops::normalize(&[1.0, 0.0, 0.0])),
        ];
        let input = vec_ops::normalize(&[0.0, 1.0, 0.0]); // very different
        let mut next_id = 1u64;
        let mut rng = rand::thread_rng();

        let result = adaptive_write(&input, &mut locations, &mut config, 0.01, &mut next_id, &mut rng, &[], &[]);
        // Should have created at least one new location (novelty split)
        assert!(!result.new_locations.is_empty());
    }

    /// Reproducible state for the aux-equivalence tests. Returns
    /// (locations, input, config, eta, next_id) ready for one write.
    fn fixture() -> (Vec<HardLocation>, Vec<f64>, EAMConfig, f64, u64) {
        let config = EAMConfig::new(3).unwrap();
        let locations = vec![
            HardLocation::new(LocationId(0), vec_ops::normalize(&[1.0, 0.0, 0.0])),
            HardLocation::new(LocationId(1), vec_ops::normalize(&[0.0, 1.0, 0.0])),
            HardLocation::new(LocationId(2), vec_ops::normalize(&[0.0, 0.0, 1.0])),
        ];
        let input = vec_ops::normalize(&[1.0, 0.1, 0.05]);
        (locations, input, config, 0.05, 3)
    }

    /// Deterministic RNG so the two paths produce identical results.
    fn det_rng() -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(0xC0FFEE)
    }

    #[test]
    fn aux_empty_is_bit_identical_to_plain_write() {
        let (mut locs_a, input, config, eta, mut nid_a) = fixture();
        let (mut locs_b, _, _, _, mut nid_b) = fixture();

        let r_plain = adaptive_write(
            &input, &mut locs_a, &config, eta, &mut nid_a, &mut det_rng(), &[], &[],
        );
        let aux = WriteAux::default();
        let (r_aux, diag) = adaptive_write_with_aux(
            &input, &mut locs_b, &config, eta, &mut nid_b, &mut det_rng(), &[], &[], Some(&aux),
        );

        assert_eq!(r_plain.modified_indices, r_aux.modified_indices);
        assert_eq!(r_plain.eta, r_aux.eta);
        assert_eq!(r_plain.new_locations.len(), r_aux.new_locations.len());
        assert_eq!(nid_a, nid_b);

        // Counters and addresses must match exactly.
        for (a, b) in locs_a.iter().zip(locs_b.iter()) {
            assert_eq!(a.counter, b.counter, "counter divergence with empty aux");
            assert_eq!(a.address, b.address, "address divergence with empty aux");
            assert_eq!(a.write_count, b.write_count);
        }

        // Diagnostics are populated even for an empty WriteAux (the
        // caller asked for them by going through the with_aux path).
        assert_eq!(diag.activations.len(), r_aux.modified_indices.len());
        assert_eq!(diag.prediction.len(), input.len());
        assert_eq!(diag.residual.len(), input.len());
    }

    #[test]
    fn aux_with_none_arg_is_bit_identical_to_plain_write() {
        let (mut locs_a, input, config, eta, mut nid_a) = fixture();
        let (mut locs_b, _, _, _, mut nid_b) = fixture();

        let r_plain = adaptive_write(
            &input, &mut locs_a, &config, eta, &mut nid_a, &mut det_rng(), &[], &[],
        );
        let (r_aux, diag) = adaptive_write_with_aux(
            &input, &mut locs_b, &config, eta, &mut nid_b, &mut det_rng(), &[], &[], None,
        );

        assert_eq!(r_plain.modified_indices, r_aux.modified_indices);
        for (a, b) in locs_a.iter().zip(locs_b.iter()) {
            assert_eq!(a.counter, b.counter);
            assert_eq!(a.address, b.address);
        }
        // None signals "no diagnostics needed" → empty diagnostics.
        assert!(diag.activations.is_empty());
        assert!(diag.prediction.is_empty());
        assert!(diag.residual.is_empty());
    }

    #[test]
    fn counter_delta_shifts_counters_by_weight() {
        // Compare two writes that differ only by counter_delta = [c, c, c].
        // Each activated location's counter should differ by exactly
        // w_j · counter_delta between the two runs.
        let (mut locs_a, input, config, eta, mut nid_a) = fixture();
        let (mut locs_b, _, _, _, mut nid_b) = fixture();

        let delta = vec![0.5, -0.25, 0.125];

        let (r_plain, diag_plain) = adaptive_write_with_aux(
            &input, &mut locs_a, &config, eta, &mut nid_a, &mut det_rng(),
            &[], &[], Some(&WriteAux::default()),
        );
        let aux = WriteAux { counter_delta: Some(delta.clone()), address_delta: None };
        let (r_aug, _) = adaptive_write_with_aux(
            &input, &mut locs_b, &config, eta, &mut nid_b, &mut det_rng(),
            &[], &[], Some(&aux),
        );

        assert_eq!(r_plain.modified_indices, r_aug.modified_indices);
        for act in &diag_plain.activations {
            let j = act.location_index;
            let w = act.weight;
            for k in 0..3 {
                let observed = locs_b[j].counter[k] - locs_a[j].counter[k];
                let expected = w * delta[k];
                assert!(
                    (observed - expected).abs() < 1e-12,
                    "counter[{j}][{k}] diff: got {observed}, want {expected}"
                );
            }
        }
    }

    #[test]
    fn address_delta_shifts_addresses_before_normalisation() {
        // For address, the post-normalise comparison is more involved
        // because the renormalisation is non-linear. We instead verify
        // that the *pre-normalisation* address would differ by lr · delta.
        // We do this by reconstructing the unnormalised intermediate:
        // since both runs apply identical (input − address) migration on
        // initially-identical addresses, the difference between the
        // post-normalisation addresses must come solely from the aux term.
        let (mut locs_a, input, config, eta, mut nid_a) = fixture();
        let (mut locs_b, _, _, _, mut nid_b) = fixture();

        let delta = vec![0.0, 0.0, 0.1]; // small push along z-axis
        let (r_plain, diag) = adaptive_write_with_aux(
            &input, &mut locs_a, &config, eta, &mut nid_a, &mut det_rng(),
            &[], &[], Some(&WriteAux::default()),
        );
        let aux = WriteAux { counter_delta: None, address_delta: Some(delta.clone()) };
        let (r_aug, _) = adaptive_write_with_aux(
            &input, &mut locs_b, &config, eta, &mut nid_b, &mut det_rng(),
            &[], &[], Some(&aux),
        );

        assert_eq!(r_plain.modified_indices, r_aug.modified_indices);
        // For each activated location, the address with aux should have
        // (slightly) more z-component than the address without aux,
        // proportional to the local learning rate.
        for act in &diag.activations {
            let j = act.location_index;
            let z_plain = locs_a[j].address[2];
            let z_aug = locs_b[j].address[2];
            assert!(
                z_aug > z_plain,
                "address[{j}].z should grow with positive aux delta on z; \
                 plain={z_plain}, aug={z_aug}"
            );
        }
    }

    #[test]
    fn aux_dimension_mismatch_rejected() {
        let aux = WriteAux {
            counter_delta: Some(vec![1.0; 5]),
            address_delta: None,
        };
        let err = aux.validate(3).expect_err("expected dimension mismatch");
        match err {
            HeatherError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 5);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn diagnostics_residual_is_input_minus_prediction() {
        let (mut locs, input, config, eta, mut nid) = fixture();
        let (_r, diag) = adaptive_write_with_aux(
            &input, &mut locs, &config, eta, &mut nid, &mut det_rng(),
            &[], &[], Some(&WriteAux::default()),
        );
        assert_eq!(diag.prediction.len(), input.len());
        for k in 0..input.len() {
            let expected = input[k] - diag.prediction[k];
            assert!((diag.residual[k] - expected).abs() < 1e-12);
        }
    }
}
