//! Energy-correct EAM composition via attractor discovery.
//!
//! Composes two EAMs by discovering the attractor landscape of E_A + E_B:
//! 1. Probe the combined energy from boundary slerps, parent addresses, random vectors
//! 2. Descend E_A + E_B via combined Hopfield dynamics (separate softmax per parent)
//! 3. Deduplicate convergent attractors
//! 4. Build C at discovered basins using confidence-weighted reads from both parents
//! 5. Dream-consolidate to deepen basins
//! 6. Validate via fingerprint comparison

use rand::Rng;
use rand_distr::StandardNormal;

use heather_db::merge::knn_merge;
use heather_db::read;
use heather_db::vec_ops;
use heather_db::{HardLocation, LocationId, EAMConfig};

use crate::error::{AlgebraError, Result};
use crate::snapshot::EAMSnapshot;

// ============================================================
// Public types
// ============================================================

/// Parameters controlling the composition algorithm.
#[derive(Debug, Clone)]
pub struct ComposeParams {
    // --- Step 2: Probe generation & descent ---
    /// Slerp interpolation points per cross-pair (default 4 → t=0.2,0.4,0.6,0.8).
    pub boundary_slerp_steps: usize,
    /// Max cross-pairs for boundary probes. 0 = use all (default 0).
    pub max_boundary_pairs: usize,
    /// Number of random unit vector probes (default 50).
    pub num_random_probes: usize,
    /// Max iterations for combined Hopfield descent (default 50).
    pub descent_t_max: usize,
    /// Convergence epsilon (default 1e-6).
    pub descent_epsilon: f64,
    /// Softmax inverse temperature for descent (default 5.0).
    pub descent_beta: f64,
    /// k-nearest for activation in descent (default 20).
    pub descent_k: usize,

    // --- Step 3: Deduplication ---
    /// Cosine similarity threshold for attractor deduplication (default 0.95).
    pub dedup_threshold: f64,

    // --- Step 4: Location building ---
    /// Locations to create per discovered attractor (default 10).
    /// Samples around each basin center to provide dense coverage.
    pub locations_per_attractor: usize,
    /// Spread of sub-samples around each attractor (default 0.1).
    pub attractor_sample_spread: f64,

    // --- Step 5: Dreaming ---
    /// Number of annealing rounds (default 10).
    pub dream_rounds: usize,
    /// Initial noise standard deviation (default 0.1).
    pub dream_noise_initial: f64,
    /// Noise decay per round (default 0.8).
    pub dream_noise_decay: f64,
    /// Write weight during dreaming (default 0.2).
    pub dream_write_weight: f64,

    // --- Step 6: Validation ---
    /// Max fingerprint similarity before C is "just a copy" (default 0.9).
    pub validation_max_similarity: f64,
    /// Min self-stability for C's fingerprint (default 0.99).
    pub validation_min_stability: f64,
}

impl Default for ComposeParams {
    fn default() -> Self {
        ComposeParams {
            boundary_slerp_steps: 4,
            max_boundary_pairs: 0,
            num_random_probes: 50,
            descent_t_max: 50,
            descent_epsilon: 1e-6,
            descent_beta: 5.0,
            descent_k: 20,
            dedup_threshold: 0.95,
            locations_per_attractor: 10,
            attractor_sample_spread: 0.1,
            dream_rounds: 10,
            dream_noise_initial: 0.1,
            dream_noise_decay: 0.8,
            dream_write_weight: 0.2,
            validation_max_similarity: 0.9,
            validation_min_stability: 0.99,
        }
    }
}

/// Diagnostic information from the composition process.
#[derive(Debug, Clone)]
pub struct ComposeDiagnostics {
    pub pool_size: usize,
    pub boundary_probes_count: usize,
    pub parent_probes_count: usize,
    pub random_probes_count: usize,
    pub raw_attractors_count: usize,
    pub unique_attractors_count: usize,
    pub locations_built: usize,
    pub post_dream_locations: usize,
    pub fingerprint_sim_a: f64,
    pub fingerprint_sim_b: f64,
    pub fingerprint_sim_naive: f64,
    pub fingerprint_stability: f64,
    pub validation_passed: bool,
}

/// Result of composing two EAMs.
#[derive(Debug, Clone)]
pub struct ComposeResult {
    pub snapshot: EAMSnapshot,
    pub diagnostics: ComposeDiagnostics,
}

// ============================================================
// Public entry point
// ============================================================

/// Compose two EAMs by discovering the attractor landscape of E_A + E_B.
///
/// This is NOT pairwise combination. This algorithm:
/// 1. Discovers true attractors of the combined energy via Hopfield descent
/// 2. Deduplicates convergent attractors
/// 3. Builds C's locations at those attractors using confidence-weighted reads
/// 4. Consolidates via annealed dreaming
/// 5. Validates the result via fingerprints
pub fn compose(
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    params: &ComposeParams,
) -> Result<ComposeResult> {
    if a.dim() != b.dim() {
        return Err(AlgebraError::DimensionMismatch {
            left: a.dim(),
            right: b.dim(),
        });
    }
    if a.locations.is_empty() || b.locations.is_empty() {
        return Err(AlgebraError::EmptySnapshot);
    }

    let mut rng = rand::thread_rng();

    // Step 1: Pool (conceptual)
    let pool_size = a.num_locations() + b.num_locations();

    // Step 2: Discover attractors from three probe sources
    let boundary = boundary_probes(a, b, params);
    let boundary_count = boundary.len();
    let parents = parent_probes(a, b);
    let parent_count = parents.len();
    let randoms = random_probes(a.dim(), params.num_random_probes, &mut rng);
    let random_count = randoms.len();

    let all_probes: Vec<Vec<f64>> = boundary
        .into_iter()
        .chain(parents)
        .chain(randoms)
        .collect();

    let raw_attractors: Vec<Vec<f64>> = all_probes
        .iter()
        .map(|probe| combined_descent(probe, &a.locations, &b.locations, params))
        .collect();
    let raw_count = raw_attractors.len();

    // Step 3: Deduplicate
    let unique_attractors = deduplicate_attractors(&raw_attractors, params.dedup_threshold);
    let unique_count = unique_attractors.len();

    // Step 4: Build C = pooled parents + hybrid attractor locations
    // Include parent locations (preserve trained knowledge)
    // plus new locations at discovered hybrid attractors.
    let mut hybrid_locs = build_locations_at_attractors(&unique_attractors, a, b, params, &mut rng);
    let built_count = hybrid_locs.len();

    // Step 5: Dream-consolidate only the hybrid locations.
    // Parent locations are already well-trained — dreaming would corrupt them.
    // Dreaming uses C (hybrid + parents) for reads, but only writes to hybrids.
    if params.dream_rounds > 0 && !hybrid_locs.is_empty() {
        // Build a temporary read-only view with all locations for Hopfield reads
        let mut all_for_read = a.locations.clone();
        all_for_read.extend(b.locations.clone());
        all_for_read.extend(hybrid_locs.clone());
        let read_config = EAMConfig {
            k: params.descent_k.min(all_for_read.len()),
            beta: params.descent_beta,
            ..a.config.clone()
        };

        // Dream only on hybrid locations, reading from the full set
        dream_consolidate_hybrid(
            &mut hybrid_locs,
            &all_for_read,
            &read_config,
            params,
            &mut rng,
        );
    }

    let mut all_locations = a.locations.clone();
    all_locations.extend(b.locations.clone());
    all_locations.extend(hybrid_locs);

    let config = a.config.clone();
    let mut c = EAMSnapshot {
        locations: all_locations,
        config,
    };
    c.reindex();
    let post_dream = c.num_locations();

    // Step 6: Validate via fingerprints
    let fp_c = compute_fingerprint(&c);
    let fp_a = compute_fingerprint(a);
    let fp_b = compute_fingerprint(b);

    let (sim_a, sim_b, sim_naive, stability) = match (&fp_c, &fp_a, &fp_b) {
        (Some(fc), Some(fa), Some(fb)) => {
            let sim_a = vec_ops::cosine_similarity(fc, fa);
            let sim_b = vec_ops::cosine_similarity(fc, fb);
            let naive_mid: Vec<f64> = fa.iter().zip(fb.iter()).map(|(a, b)| a + b).collect();
            let naive_mid = vec_ops::normalize(&naive_mid);
            let sim_naive = vec_ops::cosine_similarity(fc, &naive_mid);
            let re_read = read::hopfield_iter(fc, &c.locations, &c.config)
                .unwrap_or_else(|_| fc.clone());
            let stability = vec_ops::cosine_similarity(fc, &re_read);
            (sim_a, sim_b, sim_naive, stability)
        }
        _ => (0.0, 0.0, 0.0, 0.0),
    };

    let validation_passed = sim_a < params.validation_max_similarity
        && sim_b < params.validation_max_similarity
        && sim_naive < params.validation_max_similarity
        && stability > params.validation_min_stability;

    let diagnostics = ComposeDiagnostics {
        pool_size,
        boundary_probes_count: boundary_count,
        parent_probes_count: parent_count,
        random_probes_count: random_count,
        raw_attractors_count: raw_count,
        unique_attractors_count: unique_count,
        locations_built: built_count,
        post_dream_locations: post_dream,
        fingerprint_sim_a: sim_a,
        fingerprint_sim_b: sim_b,
        fingerprint_sim_naive: sim_naive,
        fingerprint_stability: stability,
        validation_passed,
    };

    Ok(ComposeResult {
        snapshot: c,
        diagnostics,
    })
}

// ============================================================
// Step 2: Probe generators
// ============================================================

/// Spherical linear interpolation between two unit vectors.
fn slerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    let dot = vec_ops::dot(a, b).clamp(-1.0, 1.0);
    // Fallback to normalized lerp for near-parallel vectors
    if dot.abs() > 0.9995 {
        let lerp: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| ai * (1.0 - t) + bi * t)
            .collect();
        return vec_ops::normalize(&lerp);
    }
    let omega = dot.acos();
    let sin_omega = omega.sin();
    let s0 = ((1.0 - t) * omega).sin() / sin_omega;
    let s1 = (t * omega).sin() / sin_omega;
    let result: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| s0 * ai + s1 * bi)
        .collect();
    vec_ops::normalize(&result)
}

/// Generate boundary probes: slerp between nearest cross-pairs of A and B.
fn boundary_probes(a: &EAMSnapshot, b: &EAMSnapshot, params: &ComposeParams) -> Vec<Vec<f64>> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    // For each A location, find nearest B location
    for (i, loc_a) in a.locations.iter().enumerate() {
        let mut best_j = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (j, loc_b) in b.locations.iter().enumerate() {
            let sim = vec_ops::cosine_similarity(&loc_a.address, &loc_b.address);
            if sim > best_sim {
                best_sim = sim;
                best_j = j;
            }
        }
        pairs.push((i, best_j));
    }

    // For each B location, find nearest A location
    for (j, loc_b) in b.locations.iter().enumerate() {
        let mut best_i = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, loc_a) in a.locations.iter().enumerate() {
            let sim = vec_ops::cosine_similarity(&loc_b.address, &loc_a.address);
            if sim > best_sim {
                best_sim = sim;
                best_i = i;
            }
        }
        pairs.push((best_i, j));
    }

    // Deduplicate
    pairs.sort();
    pairs.dedup();

    if params.max_boundary_pairs > 0 && pairs.len() > params.max_boundary_pairs {
        pairs.truncate(params.max_boundary_pairs);
    }

    let steps = params.boundary_slerp_steps;
    let mut probes = Vec::new();
    for (i, j) in &pairs {
        let addr_a = &a.locations[*i].address;
        let addr_b = &b.locations[*j].address;
        for s in 1..=steps {
            let t = s as f64 / (steps as f64 + 1.0);
            probes.push(slerp(addr_a, addr_b, t));
        }
    }
    probes
}

/// Parent probes: every location address from both A and B.
fn parent_probes(a: &EAMSnapshot, b: &EAMSnapshot) -> Vec<Vec<f64>> {
    a.locations
        .iter()
        .chain(b.locations.iter())
        .map(|loc| loc.address.clone())
        .collect()
}

/// Random unit vector probes.
fn random_probes(d: usize, count: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    (0..count)
        .map(|_| vec_ops::random_unit_vector(d, rng))
        .collect()
}

// ============================================================
// Step 2 core: Combined Hopfield descent on E_A + E_B
// ============================================================

/// Single-step softmax read over a set of locations.
/// Returns (reconstruction, confidence) where confidence = max activation similarity.
fn read_with_confidence(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
    beta: f64,
) -> (Vec<f64>, f64) {
    if locations.is_empty() {
        return (vec![0.0; query.len()], 0.0);
    }
    let k = k.min(locations.len());
    let (indices, sims) = read::activate(query, locations, k);

    let confidence = sims.first().copied().unwrap_or(0.0);

    let weights = vec_ops::softmax(&sims, beta);
    let patterns: Vec<Vec<f64>> = indices.iter().map(|&i| locations[i].unit_pattern()).collect();
    let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
    let recon = vec_ops::weighted_sum(&pattern_refs, &weights);

    (recon, confidence)
}

/// Run cross-weighted combined descent from a starting probe.
///
/// At each iteration:
/// - Read from A → (recon_a, conf_a)
/// - Read from B → (recon_b, conf_b)
/// - xi_new = normalize(conf_b * recon_a + conf_a * recon_b)
///
/// Cross-weighting: each parent's reconstruction is scaled by the OTHER
/// parent's confidence. This creates stable attractors where BOTH parents
/// contribute meaningfully, rather than at individual parent basins.
///
/// When xi is near A only: conf_a >> conf_b, so recon_b (the B signal)
/// gets amplified and pulls xi toward the hybrid zone. The converse holds
/// near B. The equilibrium is where conf_a ≈ conf_b — the mutual
/// resonance region between both memories.
fn combined_descent(
    start: &[f64],
    locations_a: &[HardLocation],
    locations_b: &[HardLocation],
    params: &ComposeParams,
) -> Vec<f64> {
    let mut xi = vec_ops::normalize(start);
    let k = params.descent_k;
    let beta = params.descent_beta;

    for _t in 0..params.descent_t_max {
        let (recon_a, conf_a) = read_with_confidence(&xi, locations_a, k, beta);
        let (recon_b, conf_b) = read_with_confidence(&xi, locations_b, k, beta);

        // Cross-weight: scale each reconstruction by the OTHER's confidence
        let combined: Vec<f64> = recon_a
            .iter()
            .zip(recon_b.iter())
            .map(|(a, b)| conf_b * a + conf_a * b)
            .collect();
        let xi_new = vec_ops::normalize(&combined);

        let sim = vec_ops::cosine_similarity(&xi, &xi_new);
        xi = xi_new;
        if sim > 1.0 - params.descent_epsilon {
            break;
        }
    }
    xi
}

// ============================================================
// Step 3: Deduplication
// ============================================================

/// Greedy deduplication: keep an attractor only if it's below threshold
/// similarity to all already-kept attractors.
fn deduplicate_attractors(attractors: &[Vec<f64>], threshold: f64) -> Vec<Vec<f64>> {
    let mut unique: Vec<Vec<f64>> = Vec::new();
    for attractor in attractors {
        let is_dup = unique
            .iter()
            .any(|u| vec_ops::cosine_similarity(attractor, u) > threshold);
        if !is_dup {
            unique.push(attractor.clone());
        }
    }
    unique
}

// ============================================================
// Step 4: Build C at discovered attractors
// ============================================================

/// Create a location at a query point using confidence-weighted reads from both parents.
fn build_one_location(
    id: u64,
    query: &[f64],
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    k: usize,
    beta: f64,
) -> HardLocation {
    let (recon_a, conf_a) = read_with_confidence(query, &a.locations, k, beta);
    let (recon_b, conf_b) = read_with_confidence(query, &b.locations, k, beta);

    let counter: Vec<f64> = recon_a
        .iter()
        .zip(recon_b.iter())
        .map(|(ra, rb)| conf_a * ra + conf_b * rb)
        .collect();
    let write_count = conf_a + conf_b;

    let mut loc = HardLocation::new(LocationId(id), query.to_vec());
    loc.counter = counter;
    loc.write_count = write_count;
    loc
}

/// For each unique attractor, create a cluster of locations by sampling around
/// the basin center. Each sub-sample gets its own confidence-weighted read
/// from both parents, capturing local structure within the basin.
fn build_locations_at_attractors(
    attractors: &[Vec<f64>],
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    params: &ComposeParams,
    rng: &mut impl Rng,
) -> Vec<HardLocation> {
    let k = params.descent_k;
    let beta = params.descent_beta;
    let n_per = params.locations_per_attractor;
    let spread = params.attractor_sample_spread;
    let mut locations = Vec::new();
    let mut next_id: u64 = 0;

    for attractor in attractors {
        // Center location at the attractor itself
        locations.push(build_one_location(next_id, attractor, a, b, k, beta));
        next_id += 1;

        // Sub-sample locations around the attractor
        for _ in 1..n_per {
            let perturbed: Vec<f64> = attractor
                .iter()
                .map(|&v| v + rng.sample::<f64, _>(StandardNormal) * spread)
                .collect();
            let perturbed = vec_ops::normalize(&perturbed);
            locations.push(build_one_location(next_id, &perturbed, a, b, k, beta));
            next_id += 1;
        }
    }

    locations
}

// ============================================================
// Step 5: Dream consolidation
// ============================================================

/// Simplified soft-write: activate k-nearest, distribute weighted pattern.
/// No competitive learning, no splits — just reinforcement.
fn local_soft_write(
    pattern: &[f64],
    locations: &mut [HardLocation],
    config: &EAMConfig,
    weight: f64,
) {
    if locations.is_empty() {
        return;
    }
    let k = config.k.min(locations.len());
    let (indices, sims) = read::activate(pattern, locations, k);
    let weights = vec_ops::softmax(&sims, config.beta);
    for (local_idx, &global_idx) in indices.iter().enumerate() {
        let w = weights[local_idx] * weight;
        vec_ops::add_scaled(&mut locations[global_idx].counter, pattern, w);
        locations[global_idx].write_count += w;
    }
}

/// Dream-consolidate only the hybrid locations, reading from the full set (parents + hybrids).
/// This deepens hybrid basins without corrupting well-trained parent patterns.
fn dream_consolidate_hybrid(
    hybrid_locs: &mut Vec<HardLocation>,
    all_locs_for_read: &[HardLocation],
    read_config: &EAMConfig,
    params: &ComposeParams,
    rng: &mut impl Rng,
) {
    let mut noise = params.dream_noise_initial;
    let mut write_config = read_config.clone();
    write_config.k = write_config.k.min(hybrid_locs.len());

    for _round in 0..params.dream_rounds {
        if hybrid_locs.is_empty() {
            break;
        }

        let queries: Vec<Vec<f64>> = hybrid_locs
            .iter()
            .map(|loc| {
                let perturbed: Vec<f64> = loc
                    .address
                    .iter()
                    .map(|&a| a + rng.sample::<f64, _>(StandardNormal) * noise)
                    .collect();
                vec_ops::normalize(&perturbed)
            })
            .collect();

        // Read from full set (parents + hybrids), write only to hybrids
        for query in &queries {
            let recon = read::hopfield_iter(query, all_locs_for_read, read_config)
                .unwrap_or_else(|_| query.clone());
            local_soft_write(
                &recon,
                hybrid_locs,
                &write_config,
                params.dream_write_weight,
            );
        }

        // Merge only within hybrids
        knn_merge(hybrid_locs, &write_config);

        noise *= params.dream_noise_decay;
    }
}

// ============================================================
// Step 6: Fingerprint computation
// ============================================================

/// Compute fingerprint: weighted centroid → Hopfield iterative refinement.
/// Mirrors Collection::fingerprint() but operates on bare snapshot data.
fn compute_fingerprint(snap: &EAMSnapshot) -> Option<Vec<f64>> {
    let total_writes: f64 = snap.locations.iter().map(|l| l.write_count).sum();
    if total_writes < 1e-10 {
        return None;
    }

    let d = snap.dim();
    let mut centroid = vec![0.0; d];
    for loc in &snap.locations {
        if loc.write_count < 1e-10 {
            continue;
        }
        let pattern = loc.normalized_pattern();
        for (c, &p) in centroid.iter_mut().zip(pattern.iter()) {
            *c += loc.write_count * p;
        }
    }
    centroid = vec_ops::normalize(&centroid);

    read::hopfield_iter(&centroid, &snap.locations, &snap.config).ok()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a location with specific address, pattern, and write count.
    fn make_loc(id: u64, addr: &[f64], pattern: &[f64], wc: f64) -> HardLocation {
        let mut loc = HardLocation::new(LocationId(id), vec_ops::normalize(addr));
        loc.counter = pattern.iter().map(|p| p * wc).collect();
        loc.write_count = wc;
        loc
    }

    /// Create a snapshot with given locations and sensible config.
    fn make_snapshot(locations: Vec<HardLocation>, d: usize) -> EAMSnapshot {
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 1;
        config.k = locations.len().min(20).max(1);
        config.beta = 5.0;
        config.t_max = 10;
        config.epsilon = 1e-6;
        EAMSnapshot { locations, config }
    }

    /// Build a cluster of locations around a direction with some spread.
    fn make_cluster(
        start_id: u64,
        center: &[f64],
        pattern: &[f64],
        count: usize,
        spread: f64,
    ) -> Vec<HardLocation> {
        let mut rng = rand::thread_rng();
        let center_norm = vec_ops::normalize(center);
        let pattern_norm = vec_ops::normalize(pattern);
        let mut locs = Vec::new();
        for i in 0..count {
            let mut addr: Vec<f64> = center_norm
                .iter()
                .map(|&c| c + rng.sample::<f64, _>(StandardNormal) * spread)
                .collect();
            addr = vec_ops::normalize(&addr);
            let pat: Vec<f64> = pattern_norm
                .iter()
                .map(|&p| p + rng.sample::<f64, _>(StandardNormal) * spread * 0.5)
                .collect();
            let mut loc = HardLocation::new(LocationId(start_id + i as u64), addr);
            loc.counter = pat.iter().map(|p| p * 5.0).collect();
            loc.write_count = 5.0;
            locs.push(loc);
        }
        locs
    }

    // --- Test 1: slerp ---

    #[test]
    fn test_slerp_correctness() {
        let a = vec_ops::normalize(&[1.0, 0.0, 0.0]);
        let b = vec_ops::normalize(&[0.0, 1.0, 0.0]);

        // Endpoints
        let s0 = slerp(&a, &b, 0.0);
        assert!(vec_ops::cosine_similarity(&s0, &a) > 0.999);

        let s1 = slerp(&a, &b, 1.0);
        assert!(vec_ops::cosine_similarity(&s1, &b) > 0.999);

        // Midpoint should be equidistant and unit norm
        let mid = slerp(&a, &b, 0.5);
        assert!((vec_ops::l2_norm(&mid) - 1.0).abs() < 1e-10);
        let sim_a = vec_ops::cosine_similarity(&mid, &a);
        let sim_b = vec_ops::cosine_similarity(&mid, &b);
        assert!((sim_a - sim_b).abs() < 1e-10);

        // Identity
        let same = slerp(&a, &a, 0.5);
        assert!(vec_ops::cosine_similarity(&same, &a) > 0.999);
    }

    // --- Test 2: combined descent finds hybrid ---

    #[test]
    fn test_combined_descent_finds_hybrid() {
        let d = 16;
        let mut addr_a = vec![0.0; d];
        addr_a[0] = 1.0;
        let mut addr_b = vec![0.0; d];
        addr_b[1] = 1.0;

        // A has a strong attractor at [1, 0, 0, ...]
        let locs_a = make_cluster(0, &addr_a, &addr_a, 10, 0.05);
        let a = make_snapshot(locs_a, d);

        // B has a strong attractor at [0, 1, 0, ...]
        let locs_b = make_cluster(0, &addr_b, &addr_b, 10, 0.05);
        let b = make_snapshot(locs_b, d);

        // Start from somewhere between
        let mut start = vec![0.0; d];
        start[0] = 0.5;
        start[1] = 0.5;
        start[2] = 0.1;
        let start = vec_ops::normalize(&start);

        let params = ComposeParams::default();
        let attractor = combined_descent(&start, &a.locations, &b.locations, &params);

        // Should be near normalize([1, 1, 0, ...])
        let expected = vec_ops::normalize(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = vec_ops::cosine_similarity(&attractor, &expected);
        assert!(
            sim > 0.9,
            "Combined descent should find hybrid attractor near [1,1,0,...], got sim={sim:.4}"
        );
    }

    // --- Test 3: deduplication ---

    #[test]
    fn test_dedup_collapses_duplicates() {
        let base = vec_ops::normalize(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut rng = rand::thread_rng();

        // 100 noisy copies
        let attractors: Vec<Vec<f64>> = (0..100)
            .map(|_| {
                let noisy: Vec<f64> = base
                    .iter()
                    .map(|&b| b + rng.sample::<f64, _>(StandardNormal) * 0.01)
                    .collect();
                vec_ops::normalize(&noisy)
            })
            .collect();

        let unique = deduplicate_attractors(&attractors, 0.95);
        assert_eq!(unique.len(), 1, "100 near-identical attractors should collapse to 1");
    }

    // --- Test 4: read_with_confidence ---

    #[test]
    fn test_read_with_confidence_values() {
        let d = 8;
        let mut addr = vec![0.0; d];
        addr[0] = 1.0;
        let locs = make_cluster(0, &addr, &addr, 5, 0.05);

        // Query at the cluster center: high confidence
        let (_, conf_near) = read_with_confidence(&vec_ops::normalize(&addr), &locs, 5, 5.0);
        assert!(
            conf_near > 0.8,
            "Confidence should be high near cluster, got {conf_near}"
        );

        // Query far away: lower confidence
        let mut far = vec![0.0; d];
        far[d - 1] = 1.0;
        let (_, conf_far) = read_with_confidence(&vec_ops::normalize(&far), &locs, 5, 5.0);
        assert!(
            conf_far < conf_near,
            "Confidence far ({conf_far}) should be less than near ({conf_near})"
        );
    }

    // --- Test 5: full compose of orthogonal memories ---

    #[test]
    fn test_compose_orthogonal_memories() {
        let d = 16;
        let mut center_a = vec![0.0; d];
        center_a[0] = 1.0;
        let mut center_b = vec![0.0; d];
        center_b[1] = 1.0;

        let locs_a = make_cluster(0, &center_a, &center_a, 10, 0.05);
        let a = make_snapshot(locs_a, d);

        let locs_b = make_cluster(0, &center_b, &center_b, 10, 0.05);
        let b = make_snapshot(locs_b, d);

        let mut params = ComposeParams::default();
        params.num_random_probes = 20; // fewer for test speed
        params.dream_rounds = 3;

        let result = compose(&a, &b, &params).unwrap();
        let diag = &result.diagnostics;

        // C should have some locations
        assert!(result.snapshot.num_locations() > 0);

        // C should not be identical to A or B
        assert!(
            diag.fingerprint_sim_a < 0.95,
            "C should differ from A, sim={:.4}",
            diag.fingerprint_sim_a
        );
        assert!(
            diag.fingerprint_sim_b < 0.95,
            "C should differ from B, sim={:.4}",
            diag.fingerprint_sim_b
        );

        // C should be stable (its fingerprint is an attractor)
        assert!(
            diag.fingerprint_stability > 0.95,
            "C should be stable, stability={:.4}",
            diag.fingerprint_stability
        );
    }

    // --- Test 6: compose identical preserves ---

    #[test]
    fn test_compose_identical_preserves() {
        let d = 16;
        let mut center = vec![0.0; d];
        center[0] = 1.0;
        let locs = make_cluster(0, &center, &center, 10, 0.05);
        let a = make_snapshot(locs, d);

        let mut params = ComposeParams::default();
        params.num_random_probes = 10;
        params.dream_rounds = 2;
        // Relax validation since A+A will naturally be similar to A
        params.validation_max_similarity = 1.0;

        let result = compose(&a, &a, &params).unwrap();

        // C should be very similar to A (composing A with itself)
        assert!(
            result.diagnostics.fingerprint_sim_a > 0.8,
            "compose(A, A) should be similar to A, sim={:.4}",
            result.diagnostics.fingerprint_sim_a
        );
    }

    // --- Test 7: diagnostics populated ---

    #[test]
    fn test_diagnostics_populated() {
        let d = 8;
        let mut ca = vec![0.0; d];
        ca[0] = 1.0;
        let mut cb = vec![0.0; d];
        cb[1] = 1.0;

        let a = make_snapshot(make_cluster(0, &ca, &ca, 5, 0.1), d);
        let b = make_snapshot(make_cluster(0, &cb, &cb, 5, 0.1), d);

        let mut params = ComposeParams::default();
        params.num_random_probes = 5;
        params.dream_rounds = 2;

        let result = compose(&a, &b, &params).unwrap();
        let d = &result.diagnostics;

        assert_eq!(d.pool_size, 10);
        assert!(d.boundary_probes_count > 0);
        assert_eq!(d.parent_probes_count, 10);
        assert_eq!(d.random_probes_count, 5);
        assert!(d.raw_attractors_count > 0);
        assert!(d.unique_attractors_count > 0);
        assert!(d.unique_attractors_count <= d.raw_attractors_count);
        assert!(d.locations_built > 0);
        assert!(d.post_dream_locations > 0);
        assert!(d.fingerprint_stability > 0.0);
    }

    // --- Test 8: dream reduces redundancy ---

    #[test]
    fn test_dream_reduces_redundancy() {
        let d = 8;
        let mut center = vec![0.0; d];
        center[0] = 1.0;

        // Create many redundant locations around the same point
        let locs = make_cluster(0, &center, &center, 30, 0.02);
        let mut config = EAMConfig::new(d).unwrap();
        config.k = 10;
        config.tau_merge = 0.95;
        let mut snap = EAMSnapshot {
            locations: locs,
            config,
        };

        let before = snap.num_locations();

        let params = ComposeParams {
            dream_rounds: 5,
            dream_noise_initial: 0.05,
            dream_noise_decay: 0.8,
            dream_write_weight: 0.2,
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let all_for_read = snap.locations.clone();
        let read_config = snap.config.clone();
        dream_consolidate_hybrid(
            &mut snap.locations,
            &all_for_read,
            &read_config,
            &params,
            &mut rng,
        );

        let after = snap.locations.len();
        assert!(
            after < before,
            "Dream should merge redundant locations: {before} -> {after}"
        );
    }

    // --- Test 9: error cases ---

    #[test]
    fn test_error_empty_snapshot() {
        let a = make_snapshot(vec![], 3);
        let b = make_snapshot(vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0)], 3);
        let params = ComposeParams::default();

        assert!(compose(&a, &b, &params).is_err());
        assert!(compose(&b, &a, &params).is_err());
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let a = make_snapshot(vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0)], 3);
        let b = make_snapshot(vec![make_loc(0, &[1.0, 0.0], &[1.0, 0.0], 1.0)], 2);
        let params = ComposeParams::default();

        assert!(compose(&a, &b, &params).is_err());
    }

    // --- Test 10: stress test 128d ---

    #[test]
    fn test_stress_128d() {
        let d = 128;
        let mut rng = rand::thread_rng();

        // A: cluster in one region
        let center_a: Vec<f64> = vec_ops::random_unit_vector(d, &mut rng);
        let locs_a = make_cluster(0, &center_a, &center_a, 50, 0.1);
        let a = make_snapshot(locs_a, d);

        // B: cluster in a different region
        let center_b: Vec<f64> = vec_ops::random_unit_vector(d, &mut rng);
        let locs_b = make_cluster(0, &center_b, &center_b, 50, 0.1);
        let b = make_snapshot(locs_b, d);

        let mut params = ComposeParams::default();
        params.num_random_probes = 20;
        params.dream_rounds = 3;

        let result = compose(&a, &b, &params).unwrap();
        assert!(result.snapshot.num_locations() > 0);
        assert!(result.diagnostics.unique_attractors_count > 0);
    }
}
