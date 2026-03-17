use heather_db::vec_ops;
use heather_db::EAMConfig;

use crate::consolidate::consolidate;
use crate::error::{AlgebraError, Result};
use crate::snapshot::EAMSnapshot;

/// Check that two snapshots have the same dimensionality.
fn check_dims(a: &EAMSnapshot, b: &EAMSnapshot) -> Result<()> {
    if a.dim() != b.dim() {
        return Err(AlgebraError::DimensionMismatch {
            left: a.dim(),
            right: b.dim(),
        });
    }
    Ok(())
}

/// Resolve config for binary operations.
fn result_config(a: &EAMSnapshot, b: &EAMSnapshot) -> EAMConfig {
    let mut config = a.config.clone();
    config.l_max = config.l_max.max(b.config.l_max);
    config
}

/// Extract the normalized pattern (counter / write_count) from each location.
fn extract_patterns(snap: &EAMSnapshot) -> Vec<Vec<f64>> {
    snap.locations
        .iter()
        .map(|l| l.normalized_pattern())
        .collect()
}

/// Create pairwise-combination locations from two sets of patterns.
///
/// For each pair (i, j), creates a location whose:
///   - counter = pattern_a[i] ± pattern_b[j]  (sign applied to b)
///   - address = normalize(counter)
///   - write_count = 1.0
///
/// `sign` is +1.0 for add, -1.0 for sub.
/// `max_cross_k` limits each A-location to its top-k nearest B-locations
/// (0 = full cartesian product).
fn pairwise_combine(
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    sign: f64,
    max_cross_k: usize,
) -> Vec<heather_db::HardLocation> {
    let patterns_a = extract_patterns(a);
    let patterns_b = extract_patterns(b);
    let mut locations = Vec::new();
    let mut id = 0u64;

    let use_full = max_cross_k == 0 || max_cross_k >= b.locations.len();

    for (i, pa) in patterns_a.iter().enumerate() {
        let pairs: Box<dyn Iterator<Item = usize>> = if use_full {
            Box::new(0..patterns_b.len())
        } else {
            // Top-k nearest B-locations by address similarity
            let mut sims: Vec<(usize, f64)> = b
                .locations
                .iter()
                .enumerate()
                .map(|(j, lb)| {
                    (
                        j,
                        vec_ops::cosine_similarity(&a.locations[i].address, &lb.address),
                    )
                })
                .collect();
            sims.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sims.truncate(max_cross_k);
            Box::new(sims.into_iter().map(|(j, _)| j))
        };

        for j in pairs {
            let pb = &patterns_b[j];
            let combined: Vec<f64> = pa
                .iter()
                .zip(pb.iter())
                .map(|(a, b)| a + sign * b)
                .collect();

            // Skip degenerate (near-zero) combinations
            if vec_ops::l2_norm(&combined) < 1e-10 {
                continue;
            }

            let address = vec_ops::normalize(&combined);
            let mut loc =
                heather_db::HardLocation::new(heather_db::LocationId(id), address);
            loc.counter = combined;
            loc.write_count = 1.0;
            locations.push(loc);
            id += 1;
        }
    }

    locations
}

/// Energy-correct EAM addition via pairwise pattern sums.
///
/// # Energy semantics
///
/// The modern Hopfield energy for EAM_A is:
///   E_A(x) = -1/β · log(Σ_i exp(β · x·ξ_i^A))
///
/// Adding energies E_{A+B} = E_A + E_B gives:
///   E_{A+B}(x) = -1/β · log(Σ_{i,j} exp(β · x·(ξ_i^A + ξ_j^B)))
///
/// The attractors of the combined system are at the **pairwise sums**
/// of patterns from A and B. For each pair of locations, we create
/// a new location whose pattern is ξ_i + ξ_j.
///
/// A documentation query descends this combined landscape and converges
/// to a hybrid attractor that neither EAM alone would produce.
pub fn add(a: &EAMSnapshot, b: &EAMSnapshot) -> Result<EAMSnapshot> {
    add_with_limit(a, b, 0)
}

/// Like [`add`], but limits each A-location to its `max_cross_k` nearest
/// B-locations instead of the full cartesian product. Use for large EAMs.
pub fn add_with_limit(
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    max_cross_k: usize,
) -> Result<EAMSnapshot> {
    check_dims(a, b)?;

    // Edge cases: adding to/from empty is identity
    if a.locations.is_empty() {
        return Ok(b.clone());
    }
    if b.locations.is_empty() {
        return Ok(a.clone());
    }

    let mut config = result_config(a, b);
    let locations = pairwise_combine(a, b, 1.0, max_cross_k);

    // Ensure l_max can accommodate the pairwise products during consolidation
    let pre_consolidation_count = locations.len();
    config.l_max = config.l_max.max(pre_consolidation_count);

    let mut snap = EAMSnapshot { locations, config };
    snap.reindex();
    consolidate(&mut snap);

    // Restore a reasonable l_max for the output EAM
    snap.config.l_max = result_config(a, b).l_max;

    Ok(snap)
}

/// Energy-correct EAM subtraction via pairwise pattern differences.
///
/// E_{A-B} is interpreted as E_{A+(-B)}, where -B negates all patterns.
/// The attractors are at ξ_i^A - ξ_j^B: points that are "like A but
/// away from B."
pub fn sub(a: &EAMSnapshot, b: &EAMSnapshot) -> Result<EAMSnapshot> {
    sub_with_limit(a, b, 0)
}

/// Like [`sub`], but with cross-pair limit for large EAMs.
pub fn sub_with_limit(
    a: &EAMSnapshot,
    b: &EAMSnapshot,
    max_cross_k: usize,
) -> Result<EAMSnapshot> {
    check_dims(a, b)?;

    if a.locations.is_empty() {
        // 0 - B = -B
        return negate(b);
    }
    if b.locations.is_empty() {
        return Ok(a.clone());
    }

    let mut config = result_config(a, b);
    let locations = pairwise_combine(a, b, -1.0, max_cross_k);

    let pre_consolidation_count = locations.len();
    config.l_max = config.l_max.max(pre_consolidation_count);

    let mut snap = EAMSnapshot { locations, config };
    snap.reindex();
    consolidate(&mut snap);

    snap.config.l_max = result_config(a, b).l_max;

    Ok(snap)
}

/// Scale an EAM by a scalar factor.
///
/// Scales all counters by alpha. Does NOT scale write_count,
/// because `normalized_pattern = counter / write_count`, so
/// scaling both would be a no-op.
///
/// In energy terms, scaling counters by α changes the stored patterns
/// to α·ξ_i, which is equivalent to changing the effective temperature
/// to α·β. This sharpens (α > 1) or softens (α < 1) the energy wells.
pub fn scale(a: &EAMSnapshot, alpha: f64) -> Result<EAMSnapshot> {
    if !alpha.is_finite() {
        return Err(AlgebraError::InvalidScalar(format!(
            "scalar must be finite, got {alpha}"
        )));
    }
    if alpha.abs() < 1e-15 {
        return Err(AlgebraError::InvalidScalar(
            "scalar must be non-zero".into(),
        ));
    }

    let mut locations: Vec<_> = a.locations.iter().cloned().collect();
    for loc in &mut locations {
        for c in loc.counter.iter_mut() {
            *c *= alpha;
        }
    }

    Ok(EAMSnapshot {
        locations,
        config: a.config.clone(),
    })
}

/// Negate an EAM: reverse all stored pattern directions.
/// Special case of `scale(a, -1.0)`.
pub fn negate(a: &EAMSnapshot) -> Result<EAMSnapshot> {
    scale(a, -1.0)
}

/// Intersection: keep only locations from A that have a nearby
/// match (above threshold) in B. Extracts shared knowledge.
///
/// A reasonable default threshold is `a.config.tau_merge`.
pub fn intersect(a: &EAMSnapshot, b: &EAMSnapshot, threshold: f64) -> Result<EAMSnapshot> {
    check_dims(a, b)?;

    let mut retained = Vec::new();

    for loc_a in &a.locations {
        let has_match = b
            .locations
            .iter()
            .any(|loc_b| vec_ops::cosine_similarity(&loc_a.address, &loc_b.address) > threshold);
        if has_match {
            retained.push(loc_a.clone());
        }
    }

    let mut snap = EAMSnapshot {
        locations: retained,
        config: a.config.clone(),
    };
    snap.reindex();
    Ok(snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use heather_db::location::LocationId;
    use heather_db::vec_ops;
    use heather_db::HardLocation;

    fn make_loc(id: u64, addr: &[f64], pattern: &[f64], wc: f64) -> HardLocation {
        let mut loc = HardLocation::new(LocationId(id), vec_ops::normalize(addr));
        loc.counter = pattern.iter().map(|p| p * wc).collect();
        loc.write_count = wc;
        loc
    }

    fn make_snapshot(locations: Vec<HardLocation>, d: usize) -> EAMSnapshot {
        let mut config = EAMConfig::new(d).unwrap();
        config.l_0 = 1;
        config.l_max = locations.len().max(1) * 4;
        config.k = locations.len().min(20).max(1);
        EAMSnapshot { locations, config }
    }

    #[test]
    fn test_add_pairwise_count() {
        // 2 locations in A, 2 in B → up to 4 pairwise sums
        let a = make_snapshot(
            vec![
                make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0),
                make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 1.0),
            ],
            3,
        );
        let b = make_snapshot(
            vec![
                make_loc(0, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0),
                make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 1.0),
            ],
            3,
        );

        let result = add(&a, &b).unwrap();
        // Should have pairwise sum locations (after consolidation may merge some)
        assert!(result.num_locations() > 0);
        assert!(result.num_locations() <= 4);
    }

    #[test]
    fn test_add_identity_empty() {
        // A + empty = A
        let locs = vec![
            make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0),
            make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 3.0),
        ];
        let a = make_snapshot(locs, 3);
        let empty = make_snapshot(vec![], 3);

        let result = add(&a, &empty).unwrap();
        assert_eq!(result.num_locations(), 2);
    }

    #[test]
    fn test_add_pairwise_patterns() {
        // A has one location storing pattern [1, 0, 0]
        // B has one location storing pattern [0, 0, 1]
        // The pairwise sum should be [1, 0, 1]
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0)],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0)],
            3,
        );

        let result = add(&a, &b).unwrap();
        assert_eq!(result.num_locations(), 1);

        // The counter should be [1, 0, 1]
        let counter = &result.locations[0].counter;
        assert!((counter[0] - 1.0).abs() < 0.1);
        assert!((counter[1]).abs() < 0.1);
        assert!((counter[2] - 1.0).abs() < 0.1);

        // The address should be normalize([1, 0, 1])
        let addr = &result.locations[0].address;
        let expected = vec_ops::normalize(&[1.0, 0.0, 1.0]);
        for (a, e) in addr.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 0.01);
        }
    }

    #[test]
    fn test_add_commutativity() {
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0)],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 1.0)],
            3,
        );

        let ab = add(&a, &b).unwrap();
        let ba = add(&b, &a).unwrap();

        // Same number of locations
        assert_eq!(ab.num_locations(), ba.num_locations());

        // Patterns should be the same (a+b = b+a)
        for (loc_ab, loc_ba) in ab.locations.iter().zip(ba.locations.iter()) {
            for (c1, c2) in loc_ab.counter.iter().zip(loc_ba.counter.iter()) {
                assert!((c1 - c2).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_sub_patterns() {
        // A stores [1, 0, 0], B stores [0, 0, 1]
        // Pairwise difference: [1, 0, 0] - [0, 0, 1] = [1, 0, -1]
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0)],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0)],
            3,
        );

        let result = sub(&a, &b).unwrap();
        assert_eq!(result.num_locations(), 1);

        let counter = &result.locations[0].counter;
        assert!((counter[0] - 1.0).abs() < 0.1);
        assert!((counter[1]).abs() < 0.1);
        assert!((counter[2] - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn test_sub_self_near_zero() {
        // A - A: pairwise diffs of identical patterns → near-zero
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.5, 0.0], 1.0)],
            3,
        );

        let result = sub(&a, &a).unwrap();
        // All pairwise diffs of identical patterns are zero → skipped
        assert_eq!(result.num_locations(), 0);
    }

    #[test]
    fn test_scale_identity() {
        let locs = vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.5, 0.0], 5.0)];
        let a = make_snapshot(locs, 3);
        let original_counter = a.locations[0].counter.clone();

        let result = scale(&a, 1.0).unwrap();
        assert_eq!(result.locations[0].counter, original_counter);
    }

    #[test]
    fn test_scale_composition() {
        // scale(scale(A, 2), 3) == scale(A, 6)
        let locs = vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.5, 0.0], 5.0)];
        let a = make_snapshot(locs, 3);

        let twice_then_triple = scale(&scale(&a, 2.0).unwrap(), 3.0).unwrap();
        let six_times = scale(&a, 6.0).unwrap();

        for (c1, c2) in twice_then_triple.locations[0]
            .counter
            .iter()
            .zip(six_times.locations[0].counter.iter())
        {
            assert!((c1 - c2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_negate_double() {
        // negate(negate(A)) == A
        let locs = vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.5, -0.3], 5.0)];
        let a = make_snapshot(locs, 3);
        let original_counter = a.locations[0].counter.clone();

        let result = negate(&negate(&a).unwrap()).unwrap();
        for (c1, c2) in result.locations[0].counter.iter().zip(original_counter.iter()) {
            assert!((c1 - c2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scale_zero_rejected() {
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0)],
            3,
        );
        assert!(scale(&a, 0.0).is_err());
    }

    #[test]
    fn test_scale_nan_rejected() {
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0)],
            3,
        );
        assert!(scale(&a, f64::NAN).is_err());
        assert!(scale(&a, f64::INFINITY).is_err());
    }

    #[test]
    fn test_intersect_reflexive() {
        let locs = vec![
            make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0),
            make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 3.0),
        ];
        let a = make_snapshot(locs, 3);

        let result = intersect(&a, &a, 0.9).unwrap();
        assert_eq!(result.num_locations(), 2);
    }

    #[test]
    fn test_intersect_disjoint() {
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0)],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 3.0)],
            3,
        );

        let result = intersect(&a, &b, 0.9).unwrap();
        assert_eq!(result.num_locations(), 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 5.0)],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[1.0, 0.0], &[1.0, 0.0], 5.0)],
            2,
        );

        assert!(add(&a, &b).is_err());
        assert!(sub(&a, &b).is_err());
        assert!(intersect(&a, &b, 0.9).is_err());
    }

    #[test]
    fn test_add_with_limit() {
        // 3 locations in A, 3 in B, limit cross-k to 2
        // Should produce at most 3*2 = 6 pairwise locations
        let a = make_snapshot(
            vec![
                make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0),
                make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 1.0),
                make_loc(2, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0),
            ],
            3,
        );
        let b = make_snapshot(
            vec![
                make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0),
                make_loc(1, &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], 1.0),
                make_loc(2, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0),
            ],
            3,
        );

        let full = add(&a, &b).unwrap();
        let limited = add_with_limit(&a, &b, 2).unwrap();

        // Limited should have fewer or equal locations
        assert!(limited.num_locations() <= full.num_locations());
        assert!(limited.num_locations() > 0);
    }

    #[test]
    fn test_add_merges_similar_sums() {
        // If A has two similar patterns and B has one pattern,
        // the two pairwise sums should be similar and merge
        let a = make_snapshot(
            vec![
                make_loc(0, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0),
                make_loc(1, &[1.0, 0.01, 0.0], &[1.0, 0.01, 0.0], 1.0),
            ],
            3,
        );
        let b = make_snapshot(
            vec![make_loc(0, &[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], 1.0)],
            3,
        );

        let result = add(&a, &b).unwrap();
        // Two similar sums ([1,0,1] and [1,0.01,1]) should merge into 1
        assert_eq!(result.num_locations(), 1);
    }
}
