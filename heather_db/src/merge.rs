use rayon::prelude::*;

use crate::config::EAMConfig;
use crate::location::HardLocation;
use crate::vec_ops;

/// Result of a merge operation.
pub struct MergeResult {
    /// IDs of locations that were removed (merged into others)
    pub removed_ids: Vec<u64>,
    /// Mapping of (removed_location_id, survivor_location_id)
    pub merge_map: Vec<(u64, u64)>,
    /// Number of merges performed
    pub merge_count: usize,
}

/// KNN merge: find nearest neighbor pairs above tau_merge, merge them.
/// Process highest-similarity pairs first, skip already-merged locations.
///
/// The nearest-neighbor pass is the O(N²·d) hot spot. We precompute
/// norms once (so the inner kernel is a raw dot product, not the
/// three-traversal `cosine_similarity`) and parallelise the outer
/// loop with rayon.
pub fn knn_merge(locations: &mut Vec<HardLocation>, config: &EAMConfig) -> MergeResult {
    let n = locations.len();
    if n < 2 {
        return MergeResult {
            removed_ids: Vec::new(),
            merge_map: Vec::new(),
            merge_count: 0,
        };
    }

    // Precompute norms once. Addresses are typically unit vectors at
    // this point (every algebra op + the engine write path normalises
    // them), so each norm is ≈ 1.0 and the per-pair division is free —
    // but doing it explicitly keeps the function correct for any input.
    let norms: Vec<f64> = locations
        .par_iter()
        .map(|l| vec_ops::l2_norm(&l.address))
        .collect();

    let tau = config.tau_merge;
    let pairs_unsorted: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            let ni = norms[i];
            if ni < 1e-12 {
                return None;
            }
            let addr_i = &locations[i].address;
            let mut best_j = usize::MAX;
            let mut best_sim = f64::NEG_INFINITY;
            for (j, loc_j) in locations.iter().enumerate() {
                if i == j {
                    continue;
                }
                let nj = norms[j];
                if nj < 1e-12 {
                    continue;
                }
                let sim = vec_ops::dot(addr_i, &loc_j.address) / (ni * nj);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = j;
                }
            }
            if best_j != usize::MAX && best_sim > tau {
                Some((i, best_j, best_sim))
            } else {
                None
            }
        })
        .collect();

    let mut pairs = pairs_unsorted;
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged = vec![false; n];
    let mut removed_ids = Vec::new();
    let mut merge_map = Vec::new();
    let mut merge_count = 0;

    for (i, j, _sim) in &pairs {
        let i = *i;
        let j = *j;
        if merged[i] || merged[j] {
            continue;
        }

        // Merge j into i: weighted average address, additive counters
        let total_wc = locations[i].write_count + locations[j].write_count;

        if total_wc > 1e-12 {
            let wi = locations[i].write_count / total_wc;
            let wj = locations[j].write_count / total_wc;
            let new_addr: Vec<f64> = locations[i]
                .address
                .iter()
                .zip(locations[j].address.iter())
                .map(|(a, b)| a * wi + b * wj)
                .collect();
            locations[i].address = vec_ops::normalize(&new_addr);
        }

        // Additive counters
        let j_counter = locations[j].counter.clone();
        vec_ops::add_scaled(&mut locations[i].counter, &j_counter, 1.0);
        locations[i].write_count += locations[j].write_count;

        // Merge neighbor lists: survivor takes union, capped at neighbor_cap
        let j_neighbors = locations[j].neighbors.clone();
        let j_id = locations[j].id.0;
        let i_id = locations[i].id.0;
        for &n in &j_neighbors {
            if n != i_id && !locations[i].neighbors.contains(&n) {
                locations[i].neighbors.push(n);
            }
        }
        // Remove the merged location from survivor's neighbor list
        locations[i].neighbors.retain(|&n| n != j_id);
        // Cap at adaptive neighbor capacity
        let nb_cap = config.adaptive_neighbor_cap(locations.len());
        if nb_cap > 0 && locations[i].neighbors.len() > nb_cap {
            locations[i].neighbors.truncate(nb_cap);
        }

        merged[j] = true;
        merge_map.push((locations[j].id.0, locations[i].id.0));
        removed_ids.push(locations[j].id.0);
        merge_count += 1;
    }

    // Remove merged locations (iterate in reverse to preserve indices)
    let mut idx = 0;
    locations.retain(|_| {
        let keep = !merged[idx];
        idx += 1;
        keep
    });

    MergeResult {
        removed_ids,
        merge_map,
        merge_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::LocationId;

    #[test]
    fn test_merge_similar_locations() {
        let mut config = EAMConfig::new(3).unwrap();
        config.tau_merge = 0.9;

        let mut locations = vec![
            HardLocation::new(LocationId(0), vec_ops::normalize(&[1.0, 0.0, 0.0])),
            HardLocation::new(LocationId(1), vec_ops::normalize(&[1.0, 0.01, 0.0])), // very similar to 0
            HardLocation::new(LocationId(2), vec_ops::normalize(&[0.0, 1.0, 0.0])),  // different
        ];
        locations[0].write_count = 5.0;
        locations[0].counter = vec![5.0, 0.0, 0.0];
        locations[1].write_count = 3.0;
        locations[1].counter = vec![3.0, 0.03, 0.0];

        let result = knn_merge(&mut locations, &config);
        assert_eq!(result.merge_count, 1);
        assert_eq!(locations.len(), 2);
        // The surviving merged location should have combined write counts
        assert!((locations[0].write_count - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_merge_when_dissimilar() {
        let mut config = EAMConfig::new(3).unwrap();
        config.tau_merge = 0.99;

        let mut locations = vec![
            HardLocation::new(LocationId(0), vec_ops::normalize(&[1.0, 0.0, 0.0])),
            HardLocation::new(LocationId(1), vec_ops::normalize(&[0.0, 1.0, 0.0])),
            HardLocation::new(LocationId(2), vec_ops::normalize(&[0.0, 0.0, 1.0])),
        ];

        let result = knn_merge(&mut locations, &config);
        assert_eq!(result.merge_count, 0);
        assert_eq!(locations.len(), 3);
    }
}
