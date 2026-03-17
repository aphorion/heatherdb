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
pub fn knn_merge(locations: &mut Vec<HardLocation>, config: &EAMConfig) -> MergeResult {
    // Compute all pairwise nearest-neighbor similarities
    let n = locations.len();
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        let mut best_j = None;
        let mut best_sim = f64::NEG_INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let sim = vec_ops::cosine_similarity(&locations[i].address, &locations[j].address);
            if sim > best_sim {
                best_sim = sim;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            if best_sim > config.tau_merge {
                pairs.push((i, j, best_sim));
            }
        }
    }

    // Sort by similarity descending
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
