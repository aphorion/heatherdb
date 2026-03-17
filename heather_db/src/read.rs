use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::config::EAMConfig;
use crate::error::{HeatherError, Result};
use crate::location::HardLocation;
use crate::vec_ops;

/// Minimum work (locations * dimension) before rayon parallelism kicks in.
/// Below this threshold, sequential is faster due to thread scheduling overhead.
const PARALLEL_THRESHOLD: usize = 200_000;

/// A min-heap entry for top-k selection. Ordered by similarity ascending
/// so that the smallest element is at the top (and can be evicted first).
struct MinEntry {
    index: usize,
    similarity: f64,
}

impl PartialEq for MinEntry {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for MinEntry {}

impl PartialOrd for MinEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller similarity = higher priority (min-heap)
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Find the k nearest locations by cosine similarity to the query.
/// Returns (indices, similarities) sorted descending by similarity.
/// Uses a min-heap for O(n log k) instead of O(n log n) full sort.
/// Parallelizes similarity computation with rayon when the work exceeds PARALLEL_THRESHOLD.
pub fn activate(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
) -> (Vec<usize>, Vec<f64>) {
    let work = locations.len() * query.len();

    // Compute all similarities — parallel or sequential based on work size
    let sims: Vec<(usize, f64)> = if work >= PARALLEL_THRESHOLD {
        locations
            .par_iter()
            .enumerate()
            .map(|(i, loc)| (i, vec_ops::cosine_similarity(query, &loc.address)))
            .collect()
    } else {
        locations
            .iter()
            .enumerate()
            .map(|(i, loc)| (i, vec_ops::cosine_similarity(query, &loc.address)))
            .collect()
    };

    // Top-k selection via min-heap (sequential — fast at k=20)
    let mut heap: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);
    for (i, sim) in sims {
        if heap.len() < k {
            heap.push(MinEntry { index: i, similarity: sim });
        } else if let Some(min) = heap.peek() {
            if sim > min.similarity {
                heap.pop();
                heap.push(MinEntry { index: i, similarity: sim });
            }
        }
    }

    // Drain into a vec sorted descending by similarity
    let mut entries: Vec<(usize, f64)> = heap
        .into_iter()
        .map(|e| (e.index, e.similarity))
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let indices: Vec<usize> = entries.iter().map(|(i, _)| *i).collect();
    let similarities: Vec<f64> = entries.iter().map(|(_, s)| *s).collect();
    (indices, similarities)
}

/// Hopfield iterative read (Eq. 13-14).
/// Fixed activation set, iterate softmax reweighting until convergence.
pub fn hopfield_iter(
    query: &[f64],
    locations: &[HardLocation],
    config: &EAMConfig,
) -> Result<Vec<f64>> {
    if locations.is_empty() {
        return Err(HeatherError::EmptyMemory);
    }

    let k = config.k.min(locations.len());
    let (indices, _sims) = activate(query, locations, k);

    // Collect normalized patterns from activated locations
    let patterns: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| locations[i].unit_pattern())
        .collect();

    let addresses: Vec<&[f64]> = indices.iter().map(|&i| locations[i].address.as_slice()).collect();

    // ξ₀ = normalize(query)
    let mut xi = vec_ops::normalize(query);

    for _t in 0..config.t_max {
        // Compute similarities between current estimate and activated addresses
        let sims: Vec<f64> = addresses
            .iter()
            .map(|addr| vec_ops::cosine_similarity(&xi, addr))
            .collect();

        // Softmax weights
        let alpha = vec_ops::softmax(&sims, config.beta);

        // Weighted sum of normalized patterns
        let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
        let xi_new = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));

        // Check convergence
        let sim = vec_ops::cosine_similarity(&xi, &xi_new);
        xi = xi_new;

        if sim > 1.0 - config.epsilon {
            break;
        }
    }

    Ok(xi)
}

/// Traced activation info for a single location.
#[derive(Debug, Clone)]
pub struct ActivatedLocation {
    pub id: usize,
    pub similarity: f64,
    pub weight: f64,
}

/// Result of a traced Hopfield iterative read.
#[derive(Debug, Clone)]
pub struct ReadTrace {
    pub iterations: usize,
    pub converged: bool,
    pub activated_locations: Vec<ActivatedLocation>,
    pub result: Vec<f64>,
}

/// Hopfield iterative read with activation trace.
/// Same as `hopfield_iter()` but returns full activation details.
pub fn hopfield_iter_traced(
    query: &[f64],
    locations: &[HardLocation],
    config: &EAMConfig,
) -> Result<ReadTrace> {
    if locations.is_empty() {
        return Err(HeatherError::EmptyMemory);
    }

    let k = config.k.min(locations.len());
    let (indices, initial_sims) = activate(query, locations, k);

    let patterns: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| locations[i].unit_pattern())
        .collect();

    let addresses: Vec<&[f64]> = indices.iter().map(|&i| locations[i].address.as_slice()).collect();

    let mut xi = vec_ops::normalize(query);
    let mut converged = false;
    let mut iterations = 0;
    let mut final_weights = vec_ops::softmax(&initial_sims, config.beta);

    for t in 0..config.t_max {
        iterations = t + 1;
        let sims: Vec<f64> = addresses
            .iter()
            .map(|addr| vec_ops::cosine_similarity(&xi, addr))
            .collect();

        let alpha = vec_ops::softmax(&sims, config.beta);
        final_weights = alpha.clone();

        let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
        let xi_new = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));

        let sim = vec_ops::cosine_similarity(&xi, &xi_new);
        xi = xi_new;

        if sim > 1.0 - config.epsilon {
            converged = true;
            break;
        }
    }

    let activated_locations = indices
        .iter()
        .zip(final_weights.iter())
        .map(|(&idx, &weight)| {
            let sim = vec_ops::cosine_similarity(&xi, &locations[idx].address);
            ActivatedLocation {
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

/// Hopfield single-step read (Eq. 11).
/// Single softmax pass, cheaper than iterative.
pub fn hopfield_ss(
    query: &[f64],
    locations: &[HardLocation],
    config: &EAMConfig,
) -> Result<Vec<f64>> {
    if locations.is_empty() {
        return Err(HeatherError::EmptyMemory);
    }

    let k = config.k.min(locations.len());
    let (indices, sims) = activate(query, locations, k);

    let alpha = vec_ops::softmax(&sims, config.beta);

    let patterns: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| locations[i].normalized_pattern())
        .collect();
    let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();

    let result = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::LocationId;

    fn make_loc(id: u64, addr: Vec<f64>, pattern: Vec<f64>, wc: f64) -> HardLocation {
        let mut loc = HardLocation::new(LocationId(id), addr);
        // counter = pattern * write_count so that normalized_pattern() returns pattern
        loc.counter = pattern.iter().map(|p| p * wc).collect();
        loc.write_count = wc;
        loc
    }

    #[test]
    fn test_hopfield_iter_exact_recall() {
        let config = EAMConfig::new(3).unwrap();
        let pattern = vec_ops::normalize(&[1.0, 2.0, 3.0]);
        let locations = vec![
            make_loc(0, pattern.clone(), pattern.clone(), 5.0),
            make_loc(1, vec_ops::normalize(&[-1.0, 0.0, 0.0]), vec_ops::normalize(&[-1.0, 0.0, 0.0]), 5.0),
        ];

        let result = hopfield_iter(&pattern, &locations, &config).unwrap();
        let sim = vec_ops::cosine_similarity(&result, &pattern);
        assert!(sim > 0.99, "expected near-perfect recall, got {sim}");
    }

    #[test]
    fn test_hopfield_ss() {
        let config = EAMConfig::new(3).unwrap();
        let pattern = vec_ops::normalize(&[1.0, 2.0, 3.0]);
        let locations = vec![
            make_loc(0, pattern.clone(), pattern.clone(), 5.0),
        ];

        let result = hopfield_ss(&pattern, &locations, &config).unwrap();
        let sim = vec_ops::cosine_similarity(&result, &pattern);
        assert!(sim > 0.99);
    }
}
