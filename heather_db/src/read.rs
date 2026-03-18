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

/// Select landmark locations: the most-written-to locations, spread across the space.
/// These are the most stable and central by construction.
pub fn select_landmarks(locations: &[HardLocation], num_landmarks: usize) -> Vec<usize> {
    if locations.len() <= num_landmarks {
        return (0..locations.len()).collect();
    }

    let mut indexed: Vec<(usize, f64)> = locations
        .iter()
        .enumerate()
        .map(|(i, loc)| (i, loc.write_count))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed.truncate(num_landmarks);
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Build a flat LocationId → index lookup array.
/// O(1) lookup by indexing directly: `id_lookup[location_id] = index`.
/// Invalid entries are `u32::MAX`.
pub fn build_id_lookup(locations: &[HardLocation]) -> Vec<u32> {
    let max_id = locations.iter().map(|l| l.id.0).max().unwrap_or(0) as usize;
    let mut lookup = vec![u32::MAX; max_id + 1];
    for (i, loc) in locations.iter().enumerate() {
        lookup[loc.id.0 as usize] = i as u32;
    }
    lookup
}

/// Check if the neighbor graph is populated enough for graph search.
/// Requires at least 100 locations and average neighbor count >= k/2.
pub fn is_graph_ready(locations: &[HardLocation], k: usize) -> bool {
    if locations.len() < 100 {
        return false;
    }
    let sample_count = 20.min(locations.len());
    let step = locations.len() / sample_count;
    let total_neighbors: usize = (0..sample_count)
        .map(|i| locations[i * step].neighbors.len())
        .sum();
    let avg = total_neighbors as f64 / sample_count as f64;
    avg >= k as f64 / 2.0
}

/// Single-probe greedy graph search for k-nearest neighbors.
///
/// Follows the manifold: one entry point (nearest landmark), greedy descent
/// through neighbor edges. Each hop gathers unvisited neighbor addresses into
/// a contiguous buffer and computes similarities as a batched dot product
/// (addresses are unit-normalized, so dot = cosine similarity).
///
/// Per hop: one batched mat-vec multiply over ~neighbor_cap rows × D columns.
/// Hops are sequential (each depends on the previous), work within each hop
/// is embarrassingly parallel.
///
/// Cost: O(landmarks·D + hops·neighbor_cap·D), hops ≈ O(log L).
pub fn graph_activate(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
    landmarks: &[usize],
    id_lookup: &[u32],
) -> (Vec<usize>, Vec<f64>) {
    let n = locations.len();
    let d = query.len();
    if landmarks.is_empty() || n == 0 {
        return activate(query, locations, k);
    }

    // Normalize query once — all addresses are unit vectors,
    // so dot product = cosine similarity. Saves 2D FLOPs per comparison.
    let q = vec_ops::normalize(query);

    // ── Entry point: nearest landmark via batched dot ────────────
    let mut lm_buf = Vec::with_capacity(landmarks.len() * d);
    for &lm in landmarks {
        lm_buf.extend_from_slice(&locations[lm].address);
    }
    let lm_sims = vec_ops::batch_dot_unit(&q, &lm_buf, d);

    let mut best_lm = 0usize;
    let mut best_lm_sim = lm_sims[0];
    for (i, &sim) in lm_sims.iter().enumerate().skip(1) {
        if sim > best_lm_sim {
            best_lm_sim = sim;
            best_lm = i;
        }
    }
    let entry = landmarks[best_lm];

    // ── State: bitset visited, min-heap result ───────────────────
    let mut visited = vec![false; n];
    visited[entry] = true;

    let mut result: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);
    result.push(MinEntry {
        index: entry,
        similarity: best_lm_sim,
    });

    let mut current = entry;

    // Scratch buffers — reused across hops, no allocation per hop
    let mut neighbor_buf: Vec<f64> = Vec::with_capacity(40 * d);
    let mut neighbor_indices: Vec<usize> = Vec::with_capacity(40);

    // ── Greedy descent ───────────────────────────────────────────
    loop {
        neighbor_buf.clear();
        neighbor_indices.clear();

        // Gather unvisited neighbors into contiguous buffer
        let loc = &locations[current];
        for &neighbor_id in &loc.neighbors {
            let nid = neighbor_id as usize;
            if nid >= id_lookup.len() {
                continue;
            }
            let v = id_lookup[nid];
            if v == u32::MAX {
                continue;
            }
            let idx = v as usize;
            if idx >= n || visited[idx] {
                continue;
            }
            visited[idx] = true;
            neighbor_indices.push(idx);
            neighbor_buf.extend_from_slice(&locations[idx].address);
        }

        if neighbor_indices.is_empty() {
            break;
        }

        // Batched dot product: query · [neighbor addresses]
        let sims = vec_ops::batch_dot_unit(&q, &neighbor_buf, d);

        // Update result heap and find best unvisited neighbor for greedy step
        let mut best_next: Option<(usize, f64)> = None;
        for (i, &sim) in sims.iter().enumerate() {
            let idx = neighbor_indices[i];

            if result.len() < k {
                result.push(MinEntry {
                    index: idx,
                    similarity: sim,
                });
            } else if let Some(worst) = result.peek() {
                if sim > worst.similarity {
                    result.pop();
                    result.push(MinEntry {
                        index: idx,
                        similarity: sim,
                    });
                }
            }

            match best_next {
                None => best_next = Some((idx, sim)),
                Some((_, bs)) if sim > bs => best_next = Some((idx, sim)),
                _ => {}
            }
        }

        match best_next {
            None => break,
            Some((next_idx, next_sim)) => {
                // Stop if best neighbor can't improve our k-th best
                if result.len() >= k {
                    if let Some(worst) = result.peek() {
                        if next_sim < worst.similarity {
                            break;
                        }
                    }
                }
                current = next_idx;
            }
        }
    }

    // Extract sorted descending
    let mut entries: Vec<(usize, f64)> = result
        .into_iter()
        .map(|e| (e.index, e.similarity))
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let indices = entries.iter().map(|(i, _)| *i).collect();
    let similarities = entries.iter().map(|(_, s)| *s).collect();
    (indices, similarities)
}

/// Activate using graph search if the graph is ready, otherwise brute force.
pub fn activate_auto(
    query: &[f64],
    locations: &[HardLocation],
    k: usize,
    landmarks: &[usize],
    id_lookup: &[u32],
) -> (Vec<usize>, Vec<f64>) {
    if is_graph_ready(locations, k) && !landmarks.is_empty() {
        graph_activate(query, locations, k, landmarks, id_lookup)
    } else {
        activate(query, locations, k)
    }
}

/// Hopfield iterative read from pre-computed activation set.
/// Same as `hopfield_iter` but skips the activate step.
pub fn hopfield_iter_from(
    query: &[f64],
    locations: &[HardLocation],
    config: &EAMConfig,
    indices: &[usize],
) -> Result<Vec<f64>> {
    if indices.is_empty() {
        return Err(HeatherError::EmptyMemory);
    }

    let patterns: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| locations[i].unit_pattern())
        .collect();

    let addresses: Vec<&[f64]> = indices
        .iter()
        .map(|&i| locations[i].address.as_slice())
        .collect();

    let mut xi = vec_ops::normalize(query);

    for _t in 0..config.t_max {
        let sims: Vec<f64> = addresses
            .iter()
            .map(|addr| vec_ops::cosine_similarity(&xi, addr))
            .collect();

        let alpha = vec_ops::softmax(&sims, config.beta);
        let pattern_refs: Vec<&[f64]> = patterns.iter().map(|p| p.as_slice()).collect();
        let xi_new = vec_ops::normalize(&vec_ops::weighted_sum(&pattern_refs, &alpha));

        let sim = vec_ops::cosine_similarity(&xi, &xi_new);
        xi = xi_new;

        if sim > 1.0 - config.epsilon {
            break;
        }
    }

    Ok(xi)
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
