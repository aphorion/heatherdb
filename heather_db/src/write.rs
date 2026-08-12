use rand::Rng;

use crate::config::EAMConfig;
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

/// Execute the three-phase adaptive write pipeline.
///
/// Phase 1 — Select: k-NN activation, weight computation, conscience winner.
/// Phase 2 — Update: counter accumulation + competitive address migration in one pass.
/// Phase 3 — Regulate: topology maintenance (novelty/overload split, local dedup).
// The eight parameters are the write pipeline's full state: the input, the
// locations it mutates, the config and learning rate that govern it, the id
// counter it draws from, the rng it splits with, and the landmark/lookup
// indices the k-NN search needs. Bundling them into a struct would only move
// the arity somewhere less legible.
#[allow(clippy::too_many_arguments)]
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
    let k = config.k.min(locations.len());

    // ── Phase 1: Select ─────────────────────────────────────────────
    // k-nearest via graph search (falls back to SoA/brute force when graph is young).
    let (indices, sims) = read::activate_auto(input, locations, k, landmarks, id_lookup);

    if indices.is_empty() {
        // Cold start: the index is empty (data-seeded, l_0 == 0). Seed the first
        // hard location directly from the input rather than no-op'ing — this is
        // how a data-seeded index grows its initial codebook. Subsequent writes
        // activate against it and either consolidate or novelty-split.
        let id = LocationId(*next_id);
        *next_id += 1;
        let mut new_loc = HardLocation::new(id, vec_ops::normalize(input));
        new_loc.counter = input.to_vec();
        new_loc.write_count = 1.0;
        return WriteResult {
            modified_indices: vec![],
            new_locations: vec![new_loc],
            eta,
        };
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

    for (local_idx, &global_idx) in indices.iter().enumerate() {
        let loc = &mut locations[global_idx];

        // Counter accumulation: c_j += w_j · x, n_j += w_j
        let w = weights[local_idx];
        vec_ops::add_scaled(&mut loc.counter, input, w);
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

    WriteResult {
        modified_indices: indices,
        new_locations,
        eta: eta_new,
    }
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

        let result = adaptive_write(
            &input,
            &mut locations,
            &config,
            0.01,
            &mut next_id,
            &mut rng,
            &[],
            &[],
        );
        assert!(!result.modified_indices.is_empty());
        // Location 0 should have received the most weight
        assert!(locations[0].write_count > locations[1].write_count);
    }

    #[test]
    fn test_novelty_split() {
        let mut config = EAMConfig::new(3).unwrap();
        config.tau_split = 0.99; // very high threshold -> always split
        let mut locations = vec![HardLocation::new(
            LocationId(0),
            vec_ops::normalize(&[1.0, 0.0, 0.0]),
        )];
        let input = vec_ops::normalize(&[0.0, 1.0, 0.0]); // very different
        let mut next_id = 1u64;
        let mut rng = rand::thread_rng();

        let result = adaptive_write(
            &input,
            &mut locations,
            &config,
            0.01,
            &mut next_id,
            &mut rng,
            &[],
            &[],
        );
        // Should have created at least one new location (novelty split)
        assert!(!result.new_locations.is_empty());
    }
}
