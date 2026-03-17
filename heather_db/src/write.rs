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

/// Execute the full 6-step adaptive write pipeline.
pub fn adaptive_write(
    input: &[f64],
    locations: &mut [HardLocation],
    config: &EAMConfig,
    eta: f64,
    next_id: &mut u64,
    rng: &mut impl Rng,
) -> WriteResult {
    let k = config.k.min(locations.len());
    let mut new_locations = Vec::new();

    // Step 1: k-NN activation
    let (indices, sims) = read::activate(input, locations, k);

    if indices.is_empty() {
        return WriteResult {
            modified_indices: vec![],
            new_locations: vec![],
            eta,
        };
    }

    // Activation weights: w_j = max(S(x, a_j), 0) / max_j S(x, a_j)
    let max_sim = sims[0]; // already sorted descending
    let weights: Vec<f64> = if max_sim > 1e-12 {
        sims.iter().map(|s| (s.max(0.0)) / max_sim).collect()
    } else {
        vec![1.0 / k as f64; sims.len()]
    };

    // Step 2: Winner selection via conscience mechanism
    // S_eff = S(x, a_j) - γ * n_j / Σn_i
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
        0 // first entry (highest similarity)
    };
    let winner_global = indices[winner_local];

    // Step 3: Counter update
    // c_j += w_j * x, n_j += w_j
    for (local_idx, &global_idx) in indices.iter().enumerate() {
        let w = weights[local_idx];
        vec_ops::add_scaled(&mut locations[global_idx].counter, input, w);
        locations[global_idx].write_count += w;
    }

    // Step 4: Competitive learning with damped rate
    // η_eff = η / (1 + n_j / τ_damp)
    let eta_winner = eta / (1.0 + locations[winner_global].write_count / config.tau_damp);

    // Winner moves toward input
    let diff: Vec<f64> = input
        .iter()
        .zip(locations[winner_global].address.iter())
        .map(|(x, a)| x - a)
        .collect();
    vec_ops::add_scaled(&mut locations[winner_global].address, &diff, eta_winner);
    locations[winner_global].address = vec_ops::normalize(&locations[winner_global].address);

    // Neighbors move toward input at reduced rate
    for (local_idx, &global_idx) in indices.iter().enumerate() {
        if local_idx == winner_local {
            continue;
        }
        let eta_neighbor = eta_winner * 0.1; // neighbors learn at 10% of winner rate
        let diff: Vec<f64> = input
            .iter()
            .zip(locations[global_idx].address.iter())
            .map(|(x, a)| x - a)
            .collect();
        vec_ops::add_scaled(&mut locations[global_idx].address, &diff, eta_neighbor);
        locations[global_idx].address = vec_ops::normalize(&locations[global_idx].address);
    }

    // Decay learning rate
    let eta_new = (eta * config.lambda).max(config.eta_min);

    // Step 5: Novelty split - if max similarity is below threshold, create new location
    if max_sim < config.tau_split {
        let id = LocationId(*next_id);
        *next_id += 1;
        // New location at a perturbed version of the input
        let addr = vec_ops::normalize(input);
        let mut new_loc = HardLocation::new(id, addr);
        // Initialize with the input pattern
        new_loc.counter = input.to_vec();
        new_loc.write_count = 1.0;
        new_locations.push(new_loc);
    }

    // Step 6: Overload split - if winner write count exceeds threshold
    if locations[winner_global].write_count > config.tau_overload {
        let id = LocationId(*next_id);
        *next_id += 1;

        // Create new location by perturbing winner's address
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

        new_locations.push(new_loc);
    }

    WriteResult {
        modified_indices: indices,
        new_locations,
        eta: eta_new,
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

        let result = adaptive_write(&input, &mut locations, &config, 0.01, &mut next_id, &mut rng);
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

        let result = adaptive_write(&input, &mut locations, &mut config, 0.01, &mut next_id, &mut rng);
        // Should have created at least one new location (novelty split)
        assert!(!result.new_locations.is_empty());
    }
}
