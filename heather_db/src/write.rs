use rand::Rng;

use crate::config::{EAMConfig, engram_bits, surprise_bits};
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

/// Options for the two-field write. The legacy one-field write is the special
/// case `gate = false` with `counter == address`.
#[derive(Debug, Clone, Copy)]
pub struct WriteOpts {
    /// When true, route by `address` but pick the winner — and decide
    /// join-vs-spawn — by how well the incoming `counter` (the lesson) coheres
    /// with the candidate's accumulated counter. Incoherence spawns a new
    /// location even when the address matches. Used for consolidation, where
    /// same-context-different-law episodes must land in separate families.
    pub gate: bool,
    /// Minimum `cos(counter_so_far, counter_in)` to JOIN under the gate. Below
    /// this, the write spawns. Ignored when `gate == false`.
    pub tau_cohere: f64,
    /// Minimum address similarity for a candidate to be "same context" under
    /// the gate. Scopes the candidate set to one level of a consolidation
    /// ladder — raising it stops a coarser level's structure from merging one
    /// rung too early. Ignored when `gate == false`.
    pub tau_split: f64,
}

impl Default for WriteOpts {
    fn default() -> Self {
        Self {
            gate: false,
            tau_cohere: 0.1,
            tau_split: 0.3,
        }
    }
}

/// Legacy three-phase adaptive write. Unchanged behavior: delegates to the
/// two-field rule with `address == counter` and the gate off, which reproduces
/// the original pipeline exactly.
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
    adaptive_write_two(
        input,
        input,
        locations,
        config,
        eta,
        next_id,
        rng,
        landmarks,
        id_lookup,
        WriteOpts::default(),
    )
}

/// Two-field adaptive write: route/migrate by `address`, accumulate `counter`.
///
/// With `opts.gate == false` and `address == counter` this is the legacy
/// pipeline byte-for-byte. With the gate on it becomes the consolidation
/// primitive: among address-activated candidates the winner is the one whose
/// accumulated counter coheres with the incoming counter; if none cohere (or
/// none are close in address) it spawns. Gated joins are winner-take-all
/// (k = 1) and accumulate the counter at full weight, keeping each family's
/// law pure.
#[allow(clippy::too_many_arguments)]
pub fn adaptive_write_two(
    address: &[f64],
    counter: &[f64],
    locations: &mut [HardLocation],
    config: &EAMConfig,
    eta: f64,
    next_id: &mut u64,
    rng: &mut impl Rng,
    landmarks: &[usize],
    id_lookup: &[u32],
    opts: WriteOpts,
) -> WriteResult {
    // Non-competitive mode: verbatim streaming append. Store the address
    // as-is (NO normalization) as a fresh location — no activation, no
    // merge, no migration. Makes the collection an exact growing key→value
    // store, so a raw dot read (`read_attention`) reproduces trained
    // attention bit-exactly. Memory is bounded elsewhere (an explicit
    // `merge()` pass), not by the write competing online.
    if !config.competitive {
        let id = LocationId(*next_id);
        *next_id += 1;
        let mut new_loc = HardLocation::new(id, address.to_vec());
        new_loc.counter = counter.to_vec();
        new_loc.write_count = 1.0;
        return WriteResult {
            modified_indices: vec![],
            new_locations: vec![new_loc],
            eta,
        };
    }

    let k = config.k.min(locations.len());

    // ── Phase 1: Select ─────────────────────────────────────────────
    // k-nearest via graph search (falls back to SoA/brute force when graph is young).
    let (indices, sims) = read::activate_auto(address, locations, k, landmarks, id_lookup);

    if indices.is_empty() {
        // Cold start: the index is empty (data-seeded, l_0 == 0). Seed the first
        // hard location directly from the input rather than no-op'ing — this is
        // how a data-seeded index grows its initial codebook. Subsequent writes
        // activate against it and either consolidate or novelty-split.
        let id = LocationId(*next_id);
        *next_id += 1;
        let mut new_loc = HardLocation::new(id, vec_ops::normalize(address));
        new_loc.counter = counter.to_vec();
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

    // Winner selection. Legacy: conscience S_eff = S(x, a_j) - γ · n_j / Σn_i.
    // Gated: among candidates close enough in ADDRESS to share a context
    // (sim ≥ tau_split), the winner is the one whose accumulated COUNTER best
    // coheres with the incoming counter; if the best coherence is below
    // tau_cohere (or nothing is close in address) we force a novelty spawn.
    let mut force_spawn = false;
    let winner_local = if opts.gate {
        let cn = vec_ops::normalize(counter);
        let strong: Vec<usize> = (0..indices.len())
            .filter(|&li| sims[li] >= opts.tau_split)
            .collect();
        if strong.is_empty() {
            force_spawn = true;
            0
        } else {
            let best = *strong
                .iter()
                .max_by(|&&a, &&b| {
                    let ca = vec_ops::cosine_similarity(&locations[indices[a]].counter, &cn);
                    let cb = vec_ops::cosine_similarity(&locations[indices[b]].counter, &cn);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let best_coh = vec_ops::cosine_similarity(&locations[indices[best]].counter, &cn);
            if best_coh < opts.tau_cohere {
                force_spawn = true;
            }
            best
        }
    } else {
        let total_writes: f64 = indices.iter().map(|&i| locations[i].write_count).sum();
        if total_writes > 1e-12 {
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
        }
    };
    let winner_global = indices[winner_local];

    // ── Phase 2: Update ─────────────────────────────────────────────
    // Single pass over activated set: counter accumulation + address migration.
    // Under the gate a join is winner-take-all (k = 1) and accumulates the
    // counter at full weight; a forced spawn touches no existing location.
    let eta_winner = eta / (1.0 + locations[winner_global].write_count / config.tau_damp);
    let eta_neighbor = eta_winner * 0.1;

    if !force_spawn {
        for (local_idx, &global_idx) in indices.iter().enumerate() {
            let is_winner = local_idx == winner_local;
            if opts.gate && !is_winner {
                continue; // gated join is winner-only — keep each family's law pure
            }
            let loc = &mut locations[global_idx];

            // Counter accumulation: c_j += w_j · counter, n_j += w_j
            // (gated winner accumulates at full weight, not address-weighted).
            let w = if opts.gate { 1.0 } else { weights[local_idx] };
            vec_ops::add_scaled(&mut loc.counter, counter, w);
            loc.write_count += w;

            // Address migration. Legacy: winner at η_eff, neighbors at 10%.
            // Gated: a running MEAN of the routing vectors (lr = 1/n) so a
            // family's address converges to its context prototype with the
            // per-member content averaged out — without that, the prototype
            // stays stuck near the first member and higher rungs can't group
            // same-law families.
            let lr = if opts.gate {
                1.0 / loc.write_count
            } else if is_winner {
                eta_winner
            } else {
                eta_neighbor
            };
            let diff: Vec<f64> = address
                .iter()
                .zip(loc.address.iter())
                .map(|(x, a)| x - a)
                .collect();
            vec_ops::add_scaled(&mut loc.address, &diff, lr);
            loc.address = vec_ops::normalize(&loc.address);
        }
    }

    // MDL: the joining write adds its residual surprise (how poorly it matched the
    // winner) to the winner's debt. surprise·recurrence accrues here, one write at
    // a time, until it clears an engram and the location splits in Phase 3.
    if config.mdl_gate && !force_spawn {
        locations[winner_global].surprise_mass += surprise_bits(sims[winner_local]);
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

    // Novelty spawn. Legacy: no location is close enough in address
    // (max_sim < tau_split). Gated: the address matched but the counter didn't
    // cohere with any candidate's law (force_spawn) — a same-context-different-
    // law episode gets its own family.
    let spawn_novelty = if opts.gate {
        force_spawn
    } else if config.mdl_gate {
        // MDL novelty: spawn immediately only when a single contact is surprising
        // enough to pay for its own engram outright. The bar is log2(L+1)+log2(d/32),
        // not a fixed tau_split — it rises as the index fills.
        surprise_bits(max_sim) > engram_bits(locations.len(), config.d)
    } else {
        max_sim < config.tau_split
    };
    if spawn_novelty {
        let id = LocationId(*next_id);
        *next_id += 1;
        let addr = vec_ops::normalize(address);
        let mut new_loc = HardLocation::new(id, addr);
        new_loc.counter = counter.to_vec();
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

    // Recurrence split: a saturated winner spawns a perturbed neighbor. Legacy
    // fires on write_count alone (blind to coherence). MDL fires when accumulated
    // surprise·recurrence (surprise_mass, bits) clears an engram — a high-traffic
    // but coherent location carries ~0 debt and does NOT split.
    let overloaded = if config.mdl_gate {
        locations[winner_global].surprise_mass > engram_bits(locations.len(), config.d)
    } else {
        locations[winner_global].write_count > config.tau_overload
    };
    if !force_spawn && overloaded {
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
        // Debt is paid by the split: the refined pair now represents the traffic.
        locations[winner_global].surprise_mass *= 0.5;

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

    // MDL gate: a saturated but COHERENT location does not split, where the
    // legacy count-only overload rule would. Recurrence without surprise carries
    // no description-length debt.
    #[test]
    fn test_mdl_coherent_traffic_does_not_split() {
        let mut config = EAMConfig::new(64).unwrap();
        config.mdl_gate = true;
        let mut next_id = 1u64;
        let mut rng = rand::thread_rng();
        let input = vec_ops::normalize(&{
            let mut v = vec![0.0; 64];
            v[0] = 1.0;
            v[1] = 0.3;
            v
        });
        let mut locations = vec![HardLocation::new(LocationId(0), input.clone())];

        // Far more identical writes than tau_overload (=8 at d=64): legacy splits, MDL must not.
        let n = (config.tau_overload as usize) + 20;
        for _ in 0..n {
            let r = adaptive_write(
                &input,
                &mut locations,
                &config,
                0.01,
                &mut next_id,
                &mut rng,
                &[],
                &[],
            );
            for loc in r.new_locations {
                locations.push(loc);
            }
        }
        assert!(locations[0].write_count > config.tau_overload);
        assert!(locations[0].surprise_mass < 1.0); // ~0 debt: every write matched
        assert_eq!(
            locations.len(),
            1,
            "coherent traffic must not split under MDL"
        );
    }

    // MDL gate: a location that absorbs INCOHERENT traffic accrues surprise debt
    // and splits once it clears an engram — surprise·recurrence > bits.
    #[test]
    fn test_mdl_incoherent_traffic_splits() {
        let mut config = EAMConfig::new(64).unwrap();
        config.mdl_gate = true;
        config.tau_split = 0.0; // disable novelty spawn: force everything to JOIN, so debt accrues
        let mut next_id = 1u64;
        let mut rng = rand::thread_rng();
        let seed = vec_ops::normalize(&{
            let mut v = vec![0.0; 64];
            v[0] = 1.0;
            v
        });
        let mut locations = vec![HardLocation::new(LocationId(0), seed)];

        let mut split = false;
        for i in 0..200 {
            // moderately-similar-but-varying inputs: each matches the winner only
            // partially, so each contributes surprise debt.
            let mut v = vec![0.0; 64];
            v[0] = 1.0;
            v[2 + (i % 40)] = 0.9;
            let x = vec_ops::normalize(&v);
            let r = adaptive_write(
                &x,
                &mut locations,
                &config,
                0.01,
                &mut next_id,
                &mut rng,
                &[],
                &[],
            );
            if !r.new_locations.is_empty() {
                split = true;
                break;
            }
        }
        assert!(
            split,
            "incoherent traffic must accrue surprise debt and split under MDL"
        );
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
