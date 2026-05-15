//! Full Rao & Ballard 1999 — three-layer hierarchical predictive coding
//! on the substrate.
//!
//! The previous `predictive_coding.rs` demo proved a *two-layer*
//! predictive-coding loop works. The actual cortical architecture
//! Rao & Ballard described is **stacked**: every layer predicts the
//! layer below, residuals flow upward, and abstract features emerge
//! at the higher levels because they explain shared structure across
//! many lower-level patterns.
//!
//! Here we wire three layers:
//!
//!     L3 (super-categories)
//!      ↕  bidirectional binding via `bridge23`
//!     L2 (sub-categories)
//!      ↕  bidirectional binding via `bridge12`
//!     L1 (raw noisy input)
//!
//! Data is hierarchical by construction: there are 2 *super-prototypes*,
//! and each has 3 *sub-prototypes* derived from it (super + structured
//! noise). Each sub-prototype generates a stream of noisy samples
//! (sub + more noise). After training, we should see:
//!
//!   - L2 discovers 6 sub-categories (one per sub-prototype).
//!   - L3 discovers 2 super-categories (one per super-prototype),
//!     each grouping 3 L2 sub-categories together.
//!   - Top-down predictions from a single L3 state reconstruct all 3
//!     of its sub-prototypes — the substrate has learned the
//!     generative hierarchy end-to-end.
//!
//! Sections:
//!   1. Generate hierarchical data, show 2 supers × 3 subs × N samples.
//!   2. Train the 3-layer hierarchy; report L2 and L3 cluster purity.
//!   3. Top-down generative pass: from each L3 state, reconstruct down
//!      through L2 to L1; compare reconstructions to true prototypes.
//!   4. Layer specialization: show that L3 has learned abstract features
//!      (sub-cats from the same super share more L3 mass than sub-cats
//!      from different supers).
//!
//! Run: `cargo run --release --example hierarchical_pc -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

// ----- Hyperparameters ------------------------------------------------------

const DIM: usize = 512;
const N_SUPERS: usize = 2;
const N_SUBS_PER_SUPER: usize = 3;
const N_SAMPLES_PER_SUB: usize = 20;
const SUPER_TO_SUB_NOISE: f64 = 0.35;   // how different sub-prototypes are within a super
const SUB_TO_INSTANCE_NOISE: f64 = 0.45;
// Distinct thresholds per layer — tighter at L2 (separate sub-cats),
// looser at L3 (merge sub-cats from same super into one super-cat).
const L2_THRESHOLD: f64 = 0.22;
const L3_THRESHOLD: f64 = 0.45;
// Slow learning: a surprise doesn't immediately create a category. It
// goes into a candidate buffer; only when a *similar* surprise recurs
// PROMOTION_COUNT times does it get promoted to a real category. This
// separates one-off noise from genuine new structure.
const PROMOTION_COUNT: usize = 2;
// Recurrence threshold: must be ABOVE the cross-sub cosine (0.74)
// so different sub-prototypes don't get falsely merged.
const CANDIDATE_RECURRENCE_THR: f64 = 0.78;
const BETA: f64 = 12.0;
const READ_ITERS: usize = 4;
const PC_INFERENCE_ITERS: usize = 6;
const PC_LR: f64 = 0.3;
const CONVERGENCE_EPS: f64 = 0.001;

// ----- Helpers --------------------------------------------------------------

fn rand_unit(rng: &mut StdRng) -> Vec<f64> { vec_ops::random_unit_vector(DIM, rng) }

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

fn add_noise(v: &[f64], magnitude: f64, rng: &mut StdRng) -> Vec<f64> {
    let raw: Vec<f64> = (0..v.len()).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect();
    let noise_dir = vec_ops::normalize(&raw);
    let mixed: Vec<f64> = v.iter().zip(noise_dir.iter())
        .map(|(a, b)| a + magnitude * b).collect();
    vec_ops::normalize(&mixed)
}

// ----- Mini EAM -------------------------------------------------------------

struct MiniEAM { locs: Vec<HardLocation>, next_id: u64 }
impl MiniEAM {
    fn new() -> Self { Self { locs: Vec::new(), next_id: 0 } }
    fn write(&mut self, pattern: &[f64]) {
        let address = vec_ops::normalize(pattern);
        let mut loc = HardLocation::new(LocationId(self.next_id), address);
        loc.counter = pattern.to_vec();
        loc.write_count = 1.0;
        self.locs.push(loc);
        self.next_id += 1;
    }
    fn read(&self, query: &[f64]) -> Vec<f64> {
        if self.locs.is_empty() { return vec_ops::normalize(query); }
        let mut q = vec_ops::normalize(query);
        for _ in 0..READ_ITERS {
            let sims: Vec<f64> =
                self.locs.iter().map(|l| vec_ops::dot(&l.address, &q)).collect();
            let w = vec_ops::softmax(&sims, BETA);
            let patterns: Vec<&[f64]> =
                self.locs.iter().map(|l| l.counter.as_slice()).collect();
            q = vec_ops::normalize(&vec_ops::weighted_sum(&patterns, &w));
        }
        q
    }
}

// ----- A generic "bridge" between two layers --------------------------------
// Stores bindings (above_state, below_state). Supports both directions and
// has separate cleanup lexicons for each side.

struct Bridge {
    eam: MiniEAM,
    above_key: Vec<f64>,
    below_key: Vec<f64>,
    above_states: Vec<Vec<f64>>,   // category lexicon for the upper layer
    below_states: Vec<Vec<f64>>,   // exemplar lexicon for the lower layer
    /// Per-category running mean of residual norm (precision proxy).
    /// Lower mean residual = higher precision = sharper predictions.
    precisions: Vec<f64>,
    /// Candidate buffer — surprises seen so far that haven't been
    /// promoted to categories yet. Each entry is (proposed_above_state,
    /// triggering_below_input, recurrence_count).
    candidate_states: Vec<Vec<f64>>,
    candidate_inputs: Vec<Vec<f64>>,
    candidate_counts: Vec<usize>,
}
impl Bridge {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            above_key: rand_unit(rng),
            below_key: rand_unit(rng),
            above_states: Vec::new(),
            below_states: Vec::new(),
            precisions: Vec::new(),
            candidate_states: Vec::new(),
            candidate_inputs: Vec::new(),
            candidate_counts: Vec::new(),
        }
    }

    /// Slow-learning gate: route a surprise through the candidate
    /// buffer. Returns Some(category_idx) if the candidate has been
    /// seen PROMOTION_COUNT times and is now a real category. Returns
    /// None if it's still pending (count < threshold).
    ///
    /// Best-match (greedy) candidate selection — the closest existing
    /// candidate gets the recurrence count, not the first one above
    /// threshold. Avoids cross-category confusion when candidates
    /// from different but similar prototypes coexist in the buffer.
    fn surprise(&mut self, below: &[f64], proposed_state: Vec<f64>) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..self.candidate_inputs.len() {
            let sim = vec_ops::cosine_similarity(below, &self.candidate_inputs[i]);
            if sim > CANDIDATE_RECURRENCE_THR {
                if best.map(|(_, s)| sim > s).unwrap_or(true) {
                    best = Some((i, sim));
                }
            }
        }
        if let Some((i, _)) = best {
            self.candidate_counts[i] += 1;
            if self.candidate_counts[i] >= PROMOTION_COUNT {
                let state = self.candidate_states[i].clone();
                let cat_idx = self.add_category(state.clone());
                self.write(&state, below);
                self.candidate_states.remove(i);
                self.candidate_inputs.remove(i);
                self.candidate_counts.remove(i);
                return Some(cat_idx);
            }
            return None;
        }
        // No matching candidate — store as new candidate
        self.candidate_states.push(proposed_state);
        self.candidate_inputs.push(below.to_vec());
        self.candidate_counts.push(1);
        None
    }
    fn predict_below_raw(&self, above: &[f64]) -> Vec<f64> {
        let q = bind_vec(&self.above_key, above);
        let r = self.eam.read(&q);
        vec_ops::normalize(&unbind_vec(&r, &self.below_key))
    }
    fn infer_above_raw(&self, below: &[f64]) -> Vec<f64> {
        let q = bind_vec(&self.below_key, below);
        let r = self.eam.read(&q);
        vec_ops::normalize(&unbind_vec(&r, &self.above_key))
    }
    fn cleanup_above(&self, noisy: &[f64]) -> (Vec<f64>, usize) {
        if self.above_states.is_empty() { return (noisy.to_vec(), 0); }
        let mut best = 0usize; let mut bs = f64::NEG_INFINITY;
        for (i, s) in self.above_states.iter().enumerate() {
            let cs = vec_ops::cosine_similarity(noisy, s);
            if cs > bs { bs = cs; best = i; }
        }
        (self.above_states[best].clone(), best)
    }
    fn cleanup_below(&self, noisy: &[f64]) -> Vec<f64> {
        if self.below_states.is_empty() { return noisy.to_vec(); }
        let mut best = 0usize; let mut bs = f64::NEG_INFINITY;
        for (i, s) in self.below_states.iter().enumerate() {
            let cs = vec_ops::cosine_similarity(noisy, s);
            if cs > bs { bs = cs; best = i; }
        }
        self.below_states[best].clone()
    }
    fn write(&mut self, above: &[f64], below: &[f64]) {
        let entry = bundle(&[
            &bind_vec(&self.above_key, above),
            &bind_vec(&self.below_key, below),
        ]);
        self.eam.write(&entry);
        self.below_states.push(below.to_vec());
    }
    fn add_category(&mut self, above: Vec<f64>) -> usize {
        let idx = self.above_states.len();
        self.above_states.push(above);
        self.precisions.push(1.0); // start neutral
        idx
    }
    fn update_precision(&mut self, cat_idx: usize, residual_norm: f64) {
        // Exponential moving average of residual norm
        let alpha = 0.2;
        let curr = self.precisions[cat_idx];
        // Convert running residual to precision (low residual → high precision)
        let new = (1.0 - alpha) * curr + alpha * (1.0 / (residual_norm.max(0.05)));
        self.precisions[cat_idx] = new;
    }
}

// ----- Iterative inference over a single bridge ----------------------------

/// Run the Rao-Ballard inference loop at one level. Returns the converged
/// above-state, the cleaned category index, and the final 1−cosine error
/// in the cleaned-up space (NOT raw residual norm).
fn infer_iterative(bridge: &Bridge, below: &[f64]) -> (Vec<f64>, usize, f64) {
    if bridge.above_states.is_empty() {
        return (below.to_vec(), 0, 1.0);
    }
    let mut state = bridge.infer_above_raw(below);
    let mut last_norm = f64::INFINITY;
    for iter in 0..PC_INFERENCE_ITERS {
        let pred_below = bridge.predict_below_raw(&state);
        let residual: Vec<f64> = below.iter().zip(pred_below.iter())
            .map(|(a, p)| a - p).collect();
        let r_norm = vec_ops::l2_norm(&residual);
        if iter > 0 && (last_norm - r_norm).abs() < CONVERGENCE_EPS { break; }
        last_norm = r_norm;
        let delta = bridge.infer_above_raw(&residual);
        for i in 0..state.len() {
            state[i] = (1.0 - PC_LR) * state[i] + PC_LR * delta[i];
        }
        state = vec_ops::normalize(&state);
    }
    let (cleaned, idx) = bridge.cleanup_above(&state);
    let pred = bridge.predict_below_raw(&cleaned);
    let cleaned_pred = bridge.cleanup_below(&pred);
    let cos_err = (1.0 - vec_ops::cosine_similarity(below, &cleaned_pred)).max(0.0);
    (cleaned, idx, cos_err)
}

// ----- The Hierarchy --------------------------------------------------------

struct Hierarchy {
    bridge12: Bridge,  // l2 ↔ l1
    bridge23: Bridge,  // l3 ↔ l2
}
impl Hierarchy {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            bridge12: Bridge::new(rng),
            bridge23: Bridge::new(rng),
        }
    }

    /// Process one input through the hierarchy. Returns Some((l2_cat,
    /// l3_cat)) if the input was successfully categorized, or None if
    /// it's currently in pending limbo (the slow-learning gate is
    /// waiting for recurrence before promoting it). Pending inputs
    /// are NOT written to any existing category — they don't pollute.
    fn step(&mut self, l1: &[f64], _rng: &mut StdRng) -> Option<(usize, usize)> {
        // ----- L2: process l1 -----
        if self.bridge12.above_states.is_empty() {
            // First-ever input bootstraps everything (no candidate gate
            // for the very first input — there's nothing to compare to).
            let new_l2 = vec_ops::normalize(l1);
            let l2_cat = self.bridge12.add_category(new_l2.clone());
            self.bridge12.write(&new_l2, l1);
            let new_l3 = new_l2.clone();
            let l3_cat = self.bridge23.add_category(new_l3.clone());
            self.bridge23.write(&new_l3, &new_l2);
            return Some((l2_cat, l3_cat));
        }
        let (_l2_state, l2_cat_guess, l2_err) = infer_iterative(&self.bridge12, l1);
        let l2_cat = if l2_err > L2_THRESHOLD {
            // Surprise: route through slow-learning gate. If pending,
            // return None for the whole step — do NOT pollute existing
            // categories with the unknown input.
            let proposed = vec_ops::normalize(l1);
            match self.bridge12.surprise(l1, proposed) {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            let chosen = self.bridge12.above_states[l2_cat_guess].clone();
            self.bridge12.write(&chosen, l1);
            self.bridge12.update_precision(l2_cat_guess, l2_err);
            l2_cat_guess
        };
        let l2_used = self.bridge12.above_states[l2_cat].clone();

        // ----- L3: process l2 -----
        if self.bridge23.above_states.is_empty() {
            let new_l3 = vec_ops::normalize(&l2_used);
            let l3_cat = self.bridge23.add_category(new_l3.clone());
            self.bridge23.write(&new_l3, &l2_used);
            return Some((l2_cat, l3_cat));
        }
        let (_l3_state, l3_cat_guess, l3_err) = infer_iterative(&self.bridge23, &l2_used);
        let l3_cat = if l3_err > L3_THRESHOLD {
            let proposed = vec_ops::normalize(&l2_used);
            match self.bridge23.surprise(&l2_used, proposed) {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            let chosen = self.bridge23.above_states[l3_cat_guess].clone();
            self.bridge23.write(&chosen, &l2_used);
            self.bridge23.update_precision(l3_cat_guess, l3_err);
            l3_cat_guess
        };
        Some((l2_cat, l3_cat))
    }

    /// Top-down generative pass: given an L3 category, predict down to L1.
    /// Returns the reconstructed L1 vector (cleaned against L1 exemplars).
    fn generate_from_l3(&self, l3_cat: usize) -> Vec<f64> {
        let l3 = &self.bridge23.above_states[l3_cat];
        let predicted_l2 = self.bridge23.predict_below_raw(l3);
        let cleaned_l2 = self.bridge23.cleanup_below(&predicted_l2);
        let predicted_l1 = self.bridge12.predict_below_raw(&cleaned_l2);
        self.bridge12.cleanup_below(&predicted_l1)
    }
}

// ----- Data generation ------------------------------------------------------

struct DataGen {
    supers: Vec<Vec<f64>>,
    subs: Vec<Vec<f64>>,            // 6 sub-prototypes
    sub_to_super: Vec<usize>,       // sub_idx → super_idx
}

fn make_data(rng: &mut StdRng) -> (DataGen, Vec<(Vec<f64>, usize, usize)>) {
    // Generate super-prototypes
    let supers: Vec<Vec<f64>> = (0..N_SUPERS).map(|_| rand_unit(rng)).collect();
    // Each super gets 3 sub-prototypes (super + structured noise)
    let mut subs: Vec<Vec<f64>> = Vec::new();
    let mut sub_to_super: Vec<usize> = Vec::new();
    for (super_idx, sup) in supers.iter().enumerate() {
        for _ in 0..N_SUBS_PER_SUPER {
            subs.push(add_noise(sup, SUPER_TO_SUB_NOISE, rng));
            sub_to_super.push(super_idx);
        }
    }
    // Samples: per sub, generate N noisy instances
    let mut samples: Vec<(Vec<f64>, usize, usize)> = Vec::new();
    for (sub_idx, sub) in subs.iter().enumerate() {
        let super_idx = sub_to_super[sub_idx];
        for _ in 0..N_SAMPLES_PER_SUB {
            samples.push((add_noise(sub, SUB_TO_INSTANCE_NOISE, rng), sub_idx, super_idx));
        }
    }
    use rand::seq::SliceRandom;
    samples.shuffle(rng);
    (DataGen { supers, subs, sub_to_super }, samples)
}

fn purity(assignments: &[(usize, usize)], n_cats: usize) -> f64 {
    let mut by_cat: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (cat, truth) in assignments {
        *by_cat.entry(*cat).or_default().entry(*truth).or_insert(0) += 1;
    }
    let mut hits = 0;
    for cat in 0..n_cats {
        if let Some(counts) = by_cat.get(&cat) {
            hits += counts.values().max().copied().unwrap_or(0);
        }
    }
    hits as f64 / assignments.len() as f64
}

// ===========================================================================
//                  SECTION 1 — Hierarchical data
// ===========================================================================

fn section_1(data: &DataGen, samples: &[(Vec<f64>, usize, usize)]) {
    println!("======================================================================");
    println!("  SECTION 1 — Hierarchical synthetic data");
    println!("======================================================================");
    println!("Generated {} super-prototypes, each with {} sub-prototypes,",
        N_SUPERS, N_SUBS_PER_SUPER);
    println!("each producing {} noisy instances. Total: {} samples.\n",
        N_SAMPLES_PER_SUB, samples.len());

    // Sanity: within-super cosines vs across-super
    let mut within = 0.0; let mut wc = 0;
    let mut across = 0.0; let mut ac = 0;
    for i in 0..data.subs.len() {
        for j in (i+1)..data.subs.len() {
            let cs = vec_ops::cosine_similarity(&data.subs[i], &data.subs[j]);
            if data.sub_to_super[i] == data.sub_to_super[j] {
                within += cs; wc += 1;
            } else {
                across += cs; ac += 1;
            }
        }
    }
    println!("  Sub-prototype cosines:");
    println!("    within-super:  {:.3} (siblings — should be moderate-high)", within / wc as f64);
    println!("    across-super:  {:.3} (cousins — should be low)", across / ac as f64);

    // Sample-level cosines
    let mut within_sub = 0.0; let mut wsc = 0;
    let mut within_sup_diff_sub = 0.0; let mut wsdc = 0;
    let mut across_sup = 0.0; let mut asc = 0;
    for i in 0..samples.len() {
        for j in (i+1)..samples.len() {
            let cs = vec_ops::cosine_similarity(&samples[i].0, &samples[j].0);
            if samples[i].1 == samples[j].1 { within_sub += cs; wsc += 1; }
            else if samples[i].2 == samples[j].2 { within_sup_diff_sub += cs; wsdc += 1; }
            else { across_sup += cs; asc += 1; }
        }
    }
    println!("\n  Sample cosines:");
    println!("    same sub:                {:.3}", within_sub / wsc as f64);
    println!("    same super, diff sub:    {:.3}", within_sup_diff_sub / wsdc as f64);
    println!("    different super:         {:.3}", across_sup / asc as f64);
    println!();
    println!("  The structure: same-sub > same-super-diff-sub > different-super.");
    println!("  A successful hierarchy will recover both granularities.\n");
}

// ===========================================================================
//                  SECTION 2 — Train and report cluster purity
// ===========================================================================

fn section_2(samples: &[(Vec<f64>, usize, usize)], rng: &mut StdRng) -> Hierarchy {
    println!("======================================================================");
    println!("  SECTION 2 — Train 3-layer hierarchy");
    println!("======================================================================\n");

    let mut h = Hierarchy::new(rng);
    let mut l2_assignments: Vec<(usize, usize)> = Vec::new(); // (cat, true_sub)
    let mut l3_assignments: Vec<(usize, usize)> = Vec::new(); // (cat, true_super)

    let mut pending = 0;
    for (l1, sub_idx, super_idx) in samples {
        match h.step(l1, rng) {
            Some((l2_cat, l3_cat)) => {
                l2_assignments.push((l2_cat, *sub_idx));
                l3_assignments.push((l3_cat, *super_idx));
            }
            None => { pending += 1; }
        }
    }
    println!("  Categorized: {} / {}   (pending: {} — slow-learning gate held them back)",
        l2_assignments.len(), samples.len(), pending);

    let n_l2 = h.bridge12.above_states.len();
    let n_l3 = h.bridge23.above_states.len();
    let p_l2 = purity(&l2_assignments, n_l2);
    let p_l3 = purity(&l3_assignments, n_l3);

    println!("  L2 (sub-categories):");
    println!("    discovered {} categories (true: {} subs)", n_l2, N_SUPERS * N_SUBS_PER_SUPER);
    println!("    cluster purity vs true sub: {:.0}%", p_l2 * 100.0);
    println!();
    println!("  L3 (super-categories):");
    println!("    discovered {} categories (true: {} supers)", n_l3, N_SUPERS);
    println!("    cluster purity vs true super: {:.0}%", p_l3 * 100.0);

    // L3 grouping: do L3 categories aggregate multiple L2 categories?
    let mut l3_to_l2: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for ((l2c, _), (l3c, _)) in l2_assignments.iter().zip(l3_assignments.iter()) {
        l3_to_l2.entry(*l3c).or_default().insert(*l2c);
    }
    println!("\n  L3 → L2 grouping (each L3 should cover {} L2 sub-cats):",
        N_SUBS_PER_SUPER);
    for l3c in 0..n_l3 {
        let group = l3_to_l2.get(&l3c).cloned().unwrap_or_default();
        let mut sorted: Vec<_> = group.into_iter().collect();
        sorted.sort();
        println!("    L3 cat {}: covers L2 cats {:?}", l3c, sorted);
    }

    assert!(p_l2 > 0.85, "L2 purity too low: {:.0}%", p_l2 * 100.0);
    assert!(p_l3 > 0.85, "L3 purity too low: {:.0}%", p_l3 * 100.0);
    println!("\n  PASS: hierarchy recovers both granularities.\n");
    h
}

// ===========================================================================
//          SECTION 3 — Top-down generative pass through 3 layers
// ===========================================================================

fn section_3(h: &Hierarchy, data: &DataGen) {
    println!("======================================================================");
    println!("  SECTION 3 — Top-down generative: L3 → L2 → L1");
    println!("======================================================================");
    println!("From a single L3 super-category state, predict down through L2");
    println!("to L1. The reconstructed L1 should resemble the corresponding");
    println!("super-prototype (averaged over the 3 sub-prototypes under it).\n");

    let bar = |s: f64| -> String {
        let w = (s.clamp(0.0, 1.0) * 30.0) as usize;
        (0..30).map(|i| if i < w { '█' } else { '░' }).collect()
    };

    let mut total = 0.0;
    let mut counted = 0;
    for l3c in 0..h.bridge23.above_states.len() {
        let recon = h.generate_from_l3(l3c);
        // Match this L3 cat to the best super-prototype
        let mut best_super = 0; let mut bs = f64::NEG_INFINITY;
        for (s_idx, sup) in data.supers.iter().enumerate() {
            let cs = vec_ops::cosine_similarity(&recon, sup);
            if cs > bs { bs = cs; best_super = s_idx; }
        }
        println!("  L3 cat {}  →  super-prototype S{}   cos={:.3}  {}",
            l3c, best_super, bs, bar(bs.max(0.0)));
        total += bs.max(0.0);
        counted += 1;
    }
    let avg = total / counted.max(1) as f64;
    println!("\n  Average top-down reconstruction cosine: {:.3}", avg);
    println!("  (Each L3 reconstruction is the AVERAGE of its 3 sub-prototypes;");
    println!("   sub-to-super noise means cos ≈ 0.7-0.9 is the expected ceiling.)");

    assert!(avg > 0.55,
        "top-down reconstruction too weak: avg cos {:.3}", avg);
    println!("\n  PASS: substrate runs the full hierarchy in reverse.\n");
}

// ===========================================================================
//          SECTION 4 — Layer specialization: abstraction at L3
// ===========================================================================

fn section_4(h: &Hierarchy, samples: &[(Vec<f64>, usize, usize)]) {
    println!("======================================================================");
    println!("  SECTION 4 — Layer specialization: L3 abstracts, L2 specializes");
    println!("======================================================================");
    println!("Within-super same-sub-cat similarity vs within-super different-sub-cat:");
    println!("If L3 truly abstracts, its representations of two samples from");
    println!("the SAME SUPER should be more similar than at L2 (which separates");
    println!("by sub).\n");

    // For each sample, get its L2 and L3 cat
    let mut l2_per_sample: Vec<usize> = Vec::new();
    let mut l3_per_sample: Vec<usize> = Vec::new();
    for (l1, _, _) in samples {
        // Re-run inference (no writes, just classify)
        let (l2_cleaned, l2_idx, _) = infer_iterative(&h.bridge12, l1);
        let (_l3_state, l3_idx, _) = infer_iterative(&h.bridge23, &l2_cleaned);
        l2_per_sample.push(l2_idx);
        l3_per_sample.push(l3_idx);
    }

    // Compute fractions
    let mut same_sub_same_l3 = 0; let mut same_sub_count = 0;
    let mut diff_sub_same_super_same_l3 = 0; let mut diff_sub_same_super_count = 0;
    let mut diff_super_same_l3 = 0; let mut diff_super_count = 0;
    for i in 0..samples.len() {
        for j in (i+1)..samples.len() {
            let same_l3 = l3_per_sample[i] == l3_per_sample[j];
            if samples[i].1 == samples[j].1 {
                same_sub_count += 1;
                if same_l3 { same_sub_same_l3 += 1; }
            } else if samples[i].2 == samples[j].2 {
                diff_sub_same_super_count += 1;
                if same_l3 { diff_sub_same_super_same_l3 += 1; }
            } else {
                diff_super_count += 1;
                if same_l3 { diff_super_same_l3 += 1; }
            }
        }
    }
    let f1 = same_sub_same_l3 as f64 / same_sub_count.max(1) as f64;
    let f2 = diff_sub_same_super_same_l3 as f64 / diff_sub_same_super_count.max(1) as f64;
    let f3 = diff_super_same_l3 as f64 / diff_super_count.max(1) as f64;

    println!("  Fraction of sample pairs that share an L3 category:");
    println!("    same sub:                  {:.0}%   (trivially yes if L2/L3 are correct)", f1 * 100.0);
    println!("    same super, diff sub:      {:.0}%   (the abstraction signal)", f2 * 100.0);
    println!("    different super:            {:.0}%   (should be near 0)", f3 * 100.0);
    println!();
    println!("  Interpretation:");
    println!("    - f1 high (~100%): samples from same sub map to same L3 (as they should).");
    println!("    - f2 high: L3 has ABSTRACTED across sub-categories. This is the key signal.");
    println!("    - f3 low: super-categories are properly separated.");

    assert!(f2 > 0.80, "L3 should abstract across sub-categories: got {:.0}%", f2 * 100.0);
    assert!(f3 < 0.15, "L3 should not mix different supers: got {:.0}%", f3 * 100.0);
    println!("\n  PASS: hierarchical abstraction empirically demonstrated.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let (data, samples) = make_data(&mut rng);
    section_1(&data, &samples);
    let h = section_2(&samples, &mut rng);
    section_3(&h, &data);
    section_4(&h, &samples);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — full Rao-Ballard hierarchy on substrate");
    println!("======================================================================");
    println!("  Three layers. Iterative inference at each. Top-down generative");
    println!("  reconstruction end-to-end. Abstraction emerges at the higher");
    println!("  layer. No backprop, no autograd, no global gradient anywhere.");
    println!("  Rao & Ballard 1999 made operational, 27 years later.");
}
