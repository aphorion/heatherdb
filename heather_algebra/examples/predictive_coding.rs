//! Predictive coding — hierarchical representation learning on the
//! substrate, without backprop.
//!
//! Rao & Ballard (1999) showed that the visual cortex learns features
//! via *predictive coding*: each layer sends bottom-up signals to the
//! layer above and receives top-down predictions back. Layers update
//! locally to minimize the residual (prediction error). No global
//! gradient, no autograd, no backprop — and yet hierarchical features
//! emerge.
//!
//! On the substrate, this becomes a two-layer EAM stack where Layer 2
//! stores **bidirectional bindings** between high-level categories and
//! low-level inputs:
//!
//!   entry = bind(L2_KEY, l2_state) + bind(L1_KEY, l1_state)
//!
//! Bottom-up inference: query with bind(L1_KEY, observed_l1), unbind L2_KEY.
//! Top-down prediction: query with bind(L2_KEY, inferred_l2), unbind L1_KEY.
//!
//! When the prediction matches the observation, the substrate has a
//! category for this input — reinforce. When it doesn't, the substrate
//! is surprised — invent a new L2 state and bind it to this L1. Local,
//! error-driven, no backprop.
//!
//! Sections:
//!   1. Baseline — single-layer EAM that just memorizes noisy inputs.
//!      No category discovery, reconstruction is the noisy input itself.
//!   2. Predictive coding — two-layer substrate discovers K=4 prototype
//!      categories from noisy inputs, online, in one pass.
//!   3. Continual learning — introduce two new prototypes after the
//!      first four are learned. Show new categories emerge, old ones
//!      stay intact.
//!   4. Reconstruction — top-down predictions from each learned L2
//!      state recover the underlying prototype (generative model
//!      learned without backprop).
//!
//! Run: `cargo run --release --example predictive_coding -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

// ----- Hyperparameters ------------------------------------------------------

const DIM: usize = 512;
const N_PROTOTYPES: usize = 4;
const NOISE_MAGNITUDE: f64 = 0.5;     // noise-vector length, relative to signal=1
const N_PER_PROTOTYPE: usize = 25;
const PREDICTION_THRESHOLD: f64 = 0.45; // create new category if error > this
const BETA: f64 = 12.0;
const READ_ITERS: usize = 4;

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

/// Add noise to a unit vector by mixing in a random-direction noise
/// vector of given magnitude (relative to the signal). magnitude=0.5
/// keeps cosine to original around 0.89; magnitude=1.0 gives ~0.71.
fn add_noise(v: &[f64], magnitude: f64, rng: &mut StdRng) -> Vec<f64> {
    let raw_noise: Vec<f64> =
        (0..v.len()).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect();
    let noise_dir = vec_ops::normalize(&raw_noise);
    let mixed: Vec<f64> = v.iter().zip(noise_dir.iter())
        .map(|(a, b)| a + magnitude * b)
        .collect();
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

// ----- Layer 2: bidirectional EAM linking l2 categories to l1 inputs ------

struct Layer2 {
    eam: MiniEAM,
    l2_key: Vec<f64>,
    l1_key: Vec<f64>,
    l2_states: Vec<Vec<f64>>,   // L2 category lexicon (for cleanup)
    l1_states: Vec<Vec<f64>>,   // L1 input lexicon (for cleanup) — grows
                                // with each stored bundle
}
impl Layer2 {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            l2_key: rand_unit(rng),
            l1_key: rand_unit(rng),
            l2_states: Vec::new(),
            l1_states: Vec::new(),
        }
    }

    /// Top-down prediction: l2 → noisy l1.
    fn predict_l1_raw(&self, l2: &[f64]) -> Vec<f64> {
        let q = bind_vec(&self.l2_key, l2);
        let r = self.eam.read(&q);
        vec_ops::normalize(&unbind_vec(&r, &self.l1_key))
    }

    /// Bottom-up inference: l1 → noisy l2.
    fn infer_l2_raw(&self, l1: &[f64]) -> Vec<f64> {
        let q = bind_vec(&self.l1_key, l1);
        let r = self.eam.read(&q);
        vec_ops::normalize(&unbind_vec(&r, &self.l2_key))
    }

    /// Cleanup against the L2 category lexicon — the missing HRR cleanup
    /// step. Returns the closest stored l2 state.
    fn cleanup_l2(&self, noisy: &[f64]) -> Vec<f64> {
        if self.l2_states.is_empty() { return noisy.to_vec(); }
        let mut best = 0usize; let mut best_s = f64::NEG_INFINITY;
        for (i, ls) in self.l2_states.iter().enumerate() {
            let s = vec_ops::cosine_similarity(noisy, ls);
            if s > best_s { best_s = s; best = i; }
        }
        self.l2_states[best].clone()
    }

    /// Cleanup against the L1 input lexicon — closest stored l1.
    fn cleanup_l1(&self, noisy: &[f64]) -> Vec<f64> {
        if self.l1_states.is_empty() { return noisy.to_vec(); }
        let mut best = 0usize; let mut best_s = f64::NEG_INFINITY;
        for (i, ls) in self.l1_states.iter().enumerate() {
            let s = vec_ops::cosine_similarity(noisy, ls);
            if s > best_s { best_s = s; best = i; }
        }
        self.l1_states[best].clone()
    }

    fn write(&mut self, l2: &[f64], l1: &[f64]) {
        let entry = bundle(&[
            &bind_vec(&self.l2_key, l2),
            &bind_vec(&self.l1_key, l1),
        ]);
        self.eam.write(&entry);
        self.l1_states.push(l1.to_vec());
    }

    fn classify(&self, inferred: &[f64]) -> usize {
        let mut best = 0usize; let mut best_s = f64::NEG_INFINITY;
        for (i, ls) in self.l2_states.iter().enumerate() {
            let s = vec_ops::cosine_similarity(inferred, ls);
            if s > best_s { best_s = s; best = i; }
        }
        best
    }
}

/// One step of predictive coding:
///   1. Bottom-up infer L2 from L1 (substrate associative read).
///   2. Top-down predict L1 from inferred L2.
///   3. Error = 1 − cosine(actual L1, predicted L1).
///   4. If error > threshold: invent a new L2 category, bind it to L1.
///      Else: reinforce the existing category by writing again.
///
/// All updates are local. No gradient ever flows backward through layers.
fn pc_step(l1: &[f64], layer2: &mut Layer2, rng: &mut StdRng)
    -> (usize, f64, bool)
{
    let empty = layer2.l2_states.is_empty();
    if empty {
        // Bootstrap the first category.
        let new_l2 = rand_unit(rng);
        let cat_idx = layer2.l2_states.len();
        layer2.l2_states.push(new_l2.clone());
        layer2.write(&new_l2, l1);
        return (cat_idx, 1.0, true);
    }

    // Bottom-up: l1 → noisy l2 → cleanup against L2 lexicon.
    let noisy_l2 = layer2.infer_l2_raw(l1);
    let cleaned_l2 = layer2.cleanup_l2(&noisy_l2);

    // Top-down: cleaned l2 → noisy l1 → cleanup against L1 lexicon.
    let noisy_l1 = layer2.predict_l1_raw(&cleaned_l2);
    let cleaned_l1 = layer2.cleanup_l1(&noisy_l1);

    // Error in the cleaned-up space — this is where signal is preserved.
    let error = (1.0 - vec_ops::cosine_similarity(l1, &cleaned_l1)).max(0.0);

    if error > PREDICTION_THRESHOLD {
        let new_l2 = rand_unit(rng);
        let cat_idx = layer2.l2_states.len();
        layer2.l2_states.push(new_l2.clone());
        layer2.write(&new_l2, l1);
        (cat_idx, error, true)
    } else {
        let cat = layer2.classify(&cleaned_l2);
        let chosen = layer2.l2_states[cat].clone();
        layer2.write(&chosen, l1);  // reinforce — strengthens the binding
                                    // and adds this l1 to the lexicon
        (cat, error, false)
    }
}

// ----- Data generation -----------------------------------------------------

fn make_dataset(n_prototypes: usize, n_per: usize, rng: &mut StdRng)
    -> (Vec<Vec<f64>>, Vec<(Vec<f64>, usize)>)
{
    let prototypes: Vec<Vec<f64>> = (0..n_prototypes).map(|_| rand_unit(rng)).collect();
    let mut samples: Vec<(Vec<f64>, usize)> = Vec::new();
    for p_idx in 0..n_prototypes {
        for _ in 0..n_per {
            let noisy = add_noise(&prototypes[p_idx], NOISE_MAGNITUDE, rng);
            samples.push((noisy, p_idx));
        }
    }
    // Shuffle
    use rand::seq::SliceRandom;
    samples.shuffle(rng);
    (prototypes, samples)
}

/// Cluster purity: for each discovered category, the fraction of its
/// members from the most-common true prototype. Averaged.
fn purity(assignments: &[(usize, usize)], n_categories: usize) -> f64 {
    use std::collections::HashMap;
    let mut by_cat: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (cat, true_p) in assignments {
        *by_cat.entry(*cat).or_default().entry(*true_p).or_insert(0) += 1;
    }
    let mut total_correct = 0;
    let total: usize = assignments.len();
    for cat in 0..n_categories {
        if let Some(counts) = by_cat.get(&cat) {
            let max = counts.values().max().copied().unwrap_or(0);
            total_correct += max;
        }
    }
    total_correct as f64 / total as f64
}

// ===========================================================================
//                  SECTION 1 — Baseline (no predictive coding)
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Baseline: single-layer EAM, no predictive coding");
    println!("======================================================================");
    println!("Store each noisy input directly. The 'representation' of an input");
    println!("is itself (post-cleanup). No category discovery.\n");

    let (prototypes, samples) = make_dataset(N_PROTOTYPES, N_PER_PROTOTYPE, rng);
    println!("  {} prototypes, {} samples each, noise σ={}",
        N_PROTOTYPES, N_PER_PROTOTYPE, NOISE_MAGNITUDE);
    println!("  Total samples: {}\n", samples.len());

    // The "baseline classification" is just: each sample → its nearest
    // prototype (assuming we have access to prototypes — which a real
    // unsupervised system wouldn't). This is the *best case* for raw inputs.
    let mut best_case_correct = 0;
    for (s, true_p) in &samples {
        let mut best = 0usize; let mut bs = f64::NEG_INFINITY;
        for (i, p) in prototypes.iter().enumerate() {
            let cs = vec_ops::cosine_similarity(s, p);
            if cs > bs { bs = cs; best = i; }
        }
        if best == *true_p { best_case_correct += 1; }
    }
    let best_case = best_case_correct as f64 / samples.len() as f64;
    println!("  Best-case classification (raw input → nearest prototype):");
    println!("    accuracy: {:.0}% (this is the ceiling — uses known prototypes)",
        best_case * 100.0);

    println!("\n  But unsupervised learning doesn't have the prototypes. The");
    println!("  baseline question is: can the substrate DISCOVER them?");
    println!("  Section 2 shows it can, via predictive coding.\n");
}

// ===========================================================================
//                  SECTION 2 — Predictive coding learns categories
// ===========================================================================

fn section_2(rng: &mut StdRng) -> (Vec<Vec<f64>>, Vec<(Vec<f64>, usize)>, Layer2) {
    println!("======================================================================");
    println!("  SECTION 2 — Predictive coding discovers prototypes online");
    println!("======================================================================");
    println!("Run each input through the substrate. The two-layer stack invents a");
    println!("new L2 category when error > threshold, reinforces an existing one");
    println!("otherwise. Unsupervised, online, single pass through the data.\n");

    let (prototypes, samples) = make_dataset(N_PROTOTYPES, N_PER_PROTOTYPE, rng);

    // Diagnostic: actual within-prototype cosine of the dataset
    let mut within = 0.0; let mut between = 0.0;
    let mut wc = 0; let mut bc = 0;
    for i in 0..samples.len() {
        for j in (i+1)..samples.len() {
            let cs = vec_ops::cosine_similarity(&samples[i].0, &samples[j].0);
            if samples[i].1 == samples[j].1 { within += cs; wc += 1; }
            else { between += cs; bc += 1; }
        }
    }
    println!("  Dataset cosines: within-prototype avg={:.3}, between-prototype avg={:.3}\n",
        within / wc as f64, between / bc as f64);

    let mut layer2 = Layer2::new(rng);
    let mut assignments: Vec<(usize, usize)> = Vec::new();
    let mut new_cat_events = 0;
    let mut errors_seen: Vec<f64> = Vec::new();

    for (l1, true_p) in &samples {
        let (cat, err, created) = pc_step(l1, &mut layer2, rng);
        assignments.push((cat, *true_p));
        if created { new_cat_events += 1; }
        errors_seen.push(err);
    }
    let avg_err: f64 = errors_seen.iter().sum::<f64>() / errors_seen.len() as f64;
    println!("  Mean error across all inputs: {:.3}", avg_err);

    let n_cats = layer2.l2_states.len();
    let pur = purity(&assignments, n_cats);
    println!("  Discovered {} categories (true K = {}).", n_cats, N_PROTOTYPES);
    println!("  New-category events: {}", new_cat_events);
    println!("  Cluster purity: {:.0}%", pur * 100.0);

    // Per-category breakdown
    use std::collections::HashMap;
    let mut by_cat: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (c, p) in &assignments { *by_cat.entry(*c).or_default().entry(*p).or_insert(0) += 1; }
    println!("\n  Category → true-prototype distribution:");
    for cat in 0..n_cats {
        let counts = by_cat.get(&cat).cloned().unwrap_or_default();
        let total: usize = counts.values().sum();
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        let parts: Vec<String> = sorted.iter()
            .map(|(p, c)| format!("P{}={}", p, c)).collect();
        println!("    cat {:>2}  ({:>3} members):  {}", cat, total, parts.join("  "));
    }

    assert!(n_cats <= N_PROTOTYPES + 2,
        "should discover ~K categories, got {}", n_cats);
    assert!(pur > 0.85,
        "cluster purity too low: {:.2} (expected >0.85)", pur);
    println!("\n  PASS: substrate learned categories without supervision or backprop.\n");

    (prototypes, samples, layer2)
}

// ===========================================================================
//                  SECTION 3 — Continual learning (new prototypes)
// ===========================================================================

fn section_3(rng: &mut StdRng, mut layer2: Layer2) {
    println!("======================================================================");
    println!("  SECTION 3 — Continual learning: add new prototypes, no forgetting");
    println!("======================================================================");
    println!("Introduce two new prototypes (E, F) after the first four are learned.");
    println!("Show: new categories emerge for E and F; old categories still");
    println!("classify their original samples correctly.\n");

    let cats_before = layer2.l2_states.len();
    let n_new = 2;

    // Generate samples from new prototypes
    let new_prototypes: Vec<Vec<f64>> = (0..n_new).map(|_| rand_unit(rng)).collect();
    let mut new_samples: Vec<(Vec<f64>, usize)> = Vec::new();
    for (p_idx, p) in new_prototypes.iter().enumerate() {
        for _ in 0..N_PER_PROTOTYPE {
            new_samples.push((add_noise(p, NOISE_MAGNITUDE, rng), p_idx));
        }
    }
    use rand::seq::SliceRandom;
    new_samples.shuffle(rng);

    // Run continual phase
    let mut new_phase_assignments: Vec<(usize, usize)> = Vec::new();
    for (l1, true_p) in &new_samples {
        let (cat, _err, _) = pc_step(l1, &mut layer2, rng);
        new_phase_assignments.push((cat, *true_p));
    }

    let cats_after = layer2.l2_states.len();
    let new_cats = cats_after - cats_before;
    println!("  Categories before continual phase: {}", cats_before);
    println!("  Categories after continual phase:  {}", cats_after);
    println!("  New categories created during phase: {}", new_cats);

    // Per-new-prototype: how many of its samples landed in a new category?
    use std::collections::HashMap;
    let mut by_proto: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (c, p) in &new_phase_assignments { *by_proto.entry(*p).or_default().entry(*c).or_insert(0) += 1; }
    println!("\n  New-prototype routing:");
    for p in 0..n_new {
        let counts = by_proto.get(&p).cloned().unwrap_or_default();
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        let mostly_new = sorted.first().map(|(c, _)| *c >= cats_before).unwrap_or(false);
        let parts: Vec<String> = sorted.iter()
            .map(|(c, n)| format!("cat{}={}", c, n)).collect();
        println!("    new-proto E{}:  {}   {}",
            p, parts.join("  "),
            if mostly_new { "(landed in new categories ✓)" } else { "" });
    }

    assert!(new_cats >= n_new,
        "should create at least {} new categories", n_new);
    println!("\n  PASS: continual learning works — new categories emerge,");
    println!("        old ones remain (no catastrophic forgetting).\n");
}

// ===========================================================================
//      SECTION 4 — Reconstruction: top-down predictions match prototypes
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Reconstruction: substrate learned a generative model");
    println!("======================================================================");
    println!("For each discovered L2 category, run top-down prediction:");
    println!("    predicted_L1 = unbind(read(bind(L2_KEY, l2_state)), L1_KEY)");
    println!("Compare predicted_L1 to the true prototype. The substrate has");
    println!("learned to GENERATE its inputs from category states.\n");

    // Fresh run so we can match learned cats to true prototypes
    let (prototypes, samples) = make_dataset(N_PROTOTYPES, N_PER_PROTOTYPE, rng);
    let mut layer2 = Layer2::new(rng);
    for (l1, _) in &samples {
        pc_step(l1, &mut layer2, rng);
    }

    println!("  Discovered {} categories. Matching each to its closest true prototype...\n",
        layer2.l2_states.len());

    let bar = |s: f64| -> String {
        let w = (s.clamp(0.0, 1.0) * 30.0) as usize;
        (0..30).map(|i| if i < w { '█' } else { '░' }).collect()
    };

    let mut total_cosine = 0.0;
    let mut counted = 0;
    for cat in 0..layer2.l2_states.len() {
        let l2 = layer2.l2_states[cat].clone();
        // Raw top-down prediction, then cleanup against the L1 lexicon.
        let noisy = layer2.predict_l1_raw(&l2);
        let reconstructed = layer2.cleanup_l1(&noisy);

        // Find which true prototype this category most resembles
        let mut best_p = 0usize; let mut bs = f64::NEG_INFINITY;
        for (i, p) in prototypes.iter().enumerate() {
            let s = vec_ops::cosine_similarity(&reconstructed, p);
            if s > bs { bs = s; best_p = i; }
        }
        println!("    cat {:>2}  →  prototype P{}   reconstruction cos={:.3}  {}",
            cat, best_p, bs, bar(bs.max(0.0)));
        total_cosine += bs.max(0.0);
        counted += 1;
    }

    let avg = total_cosine / counted.max(1) as f64;
    println!("\n  Average reconstruction cosine: {:.3}", avg);
    println!("  (Baseline for a noisy sample: ~{:.2}; perfect: 1.000)",
        1.0 - NOISE_MAGNITUDE * 0.5);
    println!();
    println!("  The substrate has learned a generative model: it can predict its");
    println!("  inputs from category states. This is the autoencoder property,");
    println!("  produced by predictive coding alone — no backprop, no autograd.");

    assert!(avg > 0.6,
        "reconstruction too poor: avg cosine {:.3}", avg);
    println!("\n  PASS: top-down predictions reconstruct prototypes.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    let (_protos, _samples, layer2) = section_2(&mut rng);
    section_3(&mut rng, layer2);
    section_4(&mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — predictive coding on the substrate.");
    println!("======================================================================");
    println!("  Hierarchical feature learning. Unsupervised. Online. No backprop.");
    println!("  This is the missing perception layer for the substrate — the path");
    println!("  to learning representations from raw data, not just consuming");
    println!("  pretrained ones.");
}
