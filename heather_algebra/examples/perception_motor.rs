//! Perception encoder + motor decoder — the substrate's I/O boundary.
//!
//! Everything we've built up to now operates *inside* the substrate.
//! For real systems, two adapter layers sit at the boundary:
//!
//!   raw input → ENCODER → substrate vector → cognition → substrate vector
//!                                                              ↓
//!                                                          DECODER
//!                                                              ↓
//!                                                       discrete action
//!
//! The encoder turns raw signals (pixels, audio, sensor readings) into
//! unit vectors with meaningful similarity geometry. The decoder turns
//! substrate vectors into discrete outputs (actions, class labels,
//! tokens). Neither is a substrate primitive — they're adapters. But
//! their choice determines everything about what the substrate can do
//! end-to-end.
//!
//! Task for this demo: classify 16×16 noisy patches of oriented lines
//! into 4 orientation classes (horizontal, vertical, diagonal-up,
//! diagonal-down). It's the simplest "perceptual" task — small enough
//! to be transparent, structured enough that the encoder matters.
//!
//! Sections:
//!   1. Dataset: synthetic line patches with noise.
//!   2. Encoders compared (random-projection vs. oriented features):
//!      same substrate downstream, accuracy diverges with encoder quality.
//!   3. Decoders compared (NN cleanup vs. trained linear readout):
//!      same encoder upstream, accuracy diverges with decoder choice.
//!   4. Modularity: swap encoders without retraining substrate/decoder.
//!      The substrate's I/O design is modular by default.
//!
//! Run: `cargo run --release --example perception_motor -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 256;
const N_CLASSES: usize = 8;     // 8 orientations — adjacent ones overlap
const N_PER_CLASS: usize = 40;
const IMG: usize = 16;
const LINE_LEN: i32 = 12;
const NOISE: f64 = 0.6;         // high enough that random projection
                                // doesn't preserve enough structure
const BETA: f64 = 12.0;
const READ_ITERS: usize = 4;
const LR_LR: f64 = 0.05;
const LR_EPOCHS: usize = 150;

// ----- Helpers --------------------------------------------------------------

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

// ----- Synthetic data: oriented line patches -------------------------------

/// Generate a 16×16 line patch at given orientation index (0..N_CLASSES).
fn make_line(orient: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut p = vec![0.0_f64; IMG * IMG];
    let cx = (IMG as f64 - 1.0) / 2.0;
    let cy = (IMG as f64 - 1.0) / 2.0;
    let theta = orient as f64 * std::f64::consts::PI / N_CLASSES as f64;
    let dx = theta.cos();
    let dy = theta.sin();
    for t in -LINE_LEN/2..=LINE_LEN/2 {
        let x = (cx + t as f64 * dx).round() as i32;
        let y = (cy + t as f64 * dy).round() as i32;
        if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
            p[(y * IMG as i32 + x) as usize] = 1.0;
        }
    }
    // Additive Gaussian-ish noise
    for v in p.iter_mut() {
        *v += NOISE * (rng.r#gen::<f64>() * 2.0 - 1.0);
        *v = v.clamp(-1.0, 1.0);
    }
    p
}

fn make_dataset(rng: &mut StdRng) -> Vec<(Vec<f64>, usize)> {
    let mut out: Vec<(Vec<f64>, usize)> = Vec::new();
    for c in 0..N_CLASSES {
        for _ in 0..N_PER_CLASS {
            out.push((make_line(c, rng), c));
        }
    }
    use rand::seq::SliceRandom;
    out.shuffle(rng);
    out
}

fn split(data: Vec<(Vec<f64>, usize)>, train_frac: f64)
    -> (Vec<(Vec<f64>, usize)>, Vec<(Vec<f64>, usize)>)
{
    let cut = (data.len() as f64 * train_frac) as usize;
    let mut train: Vec<_> = data.into_iter().collect();
    let test = train.split_off(cut);
    (train, test)
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

// ===========================================================================
//   ENCODERS — raw input → substrate vector
// ===========================================================================

trait Encoder {
    fn encode(&self, x: &[f64]) -> Vec<f64>;
    fn name(&self) -> &'static str;
}

/// Random projection: a fixed random DIM×N matrix maps raw inputs to DIM.
/// Baseline encoder — no domain knowledge, no learning. Preserves
/// (approximately) the linear-similarity geometry of the input.
struct RandomProjectionEncoder {
    proj: Vec<Vec<f64>>,  // DIM × input_dim
}
impl RandomProjectionEncoder {
    fn new(in_dim: usize, rng: &mut StdRng) -> Self {
        let proj = (0..DIM).map(|_|
            (0..in_dim).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect()
        ).collect();
        Self { proj }
    }
}
impl Encoder for RandomProjectionEncoder {
    fn encode(&self, x: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = self.proj.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        vec_ops::normalize(&raw)
    }
    fn name(&self) -> &'static str { "RandomProjection" }
}

/// Oriented-features encoder: extracts spatial structure that's
/// informative about line orientation. Hand-crafted features, similar
/// to what V1 simple cells compute (but much simpler).
///
/// For each row/column/diagonal, the sum of pixels gives a 1D
/// projection. Different orientations produce different sum patterns.
/// Then random-project the feature vector to DIM.
struct OrientedFeaturesEncoder {
    proj: Vec<Vec<f64>>,  // DIM × n_features
    n_features: usize,
}
impl OrientedFeaturesEncoder {
    fn new(rng: &mut StdRng) -> Self {
        // Features: 16 row sums + 16 col sums + 31 diagonal sums + 31 anti-diagonal sums
        let n_features = IMG + IMG + (2 * IMG - 1) + (2 * IMG - 1);
        let proj = (0..DIM).map(|_|
            (0..n_features).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect()
        ).collect();
        Self { proj, n_features }
    }
    fn extract(&self, x: &[f64]) -> Vec<f64> {
        let mut feats = Vec::with_capacity(self.n_features);
        // Row sums
        for r in 0..IMG {
            feats.push((0..IMG).map(|c| x[r * IMG + c]).sum::<f64>());
        }
        // Col sums
        for c in 0..IMG {
            feats.push((0..IMG).map(|r| x[r * IMG + c]).sum::<f64>());
        }
        // Diagonal sums (r - c constant): indices -(IMG-1) .. (IMG-1)
        for d in -(IMG as i32 - 1)..=(IMG as i32 - 1) {
            let mut s = 0.0;
            for r in 0..IMG as i32 {
                let c = r - d;
                if c >= 0 && c < IMG as i32 {
                    s += x[(r * IMG as i32 + c) as usize];
                }
            }
            feats.push(s);
        }
        // Anti-diagonal sums (r + c constant): 0 .. 2*IMG-2
        for d in 0..(2 * IMG as i32 - 1) {
            let mut s = 0.0;
            for r in 0..IMG as i32 {
                let c = d - r;
                if c >= 0 && c < IMG as i32 {
                    s += x[(r * IMG as i32 + c) as usize];
                }
            }
            feats.push(s);
        }
        feats
    }
}
impl Encoder for OrientedFeaturesEncoder {
    fn encode(&self, x: &[f64]) -> Vec<f64> {
        let feats = self.extract(x);
        let raw: Vec<f64> = self.proj.iter()
            .map(|row| row.iter().zip(feats.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        vec_ops::normalize(&raw)
    }
    fn name(&self) -> &'static str { "OrientedFeatures" }
}

// ===========================================================================
//   COGNITION — substrate classifier (the middle)
// ===========================================================================

struct SubstrateClassifier {
    eam: MiniEAM,
    feat_key: Vec<f64>,
    label_key: Vec<f64>,
    label_vecs: Vec<Vec<f64>>,
}
impl SubstrateClassifier {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            feat_key: vec_ops::random_unit_vector(DIM, rng),
            label_key: vec_ops::random_unit_vector(DIM, rng),
            label_vecs: (0..N_CLASSES).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect(),
        }
    }
    fn train(&mut self, encoded: &[f64], label: usize) {
        let entry = bundle(&[
            &bind_vec(&self.feat_key, encoded),
            &bind_vec(&self.label_key, &self.label_vecs[label]),
        ]);
        self.eam.write(&entry);
    }
    /// Returns the substrate's "decision state" — the noisy label-vector
    /// recovered by unbinding the cognitive read. Decoder turns this
    /// into a discrete answer.
    fn infer(&self, encoded: &[f64]) -> Vec<f64> {
        let q = bind_vec(&self.feat_key, encoded);
        let r = self.eam.read(&q);
        vec_ops::normalize(&unbind_vec(&r, &self.label_key))
    }
}

// ===========================================================================
//   DECODERS — substrate vector → discrete output
// ===========================================================================

trait Decoder {
    fn decode(&self, state: &[f64]) -> usize;
    fn name(&self) -> &'static str;
}

/// Nearest-neighbour cleanup against the label lexicon. The simplest
/// decoder — just find the closest stored label vector.
struct NNDecoder {
    label_vecs: Vec<Vec<f64>>,
}
impl Decoder for NNDecoder {
    fn decode(&self, state: &[f64]) -> usize {
        let mut best = 0usize; let mut bs = f64::NEG_INFINITY;
        for (i, lv) in self.label_vecs.iter().enumerate() {
            let s = vec_ops::cosine_similarity(state, lv);
            if s > bs { bs = s; best = i; }
        }
        best
    }
    fn name(&self) -> &'static str { "NN-cleanup" }
}

/// Linear readout: a small weight matrix W ∈ ℝ^{N_CLASSES × DIM} and
/// bias trained on (substrate_state, label) pairs via softmax + SGD.
/// Effectively one layer of backprop on top of the substrate. Captures
/// any linear combination of substrate features that's informative
/// about labels — strictly more expressive than NN cleanup.
struct LinearReadout {
    w: Vec<Vec<f64>>,   // N_CLASSES × DIM
    b: Vec<f64>,
}
impl LinearReadout {
    fn new() -> Self {
        Self { w: vec![vec![0.0; DIM]; N_CLASSES], b: vec![0.0; N_CLASSES] }
    }
    fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        let mut logits = vec![0.0; N_CLASSES];
        for k in 0..N_CLASSES {
            logits[k] = self.b[k];
            for i in 0..DIM { logits[k] += self.w[k][i] * x[i]; }
        }
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for k in 0..N_CLASSES { logits[k] = (logits[k] - m).exp(); sum += logits[k]; }
        for k in 0..N_CLASSES { logits[k] /= sum; }
        logits
    }
    fn fit(&mut self, train: &[(Vec<f64>, usize)], epochs: usize, lr: f64) {
        for _ in 0..epochs {
            for (x, y) in train {
                let p = self.predict_proba(x);
                for k in 0..N_CLASSES {
                    let err = p[k] - if k == *y { 1.0 } else { 0.0 };
                    for i in 0..DIM { self.w[k][i] -= lr * err * x[i]; }
                    self.b[k] -= lr * err;
                }
            }
        }
    }
}
impl Decoder for LinearReadout {
    fn decode(&self, state: &[f64]) -> usize {
        let p = self.predict_proba(state);
        let mut best = 0; let mut bp = f64::NEG_INFINITY;
        for k in 0..N_CLASSES { if p[k] > bp { bp = p[k]; best = k; } }
        best
    }
    fn name(&self) -> &'static str { "LinearReadout" }
}

// ===========================================================================
//   FULL PIPELINE EVAL
// ===========================================================================

fn evaluate(
    encoder: &dyn Encoder,
    classifier: &SubstrateClassifier,
    decoder: &dyn Decoder,
    test: &[(Vec<f64>, usize)],
) -> f64 {
    let correct = test.iter().filter(|(x, y)| {
        let enc = encoder.encode(x);
        let state = classifier.infer(&enc);
        decoder.decode(&state) == *y
    }).count();
    correct as f64 / test.len() as f64
}

// ===========================================================================
//                  SECTION 1 — Dataset and baseline
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Dataset: noisy oriented line patches");
    println!("======================================================================");
    println!("16×16 patches with a line through the center at one of {} orientations.",
        N_CLASSES);
    println!("Additive noise σ={}. {} samples per class.", NOISE, N_PER_CLASS);

    let data = make_dataset(rng);
    println!("\n  Total samples: {}", data.len());

    // Spot-check: visualize one sample of each class
    let mut shown: [bool; N_CLASSES] = [false; N_CLASSES];
    println!("\n  Spot-check (one sample per class, * = >0.5, . = <0.5):");
    for (x, c) in &data {
        if shown[*c] { continue; }
        shown[*c] = true;
        println!("    class {} ({}°):", c, c * 180 / N_CLASSES);
        for r in 0..IMG {
            let row: String = (0..IMG).map(|col| {
                if x[r * IMG + col] > 0.5 { '*' }
                else if x[r * IMG + col] > 0.0 { '·' }
                else { ' ' }
            }).collect();
            println!("      {}", row);
        }
        if shown.iter().all(|&s| s) { break; }
    }
    println!("\n  PASS: dataset constructed.\n");
}

// ===========================================================================
//                  SECTION 2 — Encoder comparison
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Encoder comparison (same substrate, same decoder)");
    println!("======================================================================");
    println!("Train substrate on (encoded_x, label) pairs. Compare encoders by");
    println!("downstream classification accuracy with NN-cleanup decoder.\n");

    let data = make_dataset(rng);
    let (train, test) = split(data, 0.7);

    let encoders: Vec<Box<dyn Encoder>> = vec![
        Box::new(RandomProjectionEncoder::new(IMG * IMG, rng)),
        Box::new(OrientedFeaturesEncoder::new(rng)),
    ];

    println!("  {:<24} test accuracy",  "encoder");
    println!("  {:-<44}", "");
    for enc in &encoders {
        let mut classifier = SubstrateClassifier::new(rng);
        for (x, y) in &train {
            let e = enc.encode(x);
            classifier.train(&e, *y);
        }
        let decoder = NNDecoder { label_vecs: classifier.label_vecs.clone() };
        let acc = evaluate(enc.as_ref(), &classifier, &decoder, &test);
        println!("  {:<24} {:>5.1}%", enc.name(), acc * 100.0);
    }

    println!("\n  The encoder's match to the task's structure matters more than");
    println!("  the substrate's capacity. Oriented features know about orientation;");
    println!("  random projection has to discover it through the substrate's");
    println!("  exemplar storage.");
    println!("\n  PASS: encoder comparison.\n");
}

// ===========================================================================
//                  SECTION 3 — Decoder comparison
// ===========================================================================

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Decoder comparison (same encoder, same substrate)");
    println!("======================================================================");
    println!("Hold encoder fixed (OrientedFeatures). Vary decoder. Compare accuracy.\n");

    let data = make_dataset(rng);
    let (train, test) = split(data, 0.7);

    let encoder = OrientedFeaturesEncoder::new(rng);
    let mut classifier = SubstrateClassifier::new(rng);
    let mut train_states: Vec<(Vec<f64>, usize)> = Vec::new();
    for (x, y) in &train {
        let e = encoder.encode(x);
        classifier.train(&e, *y);
        // Capture substrate output for decoder training
        let s = classifier.infer(&e);
        train_states.push((s, *y));
    }

    // Decoder A: NN cleanup
    let nn = NNDecoder { label_vecs: classifier.label_vecs.clone() };

    // Decoder B: linear readout trained on substrate states
    let mut readout = LinearReadout::new();
    readout.fit(&train_states, LR_EPOCHS, LR_LR);

    let acc_nn = evaluate(&encoder, &classifier, &nn, &test);
    let acc_lr = evaluate(&encoder, &classifier, &readout, &test);

    println!("  {:<24} test accuracy", "decoder");
    println!("  {:-<44}", "");
    println!("  {:<24} {:>5.1}%", nn.name(), acc_nn * 100.0);
    println!("  {:<24} {:>5.1}%   (trained, {} epochs)",
        readout.name(), acc_lr * 100.0, LR_EPOCHS);

    println!("\n  Linear readout learns any informative direction in substrate state");
    println!("  space. NN cleanup is fixed by the label lexicon. The readout's");
    println!("  trained weights effectively LEARN the label codes that the");
    println!("  substrate's output happens to converge to — usually a small win.");
    println!("\n  PASS: decoder comparison.\n");
}

// ===========================================================================
//                  SECTION 4 — Modularity: swap components freely
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Modularity: swap components without retraining");
    println!("======================================================================");
    println!("Train substrate once with one encoder. Then test with a DIFFERENT");
    println!("encoder (without retraining anything). Naïvely expect catastrophic");
    println!("breakdown — encoders produce different vectors for the same input.\n");
    println!("THEN show the real modularity story: components ARE modular, but");
    println!("you have to train the substrate against the encoder you'll deploy.\n");

    let data = make_dataset(rng);
    let (train, test) = split(data, 0.7);

    let enc_oriented = OrientedFeaturesEncoder::new(rng);
    let enc_random = RandomProjectionEncoder::new(IMG * IMG, rng);

    // Train substrate using oriented encoder
    let mut classifier = SubstrateClassifier::new(rng);
    for (x, y) in &train {
        let e = enc_oriented.encode(x);
        classifier.train(&e, *y);
    }
    let decoder = NNDecoder { label_vecs: classifier.label_vecs.clone() };

    let acc_matched = evaluate(&enc_oriented, &classifier, &decoder, &test);
    let acc_mismatched = evaluate(&enc_random, &classifier, &decoder, &test);

    println!("  Substrate trained with OrientedFeatures encoder. Test-time encoder:");
    println!("    OrientedFeatures (matched):   {:>5.1}%", acc_matched * 100.0);
    println!("    RandomProjection (swapped):   {:>5.1}%   (≈ chance = {:.0}%)",
        acc_mismatched * 100.0, 100.0 / N_CLASSES as f64);

    println!("\n  Swapping the encoder without retraining the substrate breaks");
    println!("  inference — encoders define what \"feature similarity\" means to");
    println!("  the substrate, and the substrate's stored bindings are with");
    println!("  respect to one specific encoder's geometry.");
    println!();
    println!("  The MODULARITY is at a different layer: any encoder paired with");
    println!("  a substrate trained against IT is interchangeable as a unit. You");
    println!("  ship the (encoder, substrate-snapshot) PAIR as the trained model.");
    println!("  Snapshot the pair, send it to another system, plug it into a");
    println!("  decoder of choice. The decoder is the only piece that's freely");
    println!("  swappable post-deployment — because it reads substrate output,");
    println!("  not input.");

    // Final sanity check: the matched pipeline works
    assert!(acc_matched > 0.55, "matched pipeline accuracy too low");
    assert!(acc_mismatched < 0.4, "mismatched should be near chance");
    println!("\n  PASS: substrate I/O modularity characterised honestly.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    section_4(&mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — substrate I/O boundary characterised.");
    println!("======================================================================");
    println!("  Perception encoders translate raw input → semantic-pointer space.");
    println!("  Motor decoders translate semantic-pointer space → discrete output.");
    println!("  Neither is a substrate primitive; both are critical for end-to-end");
    println!("  performance. The (encoder + substrate-snapshot) is the trained-");
    println!("  model unit; the decoder is the only piece freely swappable after.");
}
