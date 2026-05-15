//! Engine integration — substrate classifier on `heather_db::Collection`.
//!
//! Every cognitive demo so far has used `MiniEAM`, a ~50-line in-memory
//! shim with the same softmax-read shape as the production engine. The
//! substrate proof was therefore isolated from the storage proof. This
//! demo closes the gap: build the SAME substrate classifier twice —
//! once on `MiniEAM`, once on `heather_db::Collection` (LMDB-backed,
//! navigable-graph activation, conscience mechanism, all of it) — train
//! both on identical synthetic data, compare accuracy and latency
//! head-to-head.
//!
//! What this validates:
//!   - The Collection's write/read API does what MiniEAM's does, at the
//!     algebraic level our cognitive demos rely on.
//!   - bind/unbind from `heather_algebra` compose cleanly with the
//!     production engine — no API mismatch, no shape conversions.
//!   - Real-engine accuracy is at parity with the in-memory shim (or
//!     better — the engine has graph search and consolidation that
//!     MiniEAM doesn't).
//!
//! Task: 4-class synthetic Gaussian clustering in 4-D feature space,
//! random-projected to a 256-dim substrate vector. 120 samples total,
//! 80/20 split.
//!
//! Run: `cargo run --release --example engine_integration -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{Collection, EAMConfig, ReadStrategy, vec_ops, HardLocation, LocationId};
use heather_db::store::Store;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

// ----- Hyperparameters ------------------------------------------------------

const DIM: usize = 256;
const N_CLASSES: usize = 4;
const N_FEATURES: usize = 4;
const N_PER_CLASS: usize = 30;
const NOISE: f64 = 0.4;

// ----- Helpers --------------------------------------------------------------

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

fn add_noise(v: &[f64], noise: f64, rng: &mut StdRng) -> Vec<f64> {
    v.iter().map(|x| x + noise * (rng.r#gen::<f64>() * 2.0 - 1.0)).collect()
}

fn nearest_label(state: &[f64], label_vecs: &[Vec<f64>]) -> usize {
    let state = vec_ops::normalize(state);
    let mut best = 0; let mut bs = f64::NEG_INFINITY;
    for (i, lv) in label_vecs.iter().enumerate() {
        let s = vec_ops::cosine_similarity(&state, lv);
        if s > bs { bs = s; best = i; }
    }
    best
}

// ----- Synthetic Gaussian-clusters dataset ---------------------------------

fn make_dataset(rng: &mut StdRng) -> Vec<(Vec<f64>, usize)> {
    // 4 centroids in 4-D feature space (slightly separated)
    let centroids: Vec<Vec<f64>> = (0..N_CLASSES).map(|c| {
        let mut v = vec![0.0; N_FEATURES];
        v[c % N_FEATURES] = 1.0;
        if c >= N_FEATURES { v[(c + 1) % N_FEATURES] = -1.0; }
        v
    }).collect();
    let mut data: Vec<(Vec<f64>, usize)> = Vec::new();
    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..N_PER_CLASS {
            data.push((add_noise(centroid, NOISE, rng), c));
        }
    }
    use rand::seq::SliceRandom;
    data.shuffle(rng);
    data
}

// ----- Random projection encoder (shared) ----------------------------------

struct Encoder { proj: Vec<Vec<f64>> }
impl Encoder {
    fn new(rng: &mut StdRng) -> Self {
        let proj = (0..DIM).map(|_|
            (0..N_FEATURES).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect()
        ).collect();
        Self { proj }
    }
    fn encode(&self, x: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = self.proj.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        vec_ops::normalize(&raw)
    }
}

// ===========================================================================
//   MiniEAM-backed classifier — the in-memory shim used in earlier demos
// ===========================================================================

const BETA: f64 = 12.0;
const READ_ITERS: usize = 4;

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

struct MiniClassifier {
    eam: MiniEAM,
    feat_key: Vec<f64>,
    label_key: Vec<f64>,
    label_vecs: Vec<Vec<f64>>,
    encoder: Encoder,
}
impl MiniClassifier {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            feat_key: vec_ops::random_unit_vector(DIM, rng),
            label_key: vec_ops::random_unit_vector(DIM, rng),
            label_vecs: (0..N_CLASSES).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect(),
            encoder: Encoder::new(rng),
        }
    }
    fn train(&mut self, x: &[f64], label: usize) {
        let feat = self.encoder.encode(x);
        let entry = bundle(&[
            &bind_vec(&self.feat_key, &feat),
            &bind_vec(&self.label_key, &self.label_vecs[label]),
        ]);
        self.eam.write(&entry);
    }
    fn predict(&self, x: &[f64]) -> usize {
        let feat = self.encoder.encode(x);
        let q = bind_vec(&self.feat_key, &feat);
        let recalled = self.eam.read(&q);
        let label_noisy = unbind_vec(&recalled, &self.label_key);
        nearest_label(&label_noisy, &self.label_vecs)
    }
}

// ===========================================================================
//   Collection-backed classifier — the real heather_db engine
// ===========================================================================

struct EngineClassifier {
    // The Collection holds an Arc<Store>. We need to keep the TempDir
    // alive so the LMDB env stays valid for the lifetime of this struct.
    collection: Arc<Collection>,
    _temp: TempDir,
    feat_key: Vec<f64>,
    label_key: Vec<f64>,
    label_vecs: Vec<Vec<f64>>,
    encoder: Encoder,
}
impl EngineClassifier {
    fn new(rng: &mut StdRng) -> Self {
        let temp = TempDir::new().unwrap();
        let store = Arc::new(Store::open(temp.path(), 64).unwrap());
        let mut txn = store.write_txn().unwrap();
        let id = store.create_collection(&mut txn, "engine_demo").unwrap();
        txn.commit().unwrap();
        let mut config = EAMConfig::new(DIM).unwrap();
        config.l_0 = 50;
        config.k = 10;
        let collection = Arc::new(
            Collection::new(id, "engine_demo".into(), store, &config).unwrap()
        );
        Self {
            collection,
            _temp: temp,
            feat_key: vec_ops::random_unit_vector(DIM, rng),
            label_key: vec_ops::random_unit_vector(DIM, rng),
            label_vecs: (0..N_CLASSES).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect(),
            encoder: Encoder::new(rng),
        }
    }
    fn train(&self, x: &[f64], label: usize) {
        let feat = self.encoder.encode(x);
        let entry = bundle(&[
            &bind_vec(&self.feat_key, &feat),
            &bind_vec(&self.label_key, &self.label_vecs[label]),
        ]);
        self.collection.write(&entry).unwrap();
    }
    fn predict(&self, x: &[f64]) -> usize {
        let feat = self.encoder.encode(x);
        let q = bind_vec(&self.feat_key, &feat);
        let recalled = self.collection.read(&q, ReadStrategy::HopfieldIter).unwrap();
        let label_noisy = unbind_vec(&recalled, &self.label_key);
        nearest_label(&label_noisy, &self.label_vecs)
    }
}

// ===========================================================================
//   Evaluation
// ===========================================================================

fn evaluate_mini(c: &MiniClassifier, test: &[(Vec<f64>, usize)]) -> f64 {
    let correct = test.iter().filter(|(x, y)| c.predict(x) == *y).count();
    correct as f64 / test.len() as f64
}
fn evaluate_engine(c: &EngineClassifier, test: &[(Vec<f64>, usize)]) -> f64 {
    let correct = test.iter().filter(|(x, y)| c.predict(x) == *y).count();
    correct as f64 / test.len() as f64
}

// ===========================================================================
//   Main: side-by-side comparison
// ===========================================================================

fn main() {
    println!("======================================================================");
    println!("  Engine integration — substrate on MiniEAM vs heather_db::Collection");
    println!("======================================================================");

    let mut rng = StdRng::seed_from_u64(42);
    let data = make_dataset(&mut rng);
    let cut = (data.len() as f64 * 0.8) as usize;
    let (train, test): (Vec<_>, Vec<_>) = data.into_iter().enumerate()
        .partition(|(i, _)| *i < cut);
    let train: Vec<_> = train.into_iter().map(|(_, x)| x).collect();
    let test:  Vec<_> = test.into_iter().map(|(_, x)| x).collect();
    println!("\nTask: {} classes, {} samples/class, noise σ={}", N_CLASSES, N_PER_CLASS, NOISE);
    println!("Train: {}   Test: {}\n", train.len(), test.len());

    // ----- MiniEAM run --------------------------------------------------
    println!("--- MiniEAM (in-memory, our prior demos) ---");
    let mut mini = MiniClassifier::new(&mut rng);
    let t_train_mini = Instant::now();
    for (x, y) in &train { mini.train(x, *y); }
    let t_train_mini = t_train_mini.elapsed();
    let t_test_mini = Instant::now();
    let acc_mini = evaluate_mini(&mini, &test);
    let t_test_mini = t_test_mini.elapsed();
    println!("  train time: {:>8.2} ms   ({} writes)",
        t_train_mini.as_secs_f64() * 1000.0, train.len());
    println!("  test time:  {:>8.2} ms   ({} predictions)",
        t_test_mini.as_secs_f64() * 1000.0, test.len());
    println!("  accuracy:   {:>5.1}%", acc_mini * 100.0);

    // ----- Engine run ---------------------------------------------------
    println!("\n--- heather_db::Collection (LMDB-backed, graph-search-capable) ---");
    let engine = EngineClassifier::new(&mut rng);
    let t_train_engine = Instant::now();
    for (x, y) in &train { engine.train(x, *y); }
    let t_train_engine = t_train_engine.elapsed();
    let t_test_engine = Instant::now();
    let acc_engine = evaluate_engine(&engine, &test);
    let t_test_engine = t_test_engine.elapsed();
    println!("  train time: {:>8.2} ms   ({} writes)",
        t_train_engine.as_secs_f64() * 1000.0, train.len());
    println!("  test time:  {:>8.2} ms   ({} predictions)",
        t_test_engine.as_secs_f64() * 1000.0, test.len());
    println!("  accuracy:   {:>5.1}%", acc_engine * 100.0);

    // ----- Comparison ---------------------------------------------------
    println!("\n--- Comparison ---");
    println!("  Accuracy:   MiniEAM {:.1}%   Engine {:.1}%   (Δ = {:+.1}pt)",
        acc_mini * 100.0, acc_engine * 100.0, (acc_engine - acc_mini) * 100.0);
    let train_ratio = t_train_engine.as_secs_f64() / t_train_mini.as_secs_f64();
    let test_ratio = t_test_engine.as_secs_f64() / t_test_mini.as_secs_f64();
    println!("  Train:      Engine {:>4.1}× MiniEAM   ({:.2}ms vs {:.2}ms)",
        train_ratio, t_train_engine.as_secs_f64() * 1000.0, t_train_mini.as_secs_f64() * 1000.0);
    println!("  Test:       Engine {:>4.1}× MiniEAM   ({:.2}ms vs {:.2}ms)",
        test_ratio, t_test_engine.as_secs_f64() * 1000.0, t_test_mini.as_secs_f64() * 1000.0);
    println!();
    println!("  Engine adds: LMDB persistence, navigable-graph activation,");
    println!("  conscience mechanism, novelty-split, adaptive capacity. These");
    println!("  are real costs but they buy production-grade properties (durable,");
    println!("  concurrent-safe, scales beyond memory).");
    println!();
    println!("  The substrate primitives — bind, unbind, EAM read — compose with");
    println!("  the production engine without any API mismatch. Every cognitive");
    println!("  demo's algebra carries over unchanged; just swap MiniEAM for");
    println!("  Collection.");

    // ----- Assertions ---------------------------------------------------
    let chance = 1.0 / N_CLASSES as f64;
    assert!(acc_mini > chance * 1.5,
        "MiniEAM baseline failed: {:.1}%", acc_mini * 100.0);
    assert!(acc_engine > chance * 1.5,
        "Engine baseline failed: {:.1}%", acc_engine * 100.0);
    assert!((acc_mini - acc_engine).abs() < 0.20,
        "Engine and MiniEAM should be within 20pt: {:.1} vs {:.1}",
        acc_mini * 100.0, acc_engine * 100.0);

    println!("\n======================================================================");
    println!("  PASS — substrate works through the production engine.");
    println!("======================================================================");
}
