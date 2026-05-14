//! Working memory à la Eliasmith's Spaun, ported to HeatherDB primitives.
//!
//! Spaun (Eliasmith et al., 2012) used HRR-style "semantic pointers" to
//! implement a working-memory buffer that could hold a list of items,
//! recall them by position, manipulate them (reverse, count, etc.), and
//! drive downstream motor output — all on a single distributed substrate.
//! Spaun fought a spiking-neuron substrate (Nengo) to make this work.
//! On HeatherDB the operators are native; this is what the same idea
//! looks like when the substrate is the one Eliasmith was building toward.
//!
//! Structure:
//!   1. Capacity scaling — how list length affects recall accuracy.
//!   2. Recency via decay — older items fade, newer items dominate.
//!   3. Buffer reversal as a value transformation — the entire memory
//!      contents reversed by vector algebra, then probed in original
//!      position order.
//!   4. Spaun-style serial recall task: load a digit string, read it
//!      back forward and backward.
//!
//! Working memory in this construction:
//!
//!     buffer = Σ_i  decay^(N-1-i) · bind(POS_i, item_i)
//!
//! Push appends at the next position (decaying existing contents).
//! Recall(i) is `unbind(normalize(buffer), POS_i)` cleaned up against
//! a content lexicon. Reverse is an algebraic re-binding of the stored
//! items to inverted position keys.
//!
//! Run: `cargo run --release --example working_memory -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::vec_ops;
use rand::{SeedableRng, rngs::StdRng};

const DIM: usize = 512;
const LEXICON_SIZE: usize = 10; // 10 "digits" — generic random vectors
const MAX_LEN: usize = 25;

// ----- Working memory primitive --------------------------------------------

struct WorkingMemory {
    pos_keys: Vec<Vec<f64>>,
    buffer: Vec<f64>,
    n: usize,
    decay: f64,
}

impl WorkingMemory {
    fn new(pos_keys: Vec<Vec<f64>>, decay: f64) -> Self {
        let d = pos_keys[0].len();
        Self { pos_keys, buffer: vec![0.0; d], n: 0, decay }
    }

    /// Append an item at the next position. Existing contents decay
    /// by `self.decay` first — set to 1.0 for no decay, <1.0 for
    /// recency effects.
    fn push(&mut self, item: &[f64]) {
        if self.n >= self.pos_keys.len() {
            panic!("buffer overflow (no position keys left)");
        }
        for x in &mut self.buffer { *x *= self.decay; }
        let bound = bind_vec(&self.pos_keys[self.n], item);
        for i in 0..self.buffer.len() { self.buffer[i] += bound[i]; }
        self.n += 1;
    }

    /// Recall the item at position `pos`. Returns (best_index, similarity)
    /// where best_index is the closest entry in the supplied lexicon.
    fn recall(&self, pos: usize, lexicon: &[Vec<f64>]) -> (usize, f64) {
        let buf = vec_ops::normalize(&self.buffer);
        let noisy = unbind_vec(&buf, &self.pos_keys[pos]);
        let mut best = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, v) in lexicon.iter().enumerate() {
            let s = vec_ops::cosine_similarity(&noisy, v);
            if s > best_s { best_s = s; best = i; }
        }
        (best, best_s)
    }

    /// Reverse the buffer algebraically: extract each stored item by
    /// unbinding, then re-bind at the *complementary* position key
    /// (item at pos i goes to pos N-1-i). The substrate operations
    /// alone produce the reversed buffer; no external sequence handling.
    fn reverse(&self, lexicon: &[Vec<f64>]) -> WorkingMemory {
        let mut new_buf = vec![0.0; self.buffer.len()];
        let buf = vec_ops::normalize(&self.buffer);
        for i in 0..self.n {
            // Clean up the item at position i via lexicon (the EAM
            // cleanup step — turns noisy unbind into a clean item).
            let noisy = unbind_vec(&buf, &self.pos_keys[i]);
            let (best, _) = nearest(&noisy, lexicon);
            let clean_item = &lexicon[best];
            // Re-bind at the inverted position.
            let new_pos = self.n - 1 - i;
            let bound = bind_vec(&self.pos_keys[new_pos], clean_item);
            for k in 0..new_buf.len() { new_buf[k] += bound[k]; }
        }
        WorkingMemory {
            pos_keys: self.pos_keys.clone(),
            buffer: new_buf,
            n: self.n,
            decay: self.decay,
        }
    }
}

fn nearest(v: &[f64], lexicon: &[Vec<f64>]) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for (i, lv) in lexicon.iter().enumerate() {
        let s = vec_ops::cosine_similarity(v, lv);
        if s > best_s { best_s = s; best = i; }
    }
    (best, best_s)
}

// ----- Setup helpers --------------------------------------------------------

fn make_lexicon(n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..n).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect()
}

fn make_pos_keys(rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..MAX_LEN).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect()
}

fn load(wm: &mut WorkingMemory, list: &[usize], lexicon: &[Vec<f64>]) {
    for &d in list { wm.push(&lexicon[d]); }
}

fn recall_all(wm: &WorkingMemory, lexicon: &[Vec<f64>]) -> Vec<usize> {
    (0..wm.n).map(|i| wm.recall(i, lexicon).0).collect()
}

fn accuracy(expected: &[usize], got: &[usize]) -> f64 {
    let hits = expected.iter().zip(got.iter()).filter(|(a, b)| a == b).count();
    hits as f64 / expected.len() as f64
}

fn print_recall_table(expected: &[usize], got: &[usize]) {
    let mark: String = expected.iter().zip(got.iter())
        .map(|(a, b)| if a == b { '✓' } else { '✗' })
        .map(|c| format!(" {}", c)).collect();
    let expect_s: String = expected.iter().map(|d| format!(" {}", d)).collect();
    let got_s: String    = got.iter().map(|d| format!(" {}", d)).collect();
    println!("    expected: {}", expect_s);
    println!("    recalled: {}", got_s);
    println!("              {}    {}/{} = {:.0}%",
        mark, expected.iter().zip(got.iter()).filter(|(a,b)| a==b).count(),
        expected.len(),
        accuracy(expected, got) * 100.0);
}

// ----- Section 1: capacity scaling -----------------------------------------

fn section_capacity(lexicon: &[Vec<f64>], pos_keys: &[Vec<f64>]) -> Vec<(usize, f64)> {
    println!("============================================================");
    println!("  SECTION 1 — Capacity scaling (no decay, dim={})", DIM);
    println!("============================================================");
    println!("Lists of increasing length. Substrate noise scales as √(N-1)/√d,");
    println!("so capacity grows with dim. Each list is random digits 0-9.\n");

    let mut rng = StdRng::seed_from_u64(101);
    let lengths = [4, 7, 10, 13, 16, 20];
    let mut results = Vec::new();

    for &n in &lengths {
        use rand::Rng;
        let list: Vec<usize> = (0..n).map(|_| rng.r#gen_range(0..LEXICON_SIZE)).collect();
        let mut wm = WorkingMemory::new(pos_keys.to_vec(), 1.0);
        load(&mut wm, &list, lexicon);
        let got = recall_all(&wm, lexicon);
        let acc = accuracy(&list, &got);
        println!("  N = {:2}:", n);
        print_recall_table(&list, &got);
        println!();
        results.push((n, acc));
    }
    results
}

// ----- Section 2: recency via decay ----------------------------------------

fn section_recency(lexicon: &[Vec<f64>], pos_keys: &[Vec<f64>]) {
    println!("============================================================");
    println!("  SECTION 2 — Recency effect via decay");
    println!("============================================================");
    println!("Each push decays the existing buffer. Older items fade, newer");
    println!("items dominate. Accuracy by position should slope upward.\n");

    let mut rng = StdRng::seed_from_u64(202);
    let n = 12;
    let decay = 0.85;
    let trials = 30;

    let mut by_position = vec![0usize; n];
    for _ in 0..trials {
        use rand::Rng;
        let list: Vec<usize> = (0..n).map(|_| rng.r#gen_range(0..LEXICON_SIZE)).collect();
        let mut wm = WorkingMemory::new(pos_keys.to_vec(), decay);
        load(&mut wm, &list, lexicon);
        let got = recall_all(&wm, lexicon);
        for i in 0..n {
            if got[i] == list[i] { by_position[i] += 1; }
        }
    }

    println!("  Per-position accuracy over {} trials, decay = {}:", trials, decay);
    println!("    position:  {}", (0..n).map(|i| format!("{:>3}", i)).collect::<String>());
    let acc_strs: String = by_position.iter()
        .map(|c| format!("{:>2}%", (*c as f64 / trials as f64 * 100.0) as i32))
        .map(|s| format!(" {}", s)).collect();
    println!("    accuracy: {}", acc_strs);

    // Bar chart
    println!();
    for i in 0..n {
        let pct = by_position[i] as f64 / trials as f64;
        let bar: String = (0..((pct * 30.0) as usize)).map(|_| '█').collect();
        println!("    pos {:2}: {:3.0}%  {}", i, pct * 100.0, bar);
    }
    println!();

    // Sanity check: later positions should be better than early ones
    let early_avg: f64 = by_position[..3].iter().sum::<usize>() as f64 / 3.0 / trials as f64;
    let late_avg: f64 = by_position[n-3..].iter().sum::<usize>() as f64 / 3.0 / trials as f64;
    println!("    Early-3 avg: {:.0}%   Late-3 avg: {:.0}%   (recency gap: {:+.0}%)",
        early_avg * 100.0, late_avg * 100.0, (late_avg - early_avg) * 100.0);
    assert!(late_avg > early_avg + 0.10,
        "decay should produce recency: early={:.2}, late={:.2}", early_avg, late_avg);
    println!("\n  PASS: recency effect emerges from substrate dynamics.\n");
}

// ----- Section 3: buffer reversal via algebra ------------------------------

fn section_reversal(lexicon: &[Vec<f64>], pos_keys: &[Vec<f64>]) {
    println!("============================================================");
    println!("  SECTION 3 — Buffer reversal as a value transformation");
    println!("============================================================");
    println!("The entire working-memory buffer is *one vector*. Reversing it");
    println!("is a vector-algebra operation (unbind each pos, rebind at the");
    println!("complementary pos). After reversal, probing position i returns");
    println!("what was at position N-1-i in the original.\n");

    let mut rng = StdRng::seed_from_u64(303);
    use rand::Rng;
    let n = 8;
    let list: Vec<usize> = (0..n).map(|_| rng.r#gen_range(0..LEXICON_SIZE)).collect();

    let mut wm = WorkingMemory::new(pos_keys.to_vec(), 1.0);
    load(&mut wm, &list, lexicon);
    let forward = recall_all(&wm, lexicon);
    let reversed_wm = wm.reverse(lexicon);
    let backward = recall_all(&reversed_wm, lexicon);
    let expected_backward: Vec<usize> = list.iter().rev().copied().collect();

    println!("  Original list:    {:?}", list);
    println!("  Forward recall:   {:?}", forward);
    println!("  After substrate-level reversal:");
    println!("    expected:       {:?}", expected_backward);
    println!("    got (probed 0..N): {:?}", backward);
    let acc_fwd = accuracy(&list, &forward);
    let acc_rev = accuracy(&expected_backward, &backward);
    println!("    forward accuracy:  {:.0}%", acc_fwd * 100.0);
    println!("    reversed accuracy: {:.0}%", acc_rev * 100.0);

    assert!(acc_fwd > 0.9, "forward recall should be reliable");
    assert!(acc_rev > 0.9, "reversal should preserve accuracy");
    println!("\n  PASS: buffer transformation in pure vector algebra.\n");
}

// ----- Section 4: Spaun-style serial recall task ---------------------------

fn section_spaun(lexicon: &[Vec<f64>], pos_keys: &[Vec<f64>]) {
    println!("============================================================");
    println!("  SECTION 4 — Spaun-style task: hear digits, recall forward+back");
    println!("============================================================");
    println!("The canonical Spaun working-memory benchmark: hear a list of");
    println!("digits one at a time, then output them forward and backward.");
    println!("This is one of the eight tasks Spaun performed end-to-end in");
    println!("Eliasmith et al. 2012. We do it as a substrate primitive.\n");

    // The first 7 digits of pi after the decimal point.
    let list: Vec<usize> = vec![3, 1, 4, 1, 5, 9, 2];
    println!("  Heard:           {:?}", list);

    let mut wm = WorkingMemory::new(pos_keys.to_vec(), 1.0);
    load(&mut wm, &list, lexicon);

    let forward = recall_all(&wm, lexicon);
    let reversed = wm.reverse(lexicon);
    let backward = recall_all(&reversed, lexicon);

    println!("  Recall forward:  {:?}     {}",
        forward, if forward == list { "✓" } else { "✗" });
    println!("  Recall backward: {:?}     {}",
        backward,
        if backward.iter().eq(list.iter().rev()) { "✓" } else { "✗" });

    assert_eq!(forward, list, "forward recall should be exact");
    assert!(backward.iter().eq(list.iter().rev()), "backward recall should be exact");
    println!("\n  PASS: Spaun's working-memory task on the substrate.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let lexicon = make_lexicon(LEXICON_SIZE, &mut rng);
    let pos_keys = make_pos_keys(&mut rng);

    let capacity_results = section_capacity(&lexicon, &pos_keys);
    section_recency(&lexicon, &pos_keys);
    section_reversal(&lexicon, &pos_keys);
    section_spaun(&lexicon, &pos_keys);

    println!("============================================================");
    println!("  SUMMARY");
    println!("============================================================");
    println!("Capacity scaling (dim={}):", DIM);
    for (n, acc) in &capacity_results {
        let bar: String = (0..((acc * 30.0) as usize)).map(|_| '█').collect();
        println!("  N = {:2}   {:>3.0}%   {}", n, acc * 100.0, bar);
    }
    println!();
    println!("Capacity scales as √d / √(noise floor) — bigger dim, more items.");
    println!("All four sections passed. Eliasmith's working-memory primitive");
    println!("runs natively on the substrate, no spiking-neuron emulation");
    println!("required.");
}
