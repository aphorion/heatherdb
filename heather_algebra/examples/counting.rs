//! Counting on the substrate — Spaun's iconic discrete arithmetic on
//! continuous distributed vectors.
//!
//! The challenge: HRR vectors are continuous and content-addressable.
//! Counting feels discrete. How does the substrate iterate "0 → 1 → 2 → 3 → ..."
//! without losing the integer identity to noise?
//!
//! The answer is the HRR + EAM synthesis: numbers are random unit vectors;
//! the **successor** is a stored set of bound `(CURRENT, NEXT)` pairs in
//! the EAM; iteration is repeated read-and-unbind; the substrate's softmax
//! read cleans up the noisy intermediate states. Counting falls out.
//!
//! Sections:
//!   1. Successor operator works: 10/10 successions correct.
//!   2. Iteration with vs. without explicit cleanup — measures the
//!      similarity decay and shows where each strategy breaks down.
//!   3. Spaun's count-from-N task: input (start=3, count=4), output the
//!      sequence 3,4,5,6,7 via the working-memory + counting combo.
//!   4. Addition as iterated counting: a + b computed as "succ b times
//!      from a." No arithmetic logic unit anywhere.
//!
//! Run: `cargo run --release --example counting -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{SeedableRng, rngs::StdRng};

const DIM: usize = 1024;
const N_DIGITS: usize = 10;
const BETA: f64 = 15.0;
const READ_ITERS: usize = 4;
const WM_HORIZON: usize = 12;

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

// ----- The Successor primitive ---------------------------------------------

struct Successor {
    eam: MiniEAM,
    current_key: Vec<f64>,
    next_key: Vec<f64>,
    numbers: Vec<Vec<f64>>,
}

impl Successor {
    fn new(rng: &mut StdRng) -> Self {
        let current_key = rand_unit(rng);
        let next_key = rand_unit(rng);
        let numbers: Vec<Vec<f64>> =
            (0..N_DIGITS).map(|_| rand_unit(rng)).collect();
        let mut eam = MiniEAM::new();
        // Cyclic successor: 0→1, 1→2, ..., 9→0. Lets us iterate indefinitely.
        for n in 0..N_DIGITS {
            let next_n = (n + 1) % N_DIGITS;
            let pair = bundle(&[
                &bind_vec(&current_key, &numbers[n]),
                &bind_vec(&next_key, &numbers[next_n]),
            ]);
            eam.write(&pair);
        }
        Self { eam, current_key, next_key, numbers }
    }

    /// Raw successor: bind input to CURRENT, read the EAM, unbind NEXT.
    /// Returns the noisy successor vector — no cleanup.
    fn raw_succ(&self, x: &[f64]) -> Vec<f64> {
        let query = bind_vec(&self.current_key, x);
        let recalled = self.eam.read(&query);
        unbind_vec(&recalled, &self.next_key)
    }

    /// Successor with cleanup against the number lexicon.
    /// Returns (clean_next_vec, integer_index, similarity).
    fn clean_succ(&self, x: &[f64]) -> (Vec<f64>, usize, f64) {
        let noisy = self.raw_succ(x);
        let (idx, sim) = self.decode(&noisy);
        (self.numbers[idx].clone(), idx, sim)
    }

    /// Decode a vector to its closest digit + similarity.
    fn decode(&self, v: &[f64]) -> (usize, f64) {
        let v = vec_ops::normalize(v);
        let mut best = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, n) in self.numbers.iter().enumerate() {
            let s = vec_ops::cosine_similarity(&v, n);
            if s > best_s { best_s = s; best = i; }
        }
        (best, best_s)
    }
}

// ===========================================================================
//                    SECTION 1 — Successor operator
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Successor operator (substrate as state machine)");
    println!("======================================================================");
    println!("10 bound (CURRENT, NEXT) pairs stored. succ(n) = read-then-unbind.");
    println!("Successor is cyclic: succ(9) = 0.\n");

    let succ = Successor::new(rng);
    let mut hits = 0;
    for n in 0..N_DIGITS {
        let (_, predicted, sim) = succ.clean_succ(&succ.numbers[n]);
        let expected = (n + 1) % N_DIGITS;
        let mark = if predicted == expected { hits += 1; "✓" } else { "✗" };
        println!("  succ({}) → {} (sim {:.3}) expected {} {}",
            n, predicted, sim, expected, mark);
    }
    println!("\n  {}/{} successions correct.", hits, N_DIGITS);
    assert_eq!(hits, N_DIGITS, "successor operator failed");
    println!("  PASS: substrate computes a discrete transition function.\n");
}

// ===========================================================================
//      SECTION 2 — Iteration with vs. without explicit cleanup
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Iteration: cleanup vs no-cleanup");
    println!("======================================================================");
    println!("Apply succ 30 times starting from 0. Cyclic so the truth is k mod 10.");
    println!("With cleanup: snap to nearest digit each step. Without: feed raw noise.");
    println!("Watch the similarity bars — that's the substrate's grip on the answer.\n");

    let succ = Successor::new(rng);
    let n_iters = 30;

    let bar = |sim: f64| -> String {
        let width = (sim.clamp(0.0, 1.0) * 30.0) as usize;
        (0..30).map(|i| if i < width { '█' } else { '░' }).collect()
    };

    // --- With cleanup ---------------------------------------------------
    println!("  WITH cleanup (snap to nearest digit after each step):");
    let mut x = succ.numbers[0].clone();
    let mut clean_hits = 0;
    let mut clean_sims = Vec::new();
    for k in 1..=n_iters {
        let (clean_next, idx, sim) = succ.clean_succ(&x);
        let expected = k % N_DIGITS;
        let ok = idx == expected;
        if ok { clean_hits += 1; }
        clean_sims.push(sim);
        if k <= 10 || k % 5 == 0 {
            println!("    k={:2}: {}  sim={:.3}  decoded={} (expected {}) {}",
                k, bar(sim), sim, idx, expected, if ok {"✓"} else {"✗"});
        }
        x = clean_next;
    }
    let clean_avg: f64 = clean_sims.iter().sum::<f64>() / clean_sims.len() as f64;
    println!("    {}/{} correct, avg sim = {:.3}\n", clean_hits, n_iters, clean_avg);

    // --- Without cleanup ------------------------------------------------
    println!("  WITHOUT cleanup (feed raw unbind output back in):");
    let mut x = succ.numbers[0].clone();
    let mut raw_hits = 0;
    let mut raw_sims = Vec::new();
    let mut first_fail: Option<usize> = None;
    for k in 1..=n_iters {
        let noisy = succ.raw_succ(&x);
        let (idx, sim) = succ.decode(&noisy);
        let expected = k % N_DIGITS;
        let ok = idx == expected;
        if ok { raw_hits += 1; } else if first_fail.is_none() { first_fail = Some(k); }
        raw_sims.push(sim);
        if k <= 10 || k % 5 == 0 {
            println!("    k={:2}: {}  sim={:.3}  decoded={} (expected {}) {}",
                k, bar(sim), sim, idx, expected, if ok {"✓"} else {"✗"});
        }
        x = noisy;
    }
    let raw_avg: f64 = raw_sims.iter().sum::<f64>() / raw_sims.len() as f64;
    println!("    {}/{} correct, avg sim = {:.3}",  raw_hits, n_iters, raw_avg);
    if let Some(k) = first_fail {
        println!("    First decoding failure at iteration k={}.", k);
    } else {
        println!("    No decoding failures even without explicit cleanup —");
        println!("    the EAM's softmax read is doing the cleanup implicitly.");
    }
    println!();

    // The headline diagnostic: similarity gap.
    println!("  Summary:");
    println!("    With cleanup:    decoded {}/{}   avg sim {:.3}",
        clean_hits, n_iters, clean_avg);
    println!("    Without cleanup: decoded {}/{}   avg sim {:.3}",
        raw_hits, n_iters, raw_avg);
    println!("    Similarity gap (cleanup advantage): {:+.3}", clean_avg - raw_avg);

    assert_eq!(clean_hits, n_iters, "cleanup variant should be perfect");
    assert!(clean_avg > raw_avg - 0.01,
        "cleanup should not be worse on avg sim");
    println!("\n  PASS: substrate iterates discretely; cleanup tightens the grip.\n");
}

// ===========================================================================
//       SECTION 3 — Spaun count-from-N: working memory + counting
// ===========================================================================

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Spaun count-from-N task (WM × counting composition)");
    println!("======================================================================");
    println!("Input: (start=3, count=4). Output: the sequence 3, 4, 5, 6, 7.");
    println!("Each step: succ on current, push into a position-bound WM buffer.");
    println!("Then read back positions 0..count to recover the full sequence.\n");

    let succ = Successor::new(rng);
    let pos_keys: Vec<Vec<f64>> =
        (0..WM_HORIZON).map(|_| rand_unit(rng)).collect();

    let count_into_buffer = |start: usize, count: usize| -> Vec<usize> {
        let mut buffer = vec![0.0; DIM];
        let mut current_v = succ.numbers[start].clone();

        let push = |buf: &mut Vec<f64>, pos: usize, v: &[f64]| {
            let bound = bind_vec(&pos_keys[pos], v);
            for i in 0..buf.len() { buf[i] += bound[i]; }
        };
        push(&mut buffer, 0, &current_v);

        for k in 1..=count {
            let (clean_next, _, _) = succ.clean_succ(&current_v);
            current_v = clean_next;
            push(&mut buffer, k, &current_v);
        }

        let buf_n = vec_ops::normalize(&buffer);
        (0..=count).map(|k| {
            let noisy = unbind_vec(&buf_n, &pos_keys[k]);
            succ.decode(&noisy).0
        }).collect()
    };

    let tasks: &[(usize, usize)] = &[(3, 4), (0, 5), (7, 3), (5, 6)];
    for (start, count) in tasks {
        let sequence = count_into_buffer(*start, *count);
        let expected: Vec<usize> =
            (0..=*count).map(|k| (start + k) % N_DIGITS).collect();
        let ok = sequence == expected;
        let mark = if ok { "✓" } else { "✗" };
        println!("  count(start={}, count={}):", start, count);
        println!("    expected: {:?}", expected);
        println!("    produced: {:?}  {}", sequence, mark);
        assert_eq!(sequence, expected,
            "count-from-N failed for ({}, {})", start, count);
    }
    println!("\n  PASS: WM + counting compose into Spaun's serial-output task.\n");
}

// ===========================================================================
//             SECTION 4 — Addition as iterated counting
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Addition as iterated counting");
    println!("======================================================================");
    println!("a + b = apply succ b times starting from a. No arithmetic logic anywhere —");
    println!("just iteration over the successor primitive.\n");

    let succ = Successor::new(rng);

    let add = |a: usize, b: usize| -> usize {
        let mut current_v = succ.numbers[a].clone();
        let mut current_idx = a;
        for _ in 0..b {
            let (cv, idx, _) = succ.clean_succ(&current_v);
            current_v = cv;
            current_idx = idx;
        }
        current_idx
    };

    let pairs: &[(usize, usize)] = &[
        (3, 4), (2, 5), (1, 8), (4, 3), (0, 9),
        (7, 5), (8, 7), (9, 9),  // these wrap mod 10
    ];
    let mut hits = 0;
    for (a, b) in pairs {
        let got = add(*a, *b);
        let expected = (a + b) % N_DIGITS;
        let ok = got == expected;
        if ok { hits += 1; }
        let mark = if ok { "✓" } else { "✗" };
        let wraps = if a + b >= N_DIGITS { " (wraps)" } else { "" };
        println!("  {} + {} = {}  (expected {}{}) {}",
            a, b, got, expected, wraps, mark);
        assert_eq!(got, expected, "addition failed for {} + {}", a, b);
    }
    println!("\n  {}/{} additions correct.", hits, pairs.len());
    println!("\n  PASS: arithmetic emerges from iteration over the substrate.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    section_4(&mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — counting on a continuous substrate");
    println!("======================================================================");
    println!("  Discrete arithmetic from iteration. No ALU, no logic gates, no rules.");
    println!("  The substrate now has: state, transformation, decision, and");
    println!("  iteration — a complete computational core.");
}
