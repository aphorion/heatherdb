//! Pattern induction — Spaun's "figure out the rule" capability on the
//! substrate.
//!
//! Given a partial sequence (e.g. 1, 2, 4, 8), discover the
//! transformation that generated it, then apply that transformation to
//! predict the next term. Spaun (Eliasmith 2012) treated this as one
//! of its eight cognitive tasks; here it falls out of composing the
//! modules we already have.
//!
//! How it works on the substrate:
//!
//!   - Each candidate **operator** (succ, pred, double, add2, ...) is
//!     stored in an EAM as a bundle of bound (CURRENT, NEXT) pairs,
//!     tagged with the operator's identity vector.
//!   - Application is one substrate read: `bind(OP_TAG, op) +
//!     bind(CURRENT, n)` → read → unbind NEXT → clean up.
//!   - Inference iterates over the operator library, checking how well
//!     each predicts every consecutive pair in the observed sequence.
//!     The best match wins.
//!   - Prediction applies the inferred operator to the last term.
//!
//! Sections:
//!   1. Operator library — verify each stored operator works.
//!   2. Inference — given a sequence, identify the generating rule.
//!   3. Prediction — apply the inferred rule to extend the sequence.
//!   4. Evidence accumulation — ambiguous prefixes disambiguate as
//!      more terms are observed. Confidence is interpretable.
//!
//! Run: `cargo run --release --example pattern_induction -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{SeedableRng, rngs::StdRng};

const DIM: usize = 1024;
const N_DIGITS: usize = 10;
const BETA: f64 = 15.0;
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

// ----- The operator library ------------------------------------------------

struct OperatorLibrary {
    eam: MiniEAM,
    op_key: Vec<f64>,
    current_key: Vec<f64>,
    next_key: Vec<f64>,
    numbers: Vec<Vec<f64>>,
    operators: Vec<(String, Vec<f64>)>,
}

impl OperatorLibrary {
    fn new(rng: &mut StdRng) -> Self {
        let op_key = rand_unit(rng);
        let current_key = rand_unit(rng);
        let next_key = rand_unit(rng);
        let numbers: Vec<Vec<f64>> =
            (0..N_DIGITS).map(|_| rand_unit(rng)).collect();
        Self {
            eam: MiniEAM::new(),
            op_key, current_key, next_key, numbers,
            operators: Vec::new(),
        }
    }

    /// Add an operator to the library. `f` is the integer function the
    /// operator implements (e.g. |n| (n + 1) % 10 for succ). The
    /// substrate stores all (n, f(n)) pairs tagged with this operator.
    fn add_operator<F: Fn(usize) -> usize>(
        &mut self, name: &str, f: F, rng: &mut StdRng,
    ) {
        let tag = rand_unit(rng);
        for n in 0..N_DIGITS {
            let next_n = f(n);
            let entry = bundle(&[
                &bind_vec(&self.op_key, &tag),
                &bind_vec(&self.current_key, &self.numbers[n]),
                &bind_vec(&self.next_key, &self.numbers[next_n]),
            ]);
            self.eam.write(&entry);
        }
        self.operators.push((name.into(), tag));
    }

    fn op_tag(&self, name: &str) -> &[f64] {
        &self.operators.iter().find(|(n, _)| n == name).unwrap().1
    }

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

    /// Apply a known operator: bind tag + current, read, unbind next.
    fn apply(&self, op_name: &str, n: usize) -> usize {
        let query = bundle(&[
            &bind_vec(&self.op_key, self.op_tag(op_name)),
            &bind_vec(&self.current_key, &self.numbers[n]),
        ]);
        let recalled = self.eam.read(&query);
        let noisy = unbind_vec(&recalled, &self.next_key);
        self.decode(&noisy).0
    }

    /// Infer which operator best explains the observed sequence.
    /// Returns the operator name and its agreement rate over consecutive
    /// pairs (1.0 = explains every pair perfectly).
    fn infer(&self, sequence: &[usize]) -> Vec<(String, f64)> {
        if sequence.len() < 2 { return Vec::new(); }
        let mut scores: Vec<(String, f64)> = Vec::new();
        for (name, _) in &self.operators {
            let mut hits = 0;
            for w in sequence.windows(2) {
                if self.apply(name, w[0]) == w[1] { hits += 1; }
            }
            let score = hits as f64 / (sequence.len() - 1) as f64;
            scores.push((name.clone(), score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    /// Best-fit operator (most-supported by the evidence).
    fn best(&self, sequence: &[usize]) -> (String, f64) {
        self.infer(sequence).into_iter().next().unwrap()
    }
}

// ===========================================================================
//                  SECTION 1 — Operator library
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Operator library (each operator is a substrate read)");
    println!("======================================================================");
    println!("Five operators stored as (CURRENT, NEXT) pairs tagged by operator");
    println!("identity. Application is one bound query against the EAM.\n");

    let mut lib = OperatorLibrary::new(rng);
    lib.add_operator("succ",   |n| (n + 1) % N_DIGITS, rng);
    lib.add_operator("pred",   |n| (n + N_DIGITS - 1) % N_DIGITS, rng);
    lib.add_operator("add2",   |n| (n + 2) % N_DIGITS, rng);
    lib.add_operator("add3",   |n| (n + 3) % N_DIGITS, rng);
    lib.add_operator("double", |n| (2 * n) % N_DIGITS, rng);

    println!("  Library: {} operators × {} digits = {} stored entries",
        lib.operators.len(), N_DIGITS, lib.eam.locs.len());
    println!();

    // Check each operator on a sample input
    let cases: &[(&str, usize, usize)] = &[
        ("succ",   3, 4),
        ("pred",   5, 4),
        ("add2",   3, 5),
        ("add3",   4, 7),
        ("double", 4, 8),
        ("double", 7, 4),  // 14 mod 10
        ("succ",   9, 0),  // wrap
    ];
    let mut hits = 0;
    for (op, input, expected) in cases {
        let got = lib.apply(op, *input);
        let ok = got == *expected;
        if ok { hits += 1; }
        let mark = if ok { "✓" } else { "✗" };
        println!("  {}({}) = {}  (expected {}) {}", op, input, got, expected, mark);
        assert_eq!(got, *expected);
    }
    println!("\n  {}/{} operator applications correct.", hits, cases.len());
    println!("  PASS: substrate stores and applies discrete operators.\n");
}

// ===========================================================================
//                  SECTION 2 — Operator inference
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Inferring the rule from a sequence");
    println!("======================================================================");
    println!("Given a sequence, the substrate checks each candidate operator");
    println!("against every consecutive pair. Best agreement wins.\n");

    let mut lib = OperatorLibrary::new(rng);
    lib.add_operator("succ",   |n| (n + 1) % N_DIGITS, rng);
    lib.add_operator("pred",   |n| (n + N_DIGITS - 1) % N_DIGITS, rng);
    lib.add_operator("add2",   |n| (n + 2) % N_DIGITS, rng);
    lib.add_operator("add3",   |n| (n + 3) % N_DIGITS, rng);
    lib.add_operator("double", |n| (2 * n) % N_DIGITS, rng);

    let cases: &[(&[usize], &str)] = &[
        (&[1, 2, 3, 4],    "succ"),
        (&[7, 6, 5, 4],    "pred"),
        (&[1, 3, 5, 7],    "add2"),
        (&[0, 3, 6, 9],    "add3"),
        (&[1, 2, 4, 8],    "double"),
        (&[3, 6, 2, 4],    "double"),  // 3*2=6, 6*2=2(wrap), 2*2=4
        (&[0, 1, 2, 3, 4, 5], "succ"),
        (&[9, 8, 7, 6, 5], "pred"),
    ];

    for (seq, expected_op) in cases {
        let ranked = lib.infer(seq);
        let (winner, score) = &ranked[0];
        let (runner_up, runner_score) = &ranked[1];
        let mark = if winner == expected_op { "✓" } else { "✗" };
        println!("  seq={:?}", seq);
        println!("    winner: {:<7} ({:.0}%)   runner-up: {:<7} ({:.0}%)  {}",
            winner, score * 100.0, runner_up, runner_score * 100.0, mark);
        assert_eq!(winner, expected_op, "wrong operator for {:?}", seq);
    }
    println!("\n  PASS: substrate inferred the correct rule for every sequence.\n");
}

// ===========================================================================
//                  SECTION 3 — Next-term prediction
// ===========================================================================

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Predicting the next term");
    println!("======================================================================");
    println!("Infer the rule, then apply it to the last term to extend the sequence.\n");

    let mut lib = OperatorLibrary::new(rng);
    lib.add_operator("succ",   |n| (n + 1) % N_DIGITS, rng);
    lib.add_operator("pred",   |n| (n + N_DIGITS - 1) % N_DIGITS, rng);
    lib.add_operator("add2",   |n| (n + 2) % N_DIGITS, rng);
    lib.add_operator("add3",   |n| (n + 3) % N_DIGITS, rng);
    lib.add_operator("double", |n| (2 * n) % N_DIGITS, rng);

    let cases: &[(&[usize], usize)] = &[
        (&[1, 2, 3, 4],     5),
        (&[8, 7, 6, 5],     4),
        (&[1, 3, 5, 7],     9),
        (&[0, 3, 6],        9),
        (&[1, 2, 4, 8],     6),     // 16 mod 10
        (&[2, 4, 6, 8],     0),     // 10 mod 10
    ];

    for (seq, expected_next) in cases {
        let (op, conf) = lib.best(seq);
        let last = *seq.last().unwrap();
        let predicted = lib.apply(&op, last);
        let mark = if predicted == *expected_next { "✓" } else { "✗" };
        println!("  seq={:?}  → rule={:<7} ({:.0}%)  predict {}  (expected {}) {}",
            seq, op, conf * 100.0, predicted, expected_next, mark);
        assert_eq!(predicted, *expected_next);
    }
    println!("\n  PASS: pattern → rule → prediction, end-to-end on substrate.\n");
}

// ===========================================================================
//        SECTION 4 — Evidence accumulation (Bayesian feel without Bayes)
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Evidence accumulation under ambiguity");
    println!("======================================================================");
    println!("(1,2) is ambiguous — fits succ AND double. As more terms arrive,");
    println!("the substrate's operator ranking resolves. Confidence is interpretable.\n");

    let mut lib = OperatorLibrary::new(rng);
    lib.add_operator("succ",   |n| (n + 1) % N_DIGITS, rng);
    lib.add_operator("pred",   |n| (n + N_DIGITS - 1) % N_DIGITS, rng);
    lib.add_operator("add2",   |n| (n + 2) % N_DIGITS, rng);
    lib.add_operator("add3",   |n| (n + 3) % N_DIGITS, rng);
    lib.add_operator("double", |n| (2 * n) % N_DIGITS, rng);

    // Run two prefix chains:
    //   doubling:    1, 2, 4, 8, 6, 2
    //   incrementing: 1, 2, 3, 4, 5, 6
    let chains: &[(&str, &[usize])] = &[
        ("doubling chain",    &[1, 2, 4, 8, 6, 2]),
        ("incrementing chain", &[1, 2, 3, 4, 5, 6]),
    ];

    for (label, full) in chains {
        println!("  {} (full = {:?}):", label, full);
        for n in 2..=full.len() {
            let prefix = &full[..n];
            let ranked = lib.infer(prefix);
            let winners: Vec<String> = ranked.iter()
                .filter(|(_, s)| *s == ranked[0].1)
                .map(|(n, _)| n.clone())
                .collect();
            let top_score = ranked[0].1;
            let runner_score = ranked.get(1).map(|(_, s)| *s).unwrap_or(0.0);
            let confidence = top_score - runner_score; // gap to next-best
            let tag = if winners.len() == 1 {
                format!("{}", winners[0])
            } else {
                format!("tied: {{{}}}", winners.join(", "))
            };
            println!("    prefix {:?}   → {:<25}  top={:.0}%  gap={:+.0}%",
                prefix, tag, top_score * 100.0, confidence * 100.0);
        }
        println!();
    }

    // Specific assertions
    // (1, 2) should tie between succ and double (both produce 2 from 1).
    let amb = lib.infer(&[1, 2]);
    let top_score = amb[0].1;
    let tied: Vec<&String> = amb.iter()
        .filter(|(_, s)| *s == top_score)
        .map(|(n, _)| n)
        .collect();
    assert!(tied.len() >= 2, "(1,2) should be ambiguous, got unique winner: {:?}", amb);
    println!("  Ambiguity at (1,2): tied operators = {:?}", tied);

    // (1, 2, 4) resolves to double (succ would give 3 not 4).
    let (op_at_3, _) = lib.best(&[1, 2, 4]);
    assert_eq!(op_at_3, "double");
    // (1, 2, 3) resolves to succ.
    let (op_alt, _) = lib.best(&[1, 2, 3]);
    assert_eq!(op_alt, "succ");

    println!("  Resolution: (1,2,4) → double ✓     (1,2,3) → succ ✓");
    println!("\n  PASS: substrate's operator confidence is interpretable and");
    println!("        grows with evidence — no probabilistic model, just");
    println!("        match-rate over the candidate library.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    section_4(&mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — pattern induction on the substrate.");
    println!("======================================================================");
    println!("  Sequence forecasting, anomaly detection, next-best-action — all");
    println!("  the same shape: store operators, observe data, infer the rule,");
    println!("  apply. No training, interpretable, one-shot.");
}
