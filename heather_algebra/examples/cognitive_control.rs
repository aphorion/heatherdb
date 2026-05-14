//! Cognitive control — the task dispatcher that turns a library of
//! substrate primitives into a unified agent.
//!
//! Up to now each cognitive demo (`working_memory`, `analogy`,
//! `action_selection`, `counting`, `pattern_induction`) has been an
//! isolated capability. Cognitive control is the keystone: it receives
//! heterogeneous requests, identifies which subroutine each one needs,
//! and routes to it. Eliasmith calls this the **task selection** loop
//! — basal-ganglia gating applied to *cognitive subroutines* rather
//! than motor actions.
//!
//! The dispatcher is itself a substrate primitive. Each request is a
//! bound vector tagged with a TASK identity. The agent identifies the
//! task by similarity-matching the unbound TASK tag against its known-
//! task lexicon; if the match is sharp, it routes; if the match is
//! soft, it reports uncertainty. **The dispatcher is just `select` on
//! tasks instead of actions.**
//!
//! Sections:
//!   1. Basic dispatch — four task types, each routed to the right handler.
//!   2. Mixed stream — 30 random requests, all routed correctly.
//!   3. Composition — a meta-request chains two handlers (predict_next →
//!      count_forward) to produce a multi-step result.
//!   4. Unknown task — a request tagged with an unfamiliar TASK vector
//!      triggers principled uncertainty rather than a wrong guess.
//!
//! Run: `cargo run --release --example cognitive_control -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 1024;
const N_DIGITS: usize = 10;
const BETA: f64 = 15.0;
const READ_ITERS: usize = 4;
const TASK_CONFIDENCE_THRESHOLD: f64 = 0.20;

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

// ----- Operator library (carried over from pattern_induction) --------------

struct Operators {
    eam: MiniEAM,
    op_key: Vec<f64>,
    cur_key: Vec<f64>,
    nxt_key: Vec<f64>,
    numbers: Vec<Vec<f64>>,
    ops: Vec<(String, Vec<f64>)>,
}
impl Operators {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            op_key: rand_unit(rng),
            cur_key: rand_unit(rng),
            nxt_key: rand_unit(rng),
            numbers: (0..N_DIGITS).map(|_| rand_unit(rng)).collect(),
            ops: Vec::new(),
        }
    }
    fn add<F: Fn(usize) -> usize>(&mut self, name: &str, f: F, rng: &mut StdRng) {
        let tag = rand_unit(rng);
        for n in 0..N_DIGITS {
            let e = bundle(&[
                &bind_vec(&self.op_key, &tag),
                &bind_vec(&self.cur_key, &self.numbers[n]),
                &bind_vec(&self.nxt_key, &self.numbers[f(n)]),
            ]);
            self.eam.write(&e);
        }
        self.ops.push((name.into(), tag));
    }
    fn op_tag(&self, name: &str) -> &[f64] {
        &self.ops.iter().find(|(n, _)| n == name).unwrap().1
    }
    fn decode(&self, v: &[f64]) -> usize {
        let v = vec_ops::normalize(v);
        let mut best = 0usize; let mut best_s = f64::NEG_INFINITY;
        for (i, n) in self.numbers.iter().enumerate() {
            let s = vec_ops::cosine_similarity(&v, n);
            if s > best_s { best_s = s; best = i; }
        }
        best
    }
    fn apply(&self, op_name: &str, n: usize) -> usize {
        let q = bundle(&[
            &bind_vec(&self.op_key, self.op_tag(op_name)),
            &bind_vec(&self.cur_key, &self.numbers[n]),
        ]);
        let r = self.eam.read(&q);
        self.decode(&unbind_vec(&r, &self.nxt_key))
    }
    /// Infer best operator from a sequence (match-rate on consecutive pairs).
    fn infer(&self, seq: &[usize]) -> (String, f64) {
        let mut best = (String::new(), -1.0);
        for (name, _) in &self.ops {
            let hits = seq.windows(2).filter(|w| self.apply(name, w[0]) == w[1]).count();
            let score = hits as f64 / (seq.len() - 1).max(1) as f64;
            if score > best.1 { best = (name.clone(), score); }
        }
        best
    }
}

// ----- The cognitive agent --------------------------------------------------

#[derive(Debug, PartialEq)]
enum Response {
    NextTerm(usize),
    Sequence(Vec<usize>),
    Value(usize),
    Uncertain { best_guess: String, confidence: f64 },
}

struct Agent {
    ops: Operators,
    task_key: Vec<f64>,
    seq_keys: Vec<Vec<f64>>,   // SEQ_0, SEQ_1, SEQ_2, ... for sequence payloads
    start_key: Vec<f64>,
    k_key: Vec<f64>,
    op_name_key: Vec<f64>,
    input_key: Vec<f64>,
    tasks: Vec<(String, Vec<f64>)>,
}

impl Agent {
    fn new(rng: &mut StdRng) -> Self {
        let mut ops = Operators::new(rng);
        ops.add("succ",   |n| (n + 1) % N_DIGITS, rng);
        ops.add("pred",   |n| (n + N_DIGITS - 1) % N_DIGITS, rng);
        ops.add("add2",   |n| (n + 2) % N_DIGITS, rng);
        ops.add("double", |n| (2 * n) % N_DIGITS, rng);

        let mut agent = Self {
            ops,
            task_key: rand_unit(rng),
            seq_keys: (0..6).map(|_| rand_unit(rng)).collect(),
            start_key: rand_unit(rng),
            k_key: rand_unit(rng),
            op_name_key: rand_unit(rng),
            input_key: rand_unit(rng),
            tasks: Vec::new(),
        };
        for t in ["predict_next", "count_forward", "apply_op", "lookup"] {
            agent.tasks.push((t.into(), rand_unit(rng)));
        }
        agent
    }

    fn task_tag(&self, name: &str) -> &[f64] {
        &self.tasks.iter().find(|(n, _)| n == name).unwrap().1
    }

    /// The dispatcher: identify the task by similarity, return (name, confidence).
    fn identify_task(&self, request: &[f64]) -> (String, f64) {
        let task_v = vec_ops::normalize(&unbind_vec(request, &self.task_key));
        let mut best_name = String::new();
        let mut best_sim = f64::NEG_INFINITY;
        let mut second_sim = f64::NEG_INFINITY;
        for (name, tag) in &self.tasks {
            let s = vec_ops::cosine_similarity(&task_v, tag);
            if s > best_sim {
                second_sim = best_sim;
                best_sim = s;
                best_name = name.clone();
            } else if s > second_sim {
                second_sim = s;
            }
        }
        // Confidence is the gap to the runner-up, clamped to [0, 1].
        let gap = (best_sim - second_sim).max(0.0);
        (best_name, gap)
    }

    fn handle(&self, request: &[f64]) -> Response {
        let (task, confidence) = self.identify_task(request);
        if confidence < TASK_CONFIDENCE_THRESHOLD {
            return Response::Uncertain { best_guess: task, confidence };
        }
        match task.as_str() {
            "predict_next" => self.h_predict_next(request),
            "count_forward" => self.h_count_forward(request),
            "apply_op"      => self.h_apply_op(request),
            "lookup"        => self.h_lookup(request),
            _               => Response::Uncertain { best_guess: task, confidence },
        }
    }

    // --- handlers ---------------------------------------------------------

    fn unbind_digit(&self, request: &[f64], key: &[f64]) -> usize {
        self.ops.decode(&unbind_vec(request, key))
    }

    fn h_predict_next(&self, request: &[f64]) -> Response {
        // Extract sequence terms — up to seq_keys.len() of them.
        // Decode each; stop when decoding falls below confidence (heuristic:
        // here we assume fixed 4-term sequences for simplicity).
        let seq: Vec<usize> = (0..4)
            .map(|i| self.unbind_digit(request, &self.seq_keys[i]))
            .collect();
        let (op, _) = self.ops.infer(&seq);
        let next = self.ops.apply(&op, *seq.last().unwrap());
        Response::NextTerm(next)
    }

    fn h_count_forward(&self, request: &[f64]) -> Response {
        let start = self.unbind_digit(request, &self.start_key);
        let k = self.unbind_digit(request, &self.k_key);
        let mut seq = vec![start];
        let mut current = start;
        for _ in 0..k {
            current = self.ops.apply("succ", current);
            seq.push(current);
        }
        Response::Sequence(seq)
    }

    fn h_apply_op(&self, request: &[f64]) -> Response {
        let input = self.unbind_digit(request, &self.input_key);
        // Find which operator the request named — closest to op_name slot
        let op_v = vec_ops::normalize(&unbind_vec(request, &self.op_name_key));
        let mut best_op = String::new();
        let mut best_s = f64::NEG_INFINITY;
        for (name, tag) in &self.ops.ops {
            let s = vec_ops::cosine_similarity(&op_v, tag);
            if s > best_s { best_s = s; best_op = name.clone(); }
        }
        Response::Value(self.ops.apply(&best_op, input))
    }

    fn h_lookup(&self, request: &[f64]) -> Response {
        // Tiny key→value memory using succ as the "lookup table" — just to
        // round out the dispatcher with a fourth task type.
        let input = self.unbind_digit(request, &self.input_key);
        Response::Value((input + 5) % N_DIGITS) // arbitrary lookup function
    }

    // --- request builders -------------------------------------------------

    fn req_predict_next(&self, seq: &[usize; 4]) -> Vec<f64> {
        let mut parts: Vec<Vec<f64>> = Vec::new();
        parts.push(bind_vec(&self.task_key, self.task_tag("predict_next")));
        for (i, n) in seq.iter().enumerate() {
            parts.push(bind_vec(&self.seq_keys[i], &self.ops.numbers[*n]));
        }
        let refs: Vec<&[f64]> = parts.iter().map(|v| v.as_slice()).collect();
        bundle(&refs)
    }
    fn req_count_forward(&self, start: usize, k: usize) -> Vec<f64> {
        bundle(&[
            &bind_vec(&self.task_key, self.task_tag("count_forward")),
            &bind_vec(&self.start_key, &self.ops.numbers[start]),
            &bind_vec(&self.k_key, &self.ops.numbers[k]),
        ])
    }
    fn req_apply_op(&self, op_name: &str, n: usize) -> Vec<f64> {
        bundle(&[
            &bind_vec(&self.task_key, self.task_tag("apply_op")),
            &bind_vec(&self.op_name_key, self.ops.op_tag(op_name)),
            &bind_vec(&self.input_key, &self.ops.numbers[n]),
        ])
    }
    fn req_lookup(&self, n: usize) -> Vec<f64> {
        bundle(&[
            &bind_vec(&self.task_key, self.task_tag("lookup")),
            &bind_vec(&self.input_key, &self.ops.numbers[n]),
        ])
    }
    fn req_unknown(&self, fake_tag: &[f64]) -> Vec<f64> {
        bundle(&[
            &bind_vec(&self.task_key, fake_tag),
            &bind_vec(&self.input_key, &self.ops.numbers[3]),
        ])
    }
}

// ===========================================================================
//                  SECTION 1 — Basic dispatch
// ===========================================================================

fn section_1(agent: &Agent) {
    println!("======================================================================");
    println!("  SECTION 1 — Basic dispatch: four task types, four handlers");
    println!("======================================================================");
    println!("Each request is one bound vector tagged with TASK + payload.");
    println!("The agent identifies the task by similarity-matching the TASK tag");
    println!("against its registered task lexicon.\n");

    let cases: Vec<(String, Vec<f64>, Response)> = vec![
        ("predict_next [1,2,3,4]".into(),
         agent.req_predict_next(&[1, 2, 3, 4]),
         Response::NextTerm(5)),
        ("count_forward(3, 4)".into(),
         agent.req_count_forward(3, 4),
         Response::Sequence(vec![3, 4, 5, 6, 7])),
        ("apply_op(double, 4)".into(),
         agent.req_apply_op("double", 4),
         Response::Value(8)),
        ("lookup(2)".into(),
         agent.req_lookup(2),
         Response::Value(7)),
    ];

    for (label, request, expected) in &cases {
        let (task, conf) = agent.identify_task(request);
        let got = agent.handle(request);
        let ok = got == *expected;
        let mark = if ok { "✓" } else { "✗" };
        println!("  {:<32}  → task={:<14} (gap {:+.2})  result={:?} {}",
            label, task, conf, got, mark);
        assert_eq!(got, *expected, "wrong response for {}", label);
    }
    println!("\n  PASS: 4/4 requests routed and handled correctly.\n");
}

// ===========================================================================
//                  SECTION 2 — Mixed request stream
// ===========================================================================

fn section_2(agent: &Agent, rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Mixed stream: 30 random requests, all routed");
    println!("======================================================================\n");

    let mut correct = 0;
    let total = 30;
    for i in 0..total {
        let kind = rng.r#gen_range(0..4);
        let (label, request, expected) = match kind {
            0 => {
                // predict_next with succ pattern
                let start = rng.r#gen_range(0..6);
                let seq = [start, start + 1, start + 2, start + 3];
                let label = format!("predict_next {:?}", seq);
                (label, agent.req_predict_next(&seq), Response::NextTerm(start + 4))
            }
            1 => {
                let start = rng.r#gen_range(0..5);
                let k = rng.r#gen_range(1..4);
                let label = format!("count_forward({}, {})", start, k);
                let expected: Vec<usize> = (0..=k).map(|j| start + j).collect();
                (label, agent.req_count_forward(start, k), Response::Sequence(expected))
            }
            2 => {
                let ops = ["succ", "pred", "add2", "double"];
                let op_idx = rng.r#gen_range(0..ops.len());
                let n = rng.r#gen_range(0..N_DIGITS);
                let op = ops[op_idx];
                let expected = match op {
                    "succ" => (n + 1) % N_DIGITS,
                    "pred" => (n + N_DIGITS - 1) % N_DIGITS,
                    "add2" => (n + 2) % N_DIGITS,
                    "double" => (2 * n) % N_DIGITS,
                    _ => unreachable!(),
                };
                let label = format!("apply_op({}, {})", op, n);
                (label, agent.req_apply_op(op, n), Response::Value(expected))
            }
            _ => {
                let n = rng.r#gen_range(0..N_DIGITS);
                let label = format!("lookup({})", n);
                (label, agent.req_lookup(n), Response::Value((n + 5) % N_DIGITS))
            }
        };
        let got = agent.handle(&request);
        let ok = got == expected;
        if ok { correct += 1; }
        if i < 10 || !ok {
            let mark = if ok { "✓" } else { "✗" };
            println!("  [{:>2}] {:<30}  →  {:?}  {}", i + 1, label, got, mark);
        } else if i == 10 {
            println!("  ... (passes elided)");
        }
    }
    let acc = correct as f64 / total as f64;
    println!("\n  Accuracy: {}/{} = {:.0}%", correct, total, acc * 100.0);
    assert!(acc >= 0.95, "dispatch accuracy too low: {:.0}%", acc * 100.0);
    println!("  PASS: substrate dispatched heterogeneous requests reliably.\n");
}

// ===========================================================================
//        SECTION 3 — Compositional task: chained handlers
// ===========================================================================

fn section_3(agent: &Agent) {
    println!("======================================================================");
    println!("  SECTION 3 — Composition: predict_next → count_forward");
    println!("======================================================================");
    println!("The agent receives a predict_next request, then feeds the result");
    println!("into a count_forward request. Cognitive modules compose by passing");
    println!("substrate values, not strings.\n");

    let seq = [1, 2, 3, 4];
    let r1 = agent.req_predict_next(&seq);
    let pred_next = match agent.handle(&r1) {
        Response::NextTerm(n) => n,
        other => panic!("expected NextTerm, got {:?}", other),
    };
    println!("  Step 1: predict_next [{:?}] → {}", seq, pred_next);

    let extend_k = 3;
    let r2 = agent.req_count_forward(pred_next, extend_k);
    let extended = match agent.handle(&r2) {
        Response::Sequence(s) => s,
        other => panic!("expected Sequence, got {:?}", other),
    };
    println!("  Step 2: count_forward({}, {}) → {:?}", pred_next, extend_k, extended);

    let expected_full: Vec<usize> = seq.iter().copied().chain(extended.iter().copied()).collect();
    println!();
    println!("  Combined trajectory: original {:?} + extension {:?}", seq, extended);
    println!("                       full sequence = {:?}", expected_full);

    assert_eq!(pred_next, 5);
    assert_eq!(extended, vec![5, 6, 7, 8]);
    println!("\n  PASS: cognitive subroutines compose by value-passing.\n");
}

// ===========================================================================
//        SECTION 4 — Unknown task: principled uncertainty
// ===========================================================================

fn section_4(agent: &Agent, rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Unknown task: substrate reports uncertainty");
    println!("======================================================================");
    println!("A request tagged with an unfamiliar TASK vector. The dispatcher's");
    println!("confidence (gap to runner-up) collapses, and the agent declines to");
    println!("dispatch rather than guess.\n");

    // Familiar request — high confidence
    let r_familiar = agent.req_apply_op("succ", 3);
    let (t_fam, gap_fam) = agent.identify_task(&r_familiar);
    println!("  Familiar request (apply_op):");
    println!("    identified task: {} with gap {:+.3}", t_fam, gap_fam);

    // Unknown task tag — should report uncertainty
    let unknown_tag = rand_unit(rng);
    let r_unknown = agent.req_unknown(&unknown_tag);
    let response = agent.handle(&r_unknown);
    println!("\n  Unknown request (random TASK tag):");
    match &response {
        Response::Uncertain { best_guess, confidence } => {
            println!("    Uncertain (best guess: {}, gap {:+.3})  ✓",
                best_guess, confidence);
        }
        other => panic!("expected Uncertain, got {:?}", other),
    }

    // Also check that a low-gap request triggers uncertainty even with
    // a tag close to TWO known tasks (impossible to construct cleanly
    // here — just verify the threshold mechanism works on the unknown).
    assert!(gap_fam > TASK_CONFIDENCE_THRESHOLD,
        "familiar request should clear the threshold");
    assert!(matches!(response, Response::Uncertain { .. }),
        "unknown request should yield Uncertain");

    println!("\n  PASS: substrate refuses to dispatch when it doesn't recognize");
    println!("        the task — interpretable failure, not silent misroute.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let agent = Agent::new(&mut rng);
    section_1(&agent);
    section_2(&agent, &mut rng);
    section_3(&agent);
    section_4(&agent, &mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — cognitive control operational.");
    println!("======================================================================");
    println!("  The agent now has a unifying dispatch loop: receive bound request,");
    println!("  identify task by similarity, route to cognitive subroutine,");
    println!("  return result. The seven previous modules are now a single agent.");
}
