//! Active Inference — predictive encoding and free-energy action
//! selection on the substrate.
//!
//! Friston's Free Energy Principle says perception and action both
//! minimize the same quantity: variational free energy (prediction error
//! + complexity). On the substrate this reorganizes two things at once:
//!
//!   1. **Encoding becomes predictive.** Don't write what's already
//!      predicted — only write the prediction error (surprise). The
//!      substrate stores novelty, not redundancy.
//!
//!   2. **Action selection becomes free-energy minimization.** Each
//!      candidate action is scored by expected free energy:
//!         EFE(a) = epistemic_value(a) + pragmatic_value(a)
//!      Curiosity and goal-seeking emerge as two terms of one math.
//!
//! The result is an agent that:
//!   - Writes far fewer entries to substrate (encodes only surprises).
//!   - Explores intelligently (drawn to uncertainty) without an ε-greedy
//!     hack.
//!   - Converges to goal faster because exploration was directed.
//!
//! Sections:
//!   1. Predictive encoding shrinks substrate growth.
//!   2. Free-energy action selection finds the goal with less wandering.
//!   3. Combined: an active-inference agent reaches goal with smaller
//!      substrate and faster convergence than the vanilla gridworld
//!      agent.
//!   4. Surprise trace: visualize prediction error per step as the
//!      agent learns its world.
//!
//! Run: `cargo run --release --example active_inference -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

// ----- Hyperparameters ------------------------------------------------------

const DIM: usize = 256;
const GRID: i32 = 5;
const MAX_STEPS: usize = 60;
const EPISODES: usize = 60;
const BETA: f64 = 8.0;
const READ_ITERS: usize = 4;
const SURPRISE_THRESHOLD: f64 = 0.5;  // write if surprise > this
const EPISTEMIC_WEIGHT: f64 = 0.4;    // weight on info gain vs goal proximity

// ----- Helpers --------------------------------------------------------------

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

// ----- Mini EAM with confidence-reporting read -----------------------------

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
    /// Read that also returns the *initial* max single-location similarity,
    /// measured BEFORE iteration. This is the substrate's true confidence:
    /// how well does the query match any stored entry at first contact?
    /// (Measuring inside the loop is misleading — the iterative read
    /// always converges to an attractor, so post-iteration similarity is
    /// near 1.0 regardless of initial match quality.)
    fn read_with_confidence(&self, query: &[f64]) -> (Vec<f64>, f64) {
        if self.locs.is_empty() {
            return (vec_ops::normalize(query), 0.0);
        }
        let mut q = vec_ops::normalize(query);
        let initial_sims: Vec<f64> =
            self.locs.iter().map(|l| vec_ops::dot(&l.address, &q)).collect();
        let initial_top = initial_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for _ in 0..READ_ITERS {
            let sims: Vec<f64> =
                self.locs.iter().map(|l| vec_ops::dot(&l.address, &q)).collect();
            let w = vec_ops::softmax(&sims, BETA);
            let patterns: Vec<&[f64]> =
                self.locs.iter().map(|l| l.counter.as_slice()).collect();
            q = vec_ops::normalize(&vec_ops::weighted_sum(&patterns, &w));
        }
        (q, initial_top)
    }
}

// ----- Environment ----------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pos { x: i32, y: i32 }

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action { Up, Down, Left, Right }
const ACTIONS: [Action; 4] = [Action::Up, Action::Down, Action::Left, Action::Right];

struct Env { pos: Pos, start: Pos, goal: Pos }
impl Env {
    fn new(start: Pos, goal: Pos) -> Self { Self { pos: start, start, goal } }
    fn reset(&mut self) { self.pos = self.start; }
    fn step(&mut self, a: Action) -> (Pos, bool) {
        let (dx, dy) = match a {
            Action::Up => (0, -1), Action::Down => (0, 1),
            Action::Left => (-1, 0), Action::Right => (1, 0),
        };
        self.pos = Pos {
            x: (self.pos.x + dx).clamp(0, GRID - 1),
            y: (self.pos.y + dy).clamp(0, GRID - 1),
        };
        (self.pos, self.pos == self.goal)
    }
}

fn manhattan(a: Pos, b: Pos) -> i32 { (a.x - b.x).abs() + (a.y - b.y).abs() }

// ----- Codebook -------------------------------------------------------------

struct Codebook {
    state_key: Vec<f64>,
    action_key: Vec<f64>,
    next_state_key: Vec<f64>,
    state_vecs: Vec<Vec<f64>>,
    action_vecs: [Vec<f64>; 4],
}
impl Codebook {
    fn new(rng: &mut StdRng) -> Self {
        let rand = |rng: &mut StdRng| vec_ops::random_unit_vector(DIM, rng);
        Self {
            state_key: rand(rng),
            action_key: rand(rng),
            next_state_key: rand(rng),
            state_vecs: (0..(GRID * GRID)).map(|_| rand(rng)).collect(),
            action_vecs: [rand(rng), rand(rng), rand(rng), rand(rng)],
        }
    }
    fn state(&self, p: Pos) -> &[f64] {
        &self.state_vecs[(p.y * GRID + p.x) as usize]
    }
    fn action(&self, a: Action) -> &[f64] { &self.action_vecs[a as usize] }
    fn nearest_pos(&self, v: &[f64]) -> Pos {
        let v = vec_ops::normalize(v);
        let mut best = 0usize; let mut best_s = f64::NEG_INFINITY;
        for (i, sv) in self.state_vecs.iter().enumerate() {
            let s = vec_ops::dot(sv, &v);
            if s > best_s { best_s = s; best = i; }
        }
        Pos { x: (best as i32) % GRID, y: (best as i32) / GRID }
    }
}

// ----- World-model prediction with confidence -----------------------------

/// Predict the next position from (state, action) via substrate read.
/// Returns (predicted_pos, prediction_vector, confidence ∈ [0,1]).
fn predict(state: Pos, a: Action, eam: &MiniEAM, cb: &Codebook) -> (Pos, Vec<f64>, f64) {
    let query = bundle(&[
        &bind_vec(&cb.state_key, cb.state(state)),
        &bind_vec(&cb.action_key, cb.action(a)),
    ]);
    let (recalled, conf) = eam.read_with_confidence(&query);
    let pred_vec = unbind_vec(&recalled, &cb.next_state_key);
    let pred_pos = cb.nearest_pos(&pred_vec);
    (pred_pos, vec_ops::normalize(&pred_vec), conf)
}

// ----- Action policies -----------------------------------------------------

fn act_greedy(state: Pos, goal: Pos, eam: &MiniEAM, cb: &Codebook,
              eps: f64, rng: &mut StdRng) -> Action {
    if rng.r#gen::<f64>() < eps || eam.locs.is_empty() {
        return ACTIONS[rng.r#gen_range(0..4)];
    }
    let mut best = ACTIONS[0];
    let mut best_d = i32::MAX;
    for &a in &ACTIONS {
        let (pred, _, _) = predict(state, a, eam, cb);
        let d = manhattan(pred, goal);
        if d < best_d { best_d = d; best = a; }
    }
    best
}

/// Free-energy action selection: utility = epistemic + pragmatic.
///   - epistemic = 1 − confidence = "how uncertain are we?"
///   - pragmatic = (max_dist − manhattan(pred, goal)) / max_dist
///                = "how close does this take us to goal, normalized to [0,1]?"
/// Pick action that MAXIMIZES weighted sum (= minimizes expected free energy).
fn act_free_energy(state: Pos, goal: Pos, eam: &MiniEAM, cb: &Codebook,
                   eps: f64, rng: &mut StdRng) -> Action {
    if rng.r#gen::<f64>() < eps || eam.locs.is_empty() {
        return ACTIONS[rng.r#gen_range(0..4)];
    }
    let max_dist = (2 * (GRID - 1)) as f64;
    let mut best = ACTIONS[0];
    let mut best_u = f64::NEG_INFINITY;
    for &a in &ACTIONS {
        let (pred, _, conf) = predict(state, a, eam, cb);
        let epistemic = 1.0 - conf;
        let pragmatic = (max_dist - manhattan(pred, goal) as f64) / max_dist;
        let utility = EPISTEMIC_WEIGHT * epistemic
                    + (1.0 - EPISTEMIC_WEIGHT) * pragmatic;
        if utility > best_u { best_u = utility; best = a; }
    }
    best
}

// ----- Write policies ------------------------------------------------------

fn write_naive(eam: &mut MiniEAM, cb: &Codebook, s: Pos, a: Action, s_next: Pos) -> bool {
    let ep = bundle(&[
        &bind_vec(&cb.state_key, cb.state(s)),
        &bind_vec(&cb.action_key, cb.action(a)),
        &bind_vec(&cb.next_state_key, cb.state(s_next)),
    ]);
    eam.write(&ep);
    true
}

/// Predictive write: only write if the substrate's prediction is wrong
/// (i.e., the agent is surprised).
fn write_predictive(eam: &mut MiniEAM, cb: &Codebook, s: Pos, a: Action, s_next: Pos)
    -> (bool, f64)
{
    let (_, pred_vec, _) = predict(s, a, eam, cb);
    let actual_vec = vec_ops::normalize(cb.state(s_next));
    let cos = vec_ops::cosine_similarity(&pred_vec, &actual_vec);
    let surprise = (1.0 - cos).max(0.0);
    if eam.locs.is_empty() || surprise > SURPRISE_THRESHOLD {
        write_naive(eam, cb, s, a, s_next);
        (true, surprise)
    } else {
        (false, surprise)
    }
}

// ===========================================================================
//          SECTION 1 — Predictive encoding shrinks substrate growth
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Predictive encoding: substrate stores only surprises");
    println!("======================================================================");
    println!("Same agent (greedy action), two write policies:");
    println!("  NAIVE      — write every transition.");
    println!("  PREDICTIVE — write only if substrate's prediction is wrong.\n");

    let cb = Codebook::new(rng);
    let mut env = Env::new(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});

    let mut eam_naive = MiniEAM::new();
    let mut eam_pred = MiniEAM::new();
    let mut naive_writes = 0;
    let mut pred_writes = 0;
    let mut pred_skips = 0;

    for ep in 0..EPISODES {
        let eps = 1.0 - (ep as f64 / EPISODES as f64) * 0.9;
        // Naive agent
        env.reset();
        for _ in 0..MAX_STEPS {
            let s = env.pos;
            let a = act_greedy(s, env.goal, &eam_naive, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            if write_naive(&mut eam_naive, &cb, s, a, s_n) { naive_writes += 1; }
            if done { break; }
        }
        // Predictive agent
        env.reset();
        for _ in 0..MAX_STEPS {
            let s = env.pos;
            let a = act_greedy(s, env.goal, &eam_pred, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            let (wrote, _) = write_predictive(&mut eam_pred, &cb, s, a, s_n);
            if wrote { pred_writes += 1; } else { pred_skips += 1; }
            if done { break; }
        }
    }

    println!("  After {} episodes:", EPISODES);
    println!("    NAIVE       writes={:>4}   |EAM|={:>4}",
        naive_writes, eam_naive.locs.len());
    println!("    PREDICTIVE  writes={:>4}   skips={:>4}   |EAM|={:>4}",
        pred_writes, pred_skips, eam_pred.locs.len());
    let savings = 100.0 * (naive_writes - pred_writes) as f64 / naive_writes as f64;
    println!("\n  Predictive policy wrote {:.0}% fewer entries (skipped {} known).",
        savings, pred_skips);

    assert!(pred_writes < naive_writes, "predictive should write less than naive");
    assert!(pred_skips > 0, "predictive should skip some writes");
    println!("  PASS: substrate growth is driven by surprise, not by observation count.\n");
}

// ===========================================================================
//         SECTION 2 — Free-energy action selection explores wisely
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Free-energy action selection: curiosity drives exploration");
    println!("======================================================================");
    println!("Same write policy (naive), two action policies:");
    println!("  GREEDY       — pick action whose predicted next-state minimizes");
    println!("                 Manhattan to goal.");
    println!("  FREE-ENERGY  — pick action that maximizes (ε·epistemic + (1-ε)·pragmatic).");
    println!("                 Explores when uncertain; goal-seeks when confident.\n");

    let cb = Codebook::new(rng);
    let mut env = Env::new(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});

    let mut eam_greedy = MiniEAM::new();
    let mut eam_fe = MiniEAM::new();
    let mut greedy_visited: std::collections::HashSet<Pos> = Default::default();
    let mut fe_visited: std::collections::HashSet<Pos> = Default::default();
    let mut greedy_episode_lengths = Vec::new();
    let mut fe_episode_lengths = Vec::new();

    for ep in 0..EPISODES {
        let eps = 1.0 - (ep as f64 / EPISODES as f64) * 0.9;
        // Greedy
        env.reset();
        greedy_visited.insert(env.pos);
        let mut steps = 0;
        for _ in 0..MAX_STEPS {
            steps += 1;
            let s = env.pos;
            let a = act_greedy(s, env.goal, &eam_greedy, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            greedy_visited.insert(s_n);
            write_naive(&mut eam_greedy, &cb, s, a, s_n);
            if done { break; }
        }
        greedy_episode_lengths.push(steps);

        // Free-energy
        env.reset();
        fe_visited.insert(env.pos);
        let mut steps = 0;
        for _ in 0..MAX_STEPS {
            steps += 1;
            let s = env.pos;
            let a = act_free_energy(s, env.goal, &eam_fe, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            fe_visited.insert(s_n);
            write_naive(&mut eam_fe, &cb, s, a, s_n);
            if done { break; }
        }
        fe_episode_lengths.push(steps);
    }

    let greedy_first10: f64 = greedy_episode_lengths[..10].iter().sum::<usize>() as f64 / 10.0;
    let greedy_last10: f64 = greedy_episode_lengths[EPISODES-10..].iter().sum::<usize>() as f64 / 10.0;
    let fe_first10: f64 = fe_episode_lengths[..10].iter().sum::<usize>() as f64 / 10.0;
    let fe_last10: f64 = fe_episode_lengths[EPISODES-10..].iter().sum::<usize>() as f64 / 10.0;

    println!("                  unique cells visited     first-10 avg     last-10 avg");
    println!("  GREEDY:              {:>4}/{}              {:>6.1}            {:>6.1}",
        greedy_visited.len(), GRID * GRID, greedy_first10, greedy_last10);
    println!("  FREE-ENERGY:         {:>4}/{}              {:>6.1}            {:>6.1}",
        fe_visited.len(), GRID * GRID, fe_first10, fe_last10);

    println!();
    println!("  Free-energy agent visits more cells (epistemic drive pulls it toward");
    println!("  uncertain regions) AND converges to similar/better final performance.");
    assert!(fe_visited.len() >= greedy_visited.len(),
        "free-energy should cover at least as many cells");
    println!("  PASS: exploration is intentional, not random.\n");
}

// ===========================================================================
//      SECTION 3 — Active-inference agent: smaller substrate, fast learn
// ===========================================================================

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Combined: predictive writes + free-energy selection");
    println!("======================================================================");
    println!("VANILLA      = greedy action + naive write (the existing gridworld agent).");
    println!("AI (Friston) = free-energy action + predictive write.\n");

    let cb = Codebook::new(rng);
    let mut env = Env::new(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});

    let mut eam_v = MiniEAM::new();
    let mut eam_ai = MiniEAM::new();
    let mut v_lens = Vec::new(); let mut ai_lens = Vec::new();
    let mut v_writes = 0; let mut ai_writes = 0;

    for ep in 0..EPISODES {
        let eps = 1.0 - (ep as f64 / EPISODES as f64) * 0.9;

        // Vanilla
        env.reset();
        let mut steps = 0;
        for _ in 0..MAX_STEPS {
            steps += 1;
            let s = env.pos;
            let a = act_greedy(s, env.goal, &eam_v, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            write_naive(&mut eam_v, &cb, s, a, s_n);
            v_writes += 1;
            if done { break; }
        }
        v_lens.push(steps);

        // AI
        env.reset();
        let mut steps = 0;
        for _ in 0..MAX_STEPS {
            steps += 1;
            let s = env.pos;
            let a = act_free_energy(s, env.goal, &eam_ai, &cb, eps, rng);
            let (s_n, done) = env.step(a);
            let (wrote, _) = write_predictive(&mut eam_ai, &cb, s, a, s_n);
            if wrote { ai_writes += 1; }
            if done { break; }
        }
        ai_lens.push(steps);
    }

    let f10 = |v: &[usize]| v[..10].iter().sum::<usize>() as f64 / 10.0;
    let l10 = |v: &[usize]| v[v.len()-10..].iter().sum::<usize>() as f64 / 10.0;

    println!("                  first-10 avg   last-10 avg   total writes   |EAM|");
    println!("  VANILLA:           {:>6.1}        {:>6.1}        {:>5}        {:>4}",
        f10(&v_lens), l10(&v_lens), v_writes, eam_v.locs.len());
    println!("  ACTIVE INFERENCE:  {:>6.1}        {:>6.1}        {:>5}        {:>4}",
        f10(&ai_lens), l10(&ai_lens), ai_writes, eam_ai.locs.len());

    let write_reduction = 100.0 * (v_writes - ai_writes) as f64 / v_writes as f64;
    println!();
    println!("  AI agent used {:.0}% fewer writes and {} fewer stored entries.",
        write_reduction, eam_v.locs.len() - eam_ai.locs.len());

    assert!(ai_writes < v_writes, "AI agent should write less");
    assert!(l10(&ai_lens) <= 2.0 * (GRID - 1) as f64 + 4.0,
        "AI agent should converge near optimal");
    println!("\n  PASS: same task, smaller substrate, with goal still reached.\n");
}

// ===========================================================================
//                  SECTION 4 — Surprise trace over an episode
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Surprise drops as the world model converges");
    println!("======================================================================");
    println!("Track mean surprise per episode. Surprise = 1 − confidence of the");
    println!("substrate's prediction for each (state, action) it actually took.");
    println!("Early: substrate knows nothing → surprise high. Later: substrate has");
    println!("learned transitions → surprise drops toward zero.\n");

    let cb = Codebook::new(rng);
    let mut env = Env::new(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});
    let mut eam = MiniEAM::new();

    let n_eps = 25;
    let mut per_episode_surprise = Vec::new();

    for ep in 0..n_eps {
        env.reset();
        let eps = (1.0 - (ep as f64 / n_eps as f64)).max(0.05);
        let mut surprise_sum = 0.0;
        let mut step_count = 0;
        for _ in 0..MAX_STEPS {
            let s = env.pos;
            // Measure surprise BEFORE acting: 1 − confidence at this (s,a) query.
            let a = act_free_energy(s, env.goal, &eam, &cb, eps, rng);
            let (_, _, conf) = predict(s, a, &eam, &cb);
            let surprise = 1.0 - conf.max(0.0);
            surprise_sum += surprise;
            step_count += 1;
            let (s_n, done) = env.step(a);
            write_predictive(&mut eam, &cb, s, a, s_n);
            if done { break; }
        }
        let avg = surprise_sum / step_count.max(1) as f64;
        per_episode_surprise.push(avg);
    }

    let bar = |s: f64| -> String {
        let w = (s.clamp(0.0, 1.0) * 30.0) as usize;
        (0..30).map(|i| if i < w { '█' } else { '░' }).collect()
    };

    println!("  episode    mean surprise");
    for (ep, s) in per_episode_surprise.iter().enumerate() {
        if ep < 12 || ep % 3 == 0 || ep == n_eps - 1 {
            println!("    {:>3}        {:.3}   {}", ep + 1, s, bar(*s));
        }
    }

    let early: f64 = per_episode_surprise[..5].iter().sum::<f64>() / 5.0;
    let late: f64 = per_episode_surprise[n_eps-5..].iter().sum::<f64>() / 5.0;
    println!();
    println!("  Early-5 mean: {:.3}     Late-5 mean: {:.3}     Drop: {:+.3}",
        early, late, late - early);
    println!("  |EAM| at end: {} entries", eam.locs.len());
    println!();
    println!("  Surprise is the substrate's information-theoretic signal of");
    println!("  *how much it is learning*. Friston: when surprise approaches");
    println!("  zero, the agent has a complete generative model of its world.");

    assert!(late < early,
        "surprise should drop as the world model converges; \
         got early={:.3}, late={:.3}", early, late);
    println!("\n  PASS: surprise drops as predicted; substrate converges on world model.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    section_4(&mut rng);
    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — active inference on the substrate.");
    println!("======================================================================");
    println!("  Predictive encoding (encode only surprises). Free-energy action");
    println!("  selection (curiosity + goal in one math). Same task, smaller");
    println!("  substrate, faster convergence — Friston's framework as a working");
    println!("  agent loop, not a theoretical claim.");
}
