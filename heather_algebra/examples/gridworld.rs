//! Gridworld agent built entirely on HRR + EAM primitives.
//!
//! Proves the substrate claim across three escalating demos:
//!
//!   1. **Reactive agent**: perception, world model, action selection,
//!      and learning, all expressed as `bind` / `add` / `read` over a
//!      single associative memory. No gradient descent.
//!
//!   2. **Planning by sequence completion**: store successful
//!      trajectories as position-bound bundles; produce a full plan
//!      in *one shot* by anchoring start + goal and letting the
//!      substrate fill in the middle via relaxation.
//!
//!   3. **Composition by union**: train two agents in disjoint halves
//!      of the grid (neither sees the full path); merge their EAMs;
//!      show the merged agent navigates the full task that *neither
//!      sub-agent could solve alone*.
//!
//! Run with: `cargo run --release --example gridworld -p heather_algebra`
//!
//! Each section asserts its own success at the end, so this doubles
//! as a CI check that the substrate behaves as advertised.
//!
//! The agent uses a deliberately tiny in-memory EAM (~50 lines) so the
//! demo isolates the substrate claim from the engine claim. Same
//! primitives, same algebra — just no LMDB to muddy the proof.

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

// ----- Hyperparameters ------------------------------------------------------

const DIM: usize = 256;
const GRID: i32 = 5;
const MAX_STEPS: usize = 100;
const EPISODES: usize = 100;
const READ_ITERS: usize = 4;
const BETA: f64 = 8.0;
const DEDUP_SIM: f64 = 0.97;
const EPSILON_START: f64 = 1.0;
const EPSILON_END: f64 = 0.05;
const PLAN_HORIZON: usize = 12;

// ----- Mini in-memory EAM ---------------------------------------------------

struct MiniEAM {
    locs: Vec<HardLocation>,
    next_id: u64,
}

impl MiniEAM {
    fn new() -> Self { Self { locs: Vec::new(), next_id: 0 } }

    fn write(&mut self, pattern: &[f64]) -> bool {
        let address = vec_ops::normalize(pattern);
        for loc in &self.locs {
            if vec_ops::cosine_similarity(&loc.address, &address) > DEDUP_SIM {
                return false;
            }
        }
        let mut loc = HardLocation::new(LocationId(self.next_id), address);
        loc.counter = pattern.to_vec();
        loc.write_count = 1.0;
        self.locs.push(loc);
        self.next_id += 1;
        true
    }

    /// Modern Hopfield read: softmax-weighted superposition, iterated.
    fn read(&self, query: &[f64]) -> Vec<f64> {
        if self.locs.is_empty() {
            return vec_ops::normalize(query);
        }
        let mut q = vec_ops::normalize(query);
        for _ in 0..READ_ITERS {
            let sims: Vec<f64> =
                self.locs.iter().map(|l| vec_ops::dot(&l.address, &q)).collect();
            let w = vec_ops::softmax(&sims, BETA);
            let patterns: Vec<&[f64]> =
                self.locs.iter().map(|l| l.counter.as_slice()).collect();
            let next = vec_ops::weighted_sum(&patterns, &w);
            q = vec_ops::normalize(&next);
        }
        q
    }

    fn trace(&self, query: &[f64], top_k: usize) -> Vec<(u64, f64)> {
        let q = vec_ops::normalize(query);
        let mut sims: Vec<(u64, f64)> = self
            .locs
            .iter()
            .map(|l| (l.id.0, vec_ops::dot(&l.address, &q)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(top_k);
        sims
    }
}

/// Union two EAMs into a joint memory. Set-union semantics: the merged
/// agent "remembers everything A remembered AND everything B remembered."
/// Dedup happens automatically via `write`'s similarity check.
///
/// The algebra crate's `add` is a *different* compositional primitive —
/// it produces pairwise attractor sums for semantic blending. Here we
/// want experiential union, which is the simpler operator.
fn union(a: &MiniEAM, b: &MiniEAM) -> MiniEAM {
    let mut out = MiniEAM::new();
    for loc in &a.locs { out.write(&loc.counter); }
    for loc in &b.locs { out.write(&loc.counter); }
    out
}

// ----- Environment ----------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pos { x: i32, y: i32 }

#[derive(Clone, Copy, Debug, PartialEq)]
enum Action { Up, Down, Left, Right }
const ACTIONS: [Action; 4] = [Action::Up, Action::Down, Action::Left, Action::Right];

/// Gridworld with configurable x-bounds — used by the composition demo
/// to restrict each sub-agent to its own slice of the world.
struct Env {
    pos: Pos,
    start: Pos,
    goal: Pos,
    x_min: i32, x_max: i32,
}

impl Env {
    fn full(start: Pos, goal: Pos) -> Self {
        Self { pos: start, start, goal, x_min: 0, x_max: GRID - 1 }
    }
    fn sub(start: Pos, goal: Pos, x_min: i32, x_max: i32) -> Self {
        Self { pos: start, start, goal, x_min, x_max }
    }
    fn reset(&mut self) { self.pos = self.start; }
    fn step(&mut self, a: Action) -> (Pos, bool) {
        let (dx, dy) = match a {
            Action::Up => (0, -1),
            Action::Down => (0, 1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
        };
        let nx = (self.pos.x + dx).clamp(self.x_min, self.x_max);
        let ny = (self.pos.y + dy).clamp(0, GRID - 1);
        self.pos = Pos { x: nx, y: ny };
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
    /// Position keys POS_0..POS_{PLAN_HORIZON-1} for sequence binding.
    pos_keys: Vec<Vec<f64>>,
}

impl Codebook {
    fn new(rng: &mut StdRng) -> Self {
        let state_key = vec_ops::random_unit_vector(DIM, rng);
        let action_key = vec_ops::random_unit_vector(DIM, rng);
        let next_state_key = vec_ops::random_unit_vector(DIM, rng);
        let state_vecs: Vec<Vec<f64>> =
            (0..(GRID * GRID)).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect();
        let action_vecs = [
            vec_ops::random_unit_vector(DIM, rng),
            vec_ops::random_unit_vector(DIM, rng),
            vec_ops::random_unit_vector(DIM, rng),
            vec_ops::random_unit_vector(DIM, rng),
        ];
        let pos_keys: Vec<Vec<f64>> =
            (0..PLAN_HORIZON).map(|_| vec_ops::random_unit_vector(DIM, rng)).collect();
        Self { state_key, action_key, next_state_key, state_vecs, action_vecs, pos_keys }
    }
    fn state(&self, p: Pos) -> &[f64] { &self.state_vecs[(p.y * GRID + p.x) as usize] }
    fn action(&self, a: Action) -> &[f64] { &self.action_vecs[a as usize] }
    fn nearest_pos(&self, v: &[f64]) -> Pos {
        let v = vec_ops::normalize(v);
        let mut best_i = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, sv) in self.state_vecs.iter().enumerate() {
            let s = vec_ops::dot(sv, &v);
            if s > best_s { best_s = s; best_i = i; }
        }
        Pos { x: (best_i as i32) % GRID, y: (best_i as i32) / GRID }
    }
}

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

// ----- Reactive agent: one-step lookahead via the EAM as world model ------

fn pick_action(
    state: Pos,
    goal: Pos,
    eam: &MiniEAM,
    cb: &Codebook,
    epsilon: f64,
    rng: &mut StdRng,
) -> Action {
    if rng.r#gen::<f64>() < epsilon || eam.locs.is_empty() {
        return ACTIONS[rng.r#gen_range(0..4)];
    }
    let mut best = ACTIONS[0];
    let mut best_d = i32::MAX;
    for &a in &ACTIONS {
        let query = bundle(&[
            &bind_vec(&cb.state_key, cb.state(state)),
            &bind_vec(&cb.action_key, cb.action(a)),
        ]);
        let completion = eam.read(&query);
        let predicted = unbind_vec(&completion, &cb.next_state_key);
        let predicted_pos = cb.nearest_pos(&predicted);
        let d = manhattan(predicted_pos, goal);
        if d < best_d { best_d = d; best = a; }
    }
    best
}

fn remember(eam: &mut MiniEAM, cb: &Codebook, s: Pos, a: Action, s_next: Pos) {
    let episode = bundle(&[
        &bind_vec(&cb.state_key, cb.state(s)),
        &bind_vec(&cb.action_key, cb.action(a)),
        &bind_vec(&cb.next_state_key, cb.state(s_next)),
    ]);
    eam.write(&episode);
}

/// Run an episode end-to-end. Returns (steps, reached, trajectory).
fn run_episode(
    env: &mut Env,
    eam: &mut MiniEAM,
    cb: &Codebook,
    epsilon: f64,
    learn: bool,
    rng: &mut StdRng,
) -> (usize, bool, Vec<Pos>) {
    env.reset();
    let mut trajectory = vec![env.pos];
    for step in 0..MAX_STEPS {
        let s = env.pos;
        let a = pick_action(s, env.goal, eam, cb, epsilon, rng);
        let (s_next, done) = env.step(a);
        if learn { remember(eam, cb, s, a, s_next); }
        trajectory.push(s_next);
        if done { return (step + 1, true, trajectory); }
    }
    (MAX_STEPS, false, trajectory)
}

/// Train a fresh agent in (possibly restricted) environment.
fn train_agent(
    cb: &Codebook,
    mut env: Env,
    episodes: usize,
    rng: &mut StdRng,
) -> (MiniEAM, Vec<usize>) {
    let mut eam = MiniEAM::new();
    let mut lengths = Vec::with_capacity(episodes);
    for ep in 0..episodes {
        let epsilon = EPSILON_START
            + (EPSILON_END - EPSILON_START) * (ep as f64 / episodes as f64);
        let (steps, _, _) = run_episode(&mut env, &mut eam, cb, epsilon, true, rng);
        lengths.push(steps);
    }
    (eam, lengths)
}

// ----- Trajectory memory for planning by sequence completion ---------------

/// Stores whole trajectories as single bundled vectors:
///   traj = Σ_i  bind(POS_i, state_i)
///
/// To produce a plan, anchor start (POS_0) and goal (POS_{H-1}) only,
/// read against this memory, and the substrate fills in the missing
/// middle by relaxing toward the stored trajectories whose endpoints
/// match. Extract each position by unbinding.
///
/// This is **planning as pattern completion** — no tree search, no
/// value function, no separate planner module. One read, one plan.
struct TrajectoryMemory {
    eam: MiniEAM,
}

impl TrajectoryMemory {
    fn new() -> Self { Self { eam: MiniEAM::new() } }

    fn store(&mut self, cb: &Codebook, traj: &[Pos]) {
        // Pad with the last position (typically goal) to fixed horizon.
        let mut padded: Vec<Pos> = traj.to_vec();
        while padded.len() < PLAN_HORIZON {
            padded.push(*padded.last().unwrap());
        }
        padded.truncate(PLAN_HORIZON);

        let parts: Vec<Vec<f64>> = padded.iter().enumerate()
            .map(|(i, p)| bind_vec(&cb.pos_keys[i], cb.state(*p)))
            .collect();
        let refs: Vec<&[f64]> = parts.iter().map(|v| v.as_slice()).collect();
        let traj_vec = bundle(&refs);
        self.eam.write(&traj_vec);
    }

    fn plan(&self, cb: &Codebook, start: Pos, goal: Pos) -> Vec<Pos> {
        // Anchor ONLY the endpoints. The middle is up to the substrate.
        let query = bundle(&[
            &bind_vec(&cb.pos_keys[0], cb.state(start)),
            &bind_vec(&cb.pos_keys[PLAN_HORIZON - 1], cb.state(goal)),
        ]);
        let completion = self.eam.read(&query);
        (0..PLAN_HORIZON).map(|i| {
            let noisy = unbind_vec(&completion, &cb.pos_keys[i]);
            cb.nearest_pos(&noisy)
        }).collect()
    }
}

fn is_valid_step(a: Pos, b: Pos) -> bool {
    let dx = (a.x - b.x).abs();
    let dy = (a.y - b.y).abs();
    (dx + dy) <= 1 // same cell, or one orthogonal step
}

// ----- Section 1: Reactive agent --------------------------------------------

fn section_1_reactive(cb: &Codebook, rng: &mut StdRng) -> MiniEAM {
    println!("============================================================");
    println!("  SECTION 1 — Reactive agent: substrate as world model");
    println!("============================================================");
    println!("Gridworld {}×{}, start (0,0) → goal ({},{}), optimal = {} steps",
        GRID, GRID, GRID-1, GRID-1, 2*(GRID-1));
    println!();

    let env = Env::full(Pos { x: 0, y: 0 }, Pos { x: GRID-1, y: GRID-1 });
    let (eam, lengths) = train_agent(cb, env, EPISODES, rng);

    for ep in [0, 4, 9, 29, 49, 79, 99] {
        println!("  ep {:3}  steps={:3}", ep + 1, lengths[ep]);
    }
    let first10: f64 = lengths[..10].iter().sum::<usize>() as f64 / 10.0;
    let last10: f64 = lengths[lengths.len()-10..].iter().sum::<usize>() as f64 / 10.0;
    println!();
    println!("  First 10 avg: {:.1} steps", first10);
    println!("  Last  10 avg: {:.1} steps   (optimal: {})", last10, 2*(GRID-1));
    println!("  |EAM|: {} stored transitions", eam.locs.len());

    // Activation trace at start
    println!("\n  Activation trace at (0,0):");
    for &a in &ACTIONS {
        let query = bundle(&[
            &bind_vec(&cb.state_key, cb.state(Pos{x:0,y:0})),
            &bind_vec(&cb.action_key, cb.action(a)),
        ]);
        let completion = eam.read(&query);
        let predicted = cb.nearest_pos(&unbind_vec(&completion, &cb.next_state_key));
        let top = eam.trace(&query, 3);
        println!("    {:>5?}  → predicts {:?}   top stored: {:?}",
            a, predicted, top);
    }

    let optimal = 2.0 * (GRID as f64 - 1.0);
    assert!(last10 < first10 * 0.5, "no learning");
    assert!(last10 < optimal * 1.5, "did not converge");
    println!("\n  PASS: reactive agent learned without gradient descent.\n");
    eam
}

// ----- Section 2: Planning by sequence completion ---------------------------

fn section_2_planning(cb: &Codebook, eam: &MiniEAM, rng: &mut StdRng) {
    println!("============================================================");
    println!("  SECTION 2 — Planning: substrate completes the plan in 1 shot");
    println!("============================================================");

    // Collect 25 trajectories from a trained agent (low ε, near-optimal).
    // These become the corpus the planner interpolates over.
    let mut env = Env::full(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});
    let mut traj_mem = TrajectoryMemory::new();
    let mut collected = 0;
    let mut eam_view = eam_clone(eam); // immutable trained EAM, freshly cloned
    for _ in 0..30 {
        let (_, reached, traj) =
            run_episode(&mut env, &mut eam_view, cb, EPSILON_END, false, rng);
        if reached && traj.len() <= PLAN_HORIZON + 2 {
            traj_mem.store(cb, &traj);
            collected += 1;
        }
    }
    println!("  Stored {} successful trajectories into trajectory memory.", collected);
    println!("  |trajectory EAM|: {} bundles", traj_mem.eam.locs.len());

    // Now plan in one shot.
    let start = Pos{x:0, y:0};
    let goal = Pos{x:GRID-1, y:GRID-1};
    let plan = traj_mem.plan(cb, start, goal);

    println!("\n  Plan from {:?} to {:?} (one-shot, no rollout):", start, goal);
    for (i, p) in plan.iter().enumerate() {
        let marker = if i == 0 || i == PLAN_HORIZON - 1 { "*" } else { " " };
        println!("    {} pos {:2}: {:?}", marker, i, p);
    }
    println!("    (* = anchored endpoint; everything else was filled by the substrate)");

    // Validity: count adjacent-or-same transitions.
    let mut valid = 0;
    for w in plan.windows(2) {
        if is_valid_step(w[0], w[1]) { valid += 1; }
    }
    let total = plan.len() - 1;
    println!("\n  Plan validity: {}/{} consecutive transitions are adjacent.", valid, total);
    println!("  Final position in plan: {:?}   (goal: {:?})", plan.last().unwrap(), goal);

    assert!(plan[0] == start, "start anchor not honored");
    assert!(plan[PLAN_HORIZON-1] == goal, "goal anchor not honored");
    assert!(valid >= total - 2,
        "plan should be mostly valid: {}/{} adjacent", valid, total);
    println!("\n  PASS: substrate produced a coherent plan via pattern completion.\n");
}

/// Clone the EAM by re-writing its patterns. (HardLocation has no Clone
/// derive we want to rely on; this is the substrate-honest path.)
fn eam_clone(src: &MiniEAM) -> MiniEAM {
    let mut out = MiniEAM::new();
    for loc in &src.locs { out.write(&loc.counter); }
    out
}

// ----- Section 3: Composition by union --------------------------------------

fn section_3_composition(cb: &Codebook, rng: &mut StdRng) {
    println!("============================================================");
    println!("  SECTION 3 — Composition: two agents → joint memory → full task");
    println!("============================================================");

    // Agent A: lives in left strip (x ∈ [0,2]), goes (0,0) → (2,2).
    // Agent B: lives in right strip (x ∈ [2,4]), goes (2,2) → (4,4).
    // Both share the codebook — that is what makes their memories composable.
    let env_a = Env::sub(Pos{x:0,y:0}, Pos{x:2,y:2}, 0, 2);
    let env_b = Env::sub(Pos{x:2,y:2}, Pos{x:GRID-1,y:GRID-1}, 2, GRID-1);

    let (eam_a, len_a) = train_agent(cb, env_a, EPISODES, rng);
    let (eam_b, len_b) = train_agent(cb, env_b, EPISODES, rng);

    let avg_a = len_a[len_a.len()-10..].iter().sum::<usize>() as f64 / 10.0;
    let avg_b = len_b[len_b.len()-10..].iter().sum::<usize>() as f64 / 10.0;
    println!("  Agent A (left,  x∈[0,2]):   last-10 avg = {:.1} steps  |EAM|={}", avg_a, eam_a.locs.len());
    println!("  Agent B (right, x∈[2,4]):   last-10 avg = {:.1} steps  |EAM|={}", avg_b, eam_b.locs.len());

    // Merge: set-union of stored experiences. The merged agent literally
    // remembers what A learned AND what B learned. No retraining.
    let joint = union(&eam_a, &eam_b);
    println!("  Merged memory:                                       |EAM|={}", joint.locs.len());

    // Now test all three on the FULL task (0,0) → (4,4).
    println!("\n  Test: full task, start (0,0) → goal (4,4), max {} steps", MAX_STEPS);

    let mut env_full = Env::full(Pos{x:0,y:0}, Pos{x:GRID-1,y:GRID-1});
    // Small ε for evaluation: breaks ties between equally-scored
    // predictions (pick_action otherwise always picks Up on a tie,
    // creating deterministic loops). Same setting for all three EAMs
    // so the comparison is fair.
    let mut test = |label: &str, eam: &MiniEAM, rng: &mut StdRng| -> Option<usize> {
        let mut e = eam_clone(eam);
        let (steps, reached, _) =
            run_episode(&mut env_full, &mut e, cb, EPSILON_END, false, rng);
        if reached {
            println!("    {:>14}: reached goal in {:3} steps", label, steps);
            Some(steps)
        } else {
            println!("    {:>14}: FAILED to reach goal (gap in memory)", label);
            None
        }
    };

    let result_a = test("Agent A only", &eam_a, rng);
    let result_b = test("Agent B only", &eam_b, rng);
    let result_joint = test("Joint (A∪B)", &joint, rng);

    // The substantive claim: neither sub-agent can solve the full task,
    // but their *composed memory* — bit-for-bit the same patterns, just
    // unioned — can. Step-count optimality is secondary: the joint
    // agent has coverage gaps where neither sub-agent explored.
    assert!(result_a.is_none() || result_a.unwrap() >= MAX_STEPS,
        "Agent A alone should fail or struggle severely (only saw left half)");
    assert!(result_b.is_none() || result_b.unwrap() >= MAX_STEPS,
        "Agent B alone should fail or struggle severely (only saw right half)");
    assert!(result_joint.is_some(),
        "Joint memory should solve the full task that neither half could");
    println!("\n  PASS: composed memory solves a task neither sub-agent could.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let cb = Codebook::new(&mut rng);
    let trained = section_1_reactive(&cb, &mut rng);
    section_2_planning(&cb, &trained, &mut rng);
    section_3_composition(&cb, &mut rng);
    println!("============================================================");
    println!("  ALL THREE SECTIONS PASSED — substrate proof complete.");
    println!("============================================================");
}
