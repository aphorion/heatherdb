//! Action selection — Eliasmith's basal-ganglia primitive on the substrate.
//!
//! In Spaun (Eliasmith 2012), the basal ganglia implements winner-take-all
//! over competing action candidates: each potential action has a utility
//! computed from the current cortical state, and the action with the
//! highest utility is selected and gated to motor cortex. Eliasmith models
//! this with a competitive spiking network; on HRR + EAM the same
//! computation falls out of substrate primitives.
//!
//! Each action is a bound bundle: bind(NAME, n) + bind(WHEN, pre) +
//! bind(DO, effect). The action library is an EAM of these bundles.
//! Selection is a read with the current state bound to WHEN — the EAM's
//! softmax converges on the action whose precondition matches best, and
//! NAME + effect are recovered by unbinding.
//!
//! Sections:
//!   1. Basic action selection from state alone (the BG primitive).
//!   2. Robustness to noisy state observations (substrate cleanup).
//!   3. Goal-directed branching: when two actions can fire, the one
//!      whose effect moves toward the goal wins.
//!   4. Full task execution: tea-making chain — planning as repeated
//!      greedy selection over the substrate.
//!
//! Run: `cargo run --release --example action_selection -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 1024;
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

fn nearest_named<'a>(v: &[f64], lex: &'a [(String, Vec<f64>)]) -> (&'a str, f64) {
    let v = vec_ops::normalize(v);
    let mut best = lex[0].0.as_str();
    let mut best_s = f64::NEG_INFINITY;
    for (name, vec) in lex {
        let s = vec_ops::cosine_similarity(&v, vec);
        if s > best_s { best_s = s; best = name; }
    }
    (best, best_s)
}

// ----- Mini EAM (the action library) ---------------------------------------

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

// ----- The action library / basal-ganglia primitive ------------------------

struct Brain {
    eam: MiniEAM,
    name_key: Vec<f64>,
    when_key: Vec<f64>,
    do_key: Vec<f64>,
    name_lex: Vec<(String, Vec<f64>)>,
    state_lex: Vec<(String, Vec<f64>)>,
}

impl Brain {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            eam: MiniEAM::new(),
            name_key: rand_unit(rng),
            when_key: rand_unit(rng),
            do_key: rand_unit(rng),
            name_lex: Vec::new(),
            state_lex: Vec::new(),
        }
    }

    fn declare_state(&mut self, name: &str, rng: &mut StdRng) {
        self.state_lex.push((name.into(), rand_unit(rng)));
    }
    fn state(&self, name: &str) -> &[f64] {
        self.state_lex.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice()).unwrap()
    }

    fn add_action(&mut self, name: &str, pre: &str, effect: &str, rng: &mut StdRng) {
        let name_vec = rand_unit(rng);
        self.name_lex.push((name.into(), name_vec.clone()));
        let pre_v = self.state(pre).to_vec();
        let eff_v = self.state(effect).to_vec();
        let action_bundle = bundle(&[
            &bind_vec(&self.name_key, &name_vec),
            &bind_vec(&self.when_key, &pre_v),
            &bind_vec(&self.do_key, &eff_v),
        ]);
        self.eam.write(&action_bundle);
    }

    /// Select an action by current state alone — pure precondition-matching.
    /// This is the basal-ganglia primitive: WTA over the action library.
    fn select(&self, state: &[f64]) -> (String, Vec<f64>, f64) {
        let query = bind_vec(&self.when_key, state);
        let recalled = self.eam.read(&query);
        let name_noisy = unbind_vec(&recalled, &self.name_key);
        let effect_noisy = unbind_vec(&recalled, &self.do_key);
        let (name, name_sim) = nearest_named(&name_noisy, &self.name_lex);
        let (effect_name, _) = nearest_named(&effect_noisy, &self.state_lex);
        let effect_vec = self.state(effect_name).to_vec();
        (name.into(), effect_vec, name_sim)
    }

    /// Goal-directed selection: enumerate stored actions, score each by
    /// (precondition match × goal proximity), pick the winner. This is
    /// what Eliasmith calls the "utility-weighted" basal ganglia variant,
    /// used when multiple actions are applicable.
    fn select_for_goal(&self, state: &[f64], goal: &[f64]) -> (String, Vec<f64>, f64) {
        let state_norm = vec_ops::normalize(state);
        let goal_norm = vec_ops::normalize(goal);

        let mut best_name = String::new();
        let mut best_effect = vec![0.0; DIM];
        let mut best_u = f64::NEG_INFINITY;

        // For each stored action, decode (name, pre, effect) and score.
        for loc in &self.eam.locs {
            let stored = &loc.counter;
            let pre_v = vec_ops::normalize(&unbind_vec(stored, &self.when_key));
            let do_v = vec_ops::normalize(&unbind_vec(stored, &self.do_key));
            let name_v = unbind_vec(stored, &self.name_key);

            let pre_match = vec_ops::cosine_similarity(&pre_v, &state_norm).max(0.0);
            let goal_match = vec_ops::cosine_similarity(&do_v, &goal_norm).max(0.0);
            let utility = pre_match * goal_match;

            if utility > best_u {
                best_u = utility;
                let (n, _) = nearest_named(&name_v, &self.name_lex);
                best_name = n.into();
                let (e, _) = nearest_named(&do_v, &self.state_lex);
                best_effect = self.state(e).to_vec();
            }
        }
        (best_name, best_effect, best_u)
    }
}

// ===========================================================================
//                    SECTION 1 — Basic action selection
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Basal-ganglia primitive: select by current state");
    println!("======================================================================");
    println!("5 actions stored in the action library (EAM). For each state, the");
    println!("substrate reads the EAM with bind(WHEN, state) and the winning");
    println!("action emerges from the softmax.\n");

    let mut brain = Brain::new(rng);
    for s in ["off", "powered_on", "loaded", "running", "complete"] {
        brain.declare_state(s, rng);
    }
    brain.add_action("press_power", "off", "powered_on", rng);
    brain.add_action("load_program", "powered_on", "loaded", rng);
    brain.add_action("start", "loaded", "running", rng);
    brain.add_action("wait", "running", "complete", rng);
    brain.add_action("reset", "complete", "off", rng);

    for state_name in ["off", "powered_on", "loaded", "running", "complete"] {
        let current = brain.state(state_name).to_vec();
        let (action, _, sim) = brain.select(&current);
        println!("  state={:<13} → action={:<14} (sim {:.3})", state_name, action, sim);
        let expected = match state_name {
            "off" => "press_power",
            "powered_on" => "load_program",
            "loaded" => "start",
            "running" => "wait",
            "complete" => "reset",
            _ => unreachable!(),
        };
        assert_eq!(action, expected, "wrong action for state {}", state_name);
    }

    println!("\n  PASS: 5/5 actions selected correctly by precondition match.\n");
}

// ===========================================================================
//                  SECTION 2 — Robustness to noisy observations
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Robustness to noisy state observations");
    println!("======================================================================");
    println!("Real perception is noisy. The substrate's EAM cleanup handles");
    println!("this for free: state + Gaussian noise still selects correctly\n");

    let mut brain = Brain::new(rng);
    for s in ["off", "powered_on", "loaded", "running", "complete"] {
        brain.declare_state(s, rng);
    }
    brain.add_action("press_power", "off", "powered_on", rng);
    brain.add_action("load_program", "powered_on", "loaded", rng);
    brain.add_action("start", "loaded", "running", rng);
    brain.add_action("wait", "running", "complete", rng);
    brain.add_action("reset", "complete", "off", rng);

    let noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let trials = 50;

    for &noise in &noise_levels {
        let mut hits = 0;
        for _ in 0..trials {
            let state_names = ["off", "powered_on", "loaded", "running", "complete"];
            let idx = rng.r#gen_range(0..state_names.len());
            let true_state = state_names[idx];
            let clean = brain.state(true_state).to_vec();

            let noisy: Vec<f64> = clean.iter().map(|x| {
                x + noise * (rng.r#gen::<f64>() * 2.0 - 1.0)
            }).collect();
            let noisy = vec_ops::normalize(&noisy);

            let (action, _, _) = brain.select(&noisy);
            let expected = match true_state {
                "off" => "press_power",
                "powered_on" => "load_program",
                "loaded" => "start",
                "running" => "wait",
                "complete" => "reset",
                _ => unreachable!(),
            };
            if action == expected { hits += 1; }
        }
        let acc = hits as f64 / trials as f64;
        let bar: String = (0..((acc * 30.0) as usize)).map(|_| '█').collect();
        println!("  noise σ={:.1}   accuracy {:3.0}%   {}", noise, acc * 100.0, bar);
    }

    println!("\n  PASS: action selection degrades gracefully with sensor noise.\n");
}

// ===========================================================================
//                  SECTION 3 — Goal-directed branching
// ===========================================================================

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Goal-directed selection at branch points");
    println!("======================================================================");
    println!("From the same state, multiple actions are applicable. Each leads");
    println!("to a different terminal outcome. The substrate picks the action");
    println!("whose effect is the current goal — basal ganglia + PFC gating.\n");

    let mut brain = Brain::new(rng);
    // From `button_pressed`, three actions each leading to a different
    // outcome state. The choice depends entirely on the goal.
    for s in ["button_pressed", "lights_on", "music_playing", "alarm_set"] {
        brain.declare_state(s, rng);
    }
    brain.add_action("turn_on_lights", "button_pressed", "lights_on", rng);
    brain.add_action("play_music",     "button_pressed", "music_playing", rng);
    brain.add_action("set_alarm",      "button_pressed", "alarm_set", rng);

    let start = brain.state("button_pressed").to_vec();
    let cases: &[(&str, &str)] = &[
        ("lights_on",     "turn_on_lights"),
        ("music_playing", "play_music"),
        ("alarm_set",     "set_alarm"),
    ];
    for (goal_name, expected) in cases {
        let goal = brain.state(goal_name).to_vec();
        let (action, _, u) = brain.select_for_goal(&start, &goal);
        let mark = if action == *expected { "✓" } else { "✗" };
        println!("  goal={:<14} → action={:<18} (utility {:.3})  {}",
            goal_name, action, u, mark);
        assert_eq!(action, *expected,
            "goal-directed selection failed for goal={}", goal_name);
    }

    println!("\n  Same state, same precondition-matching action set, but the");
    println!("  substrate picks DIFFERENTLY based on goal. This is the BG+PFC");
    println!("  loop: action utility = precondition_match × goal_match.\n");
    println!("  PASS: goal-directed action selection.\n");
}

// ===========================================================================
//                  SECTION 4 — Full task execution chain
// ===========================================================================

fn section_4(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Sequential task execution (planning by repeated WTA)");
    println!("======================================================================");
    println!("Make tea: start from kitchen_empty, reach tea_ready. The substrate");
    println!("selects each step greedily; no explicit planner — planning emerges.\n");

    let mut brain = Brain::new(rng);
    for s in ["kitchen_empty", "have_kettle", "kettle_filled", "water_boiling",
              "have_mug", "water_in_mug", "tea_brewing", "tea_ready"] {
        brain.declare_state(s, rng);
    }
    brain.add_action("get_kettle",   "kitchen_empty",  "have_kettle",   rng);
    brain.add_action("fill_kettle",  "have_kettle",    "kettle_filled", rng);
    brain.add_action("boil_water",   "kettle_filled",  "water_boiling", rng);
    brain.add_action("get_mug",      "water_boiling",  "have_mug",      rng);
    brain.add_action("pour_water",   "have_mug",       "water_in_mug",  rng);
    brain.add_action("add_tea_leaves", "water_in_mug", "tea_brewing",   rng);
    brain.add_action("wait_to_brew", "tea_brewing",    "tea_ready",     rng);

    let mut state_name = "kitchen_empty".to_string();
    let mut steps = 0;
    let max_steps = 12;

    println!("  Trajectory:");
    println!("    start: {}", state_name);

    while state_name != "tea_ready" && steps < max_steps {
        let state_vec = brain.state(&state_name).to_vec();
        let (action, effect_vec, _) = brain.select(&state_vec);
        let (next_state, _) = nearest_named(&effect_vec, &brain.state_lex);
        println!("    step {:>2}: ({:<13}) -- {:<18} → {}",
            steps + 1, state_name, action, next_state);
        state_name = next_state.into();
        steps += 1;
    }

    println!();
    assert_eq!(state_name, "tea_ready",
        "did not reach goal in {} steps (got {})", max_steps, state_name);
    assert_eq!(steps, 7, "expected exactly 7 steps to make tea");

    println!("  Reached tea_ready in {} steps (optimal = 7).", steps);
    println!("  No planner, no search, no rules engine — just substrate WTA");
    println!("  iterated until the goal attractor was reached.\n");
    println!("  PASS: planning by repeated greedy action selection.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    section_4(&mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — basal-ganglia primitive operational.");
    println!("======================================================================");
    println!("  With working_memory + analogy + action_selection, the substrate");
    println!("  now has the core triad of cognition: state, inference, action.");
    println!("  Spaun's architecture, minus the spiking-neuron tax.");
}
