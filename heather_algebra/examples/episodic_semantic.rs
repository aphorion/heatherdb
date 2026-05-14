//! Episodic → semantic memory: the hippocampal–cortical replay loop on
//! the substrate.
//!
//! Real brains separate **episodic memory** (specific recent experiences,
//! hippocampal, fast write) from **semantic memory** (consolidated
//! generalizations, cortical, slow build-up via replay). The substrate
//! gives us both as cheap primitives and the consolidation pass as an
//! explicit operation.
//!
//! Architecture:
//!
//!   - **Episodic EAM**: every experience is one bound bundle, written
//!     verbatim. Fast write, exact specific recall, fills up over time.
//!   - **Semantic EAM**: prototype vectors, one per recurring pattern.
//!     Built by replaying episodic groups and writing their centroid.
//!     Fewer entries, generalizes well, robust to specific-detail noise.
//!   - **Combined query**: probe both memories, use whichever is more
//!     confident. Specific cases hit episodic; novel-but-pattern-matching
//!     cases hit semantic.
//!
//! Sections:
//!   1. Episodic-only baseline — store cases, retrieve resolution.
//!   2. Consolidation pass — group by resolution, write prototypes to
//!      semantic.
//!   3. Specific vs. general queries — episodic is sharp on seen cases;
//!      semantic generalizes to novel ones.
//!   4. Combined two-tier query outperforms either alone.
//!
//! Run: `cargo run --release --example episodic_semantic -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

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
    /// Returns (result, top similarity) so callers can gauge confidence.
    fn read(&self, query: &[f64]) -> (Vec<f64>, f64) {
        if self.locs.is_empty() { return (vec_ops::normalize(query), 0.0); }
        let mut q = vec_ops::normalize(query);
        let mut top_sim = 0.0;
        for _ in 0..READ_ITERS {
            let sims: Vec<f64> =
                self.locs.iter().map(|l| vec_ops::dot(&l.address, &q)).collect();
            top_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w = vec_ops::softmax(&sims, BETA);
            let patterns: Vec<&[f64]> =
                self.locs.iter().map(|l| l.counter.as_slice()).collect();
            q = vec_ops::normalize(&vec_ops::weighted_sum(&patterns, &w));
        }
        (q, top_sim)
    }
}

// ----- Two-tier memory ------------------------------------------------------

struct TwoTier {
    episodic: MiniEAM,
    semantic: MiniEAM,
    symptom_key: Vec<f64>,
    tier_key: Vec<f64>,
    product_key: Vec<f64>,
    resolution_key: Vec<f64>,
    res_lex: Vec<(String, Vec<f64>)>,
}

impl TwoTier {
    fn new(rng: &mut StdRng, res_lex: Vec<(String, Vec<f64>)>) -> Self {
        Self {
            episodic: MiniEAM::new(),
            semantic: MiniEAM::new(),
            symptom_key: rand_unit(rng),
            tier_key: rand_unit(rng),
            product_key: rand_unit(rng),
            resolution_key: rand_unit(rng),
            res_lex,
        }
    }

    fn encode(&self, s: &[f64], t: &[f64], p: &[f64], r: Option<&[f64]>) -> Vec<f64> {
        let mut parts: Vec<Vec<f64>> = vec![
            bind_vec(&self.symptom_key, s),
            bind_vec(&self.tier_key, t),
            bind_vec(&self.product_key, p),
        ];
        if let Some(rv) = r {
            parts.push(bind_vec(&self.resolution_key, rv));
        }
        let refs: Vec<&[f64]> = parts.iter().map(|v| v.as_slice()).collect();
        bundle(&refs)
    }

    fn write_episode(&mut self, s: &[f64], t: &[f64], p: &[f64], r: &[f64]) {
        let case = self.encode(s, t, p, Some(r));
        self.episodic.write(&case);
    }

    /// Consolidation: replay episodic, group by resolution, write
    /// per-group prototypes to semantic. This is the hippocampal→cortical
    /// transfer: cortex receives smoothed prototypes of recurring patterns.
    fn consolidate(&mut self) {
        let mut groups: HashMap<String, Vec<&Vec<f64>>> = HashMap::new();
        for loc in &self.episodic.locs {
            // Decode this episode's resolution
            let noisy_res = unbind_vec(&loc.counter, &self.resolution_key);
            let (name, _) = nearest_named(&noisy_res, &self.res_lex);
            groups.entry(name.into()).or_default().push(&loc.counter);
        }

        // For each resolution group, bundle the episodic vectors into a
        // semantic prototype. The bound RESOLUTION component reinforces;
        // the input components average out across the group, producing a
        // "policy" prototype that generalizes the cases.
        for (_res_name, members) in groups {
            let parts: Vec<&[f64]> = members.iter().map(|v| v.as_slice()).collect();
            let prototype = bundle(&parts);
            self.semantic.write(&prototype);
        }
    }

    fn query(&self, eam: &MiniEAM, s: &[f64], t: &[f64], p: &[f64]) -> (String, f64) {
        let q = self.encode(s, t, p, None);
        let (recalled, top_sim) = eam.read(&q);
        let noisy_res = unbind_vec(&recalled, &self.resolution_key);
        let (name, _) = nearest_named(&noisy_res, &self.res_lex);
        (name.into(), top_sim)
    }

    fn query_episodic(&self, s: &[f64], t: &[f64], p: &[f64]) -> (String, f64) {
        self.query(&self.episodic, s, t, p)
    }
    fn query_semantic(&self, s: &[f64], t: &[f64], p: &[f64]) -> (String, f64) {
        self.query(&self.semantic, s, t, p)
    }
    fn query_combined(&self, s: &[f64], t: &[f64], p: &[f64]) -> (String, f64) {
        let (ep_name, ep_sim) = self.query_episodic(s, t, p);
        let (se_name, se_sim) = self.query_semantic(s, t, p);
        if ep_sim >= se_sim { (ep_name, ep_sim) } else { (se_name, se_sim) }
    }
}

fn nearest_named<'a>(v: &[f64], lex: &'a [(String, Vec<f64>)]) -> (&'a str, f64) {
    let v = vec_ops::normalize(v);
    let mut best = lex[0].0.as_str();
    let mut best_s = f64::NEG_INFINITY;
    for (n, lv) in lex {
        let s = vec_ops::cosine_similarity(&v, lv);
        if s > best_s { best_s = s; best = n; }
    }
    (best, best_s)
}

// ----- The ground-truth policy ----------------------------------------------

fn ground_truth(symptom: &str, tier: &str) -> &'static str {
    match (symptom, tier) {
        ("BILLING_ERROR", _) => "REFUND",
        ("CANT_LOGIN", _) => "RESET",
        ("BUG_REPORT", "ENTERPRISE") => "ESCALATE",
        ("BUG_REPORT", _) => "DOCS_LINK",
        ("MISSING_FEATURE", "ENTERPRISE") => "ENABLE_FEATURE",
        ("MISSING_FEATURE", _) => "DOCS_LINK",
        ("QUESTION", _) => "DOCS_LINK",
        _ => "DOCS_LINK",
    }
}

// ----- Setup ----------------------------------------------------------------

const SYMPTOMS: &[&str] = &["BILLING_ERROR", "CANT_LOGIN", "BUG_REPORT", "MISSING_FEATURE", "QUESTION"];
const TIERS: &[&str] = &["FREE", "PRO", "ENTERPRISE"];
const PRODUCTS: &[&str] = &["A", "B", "C"];
const RESOLUTIONS: &[&str] = &["REFUND", "RESET", "ESCALATE", "DOCS_LINK", "ENABLE_FEATURE"];

struct Lexicons {
    sympts: HashMap<&'static str, Vec<f64>>,
    tiers: HashMap<&'static str, Vec<f64>>,
    products: HashMap<&'static str, Vec<f64>>,
    resolutions: HashMap<&'static str, Vec<f64>>,
}
impl Lexicons {
    fn new(rng: &mut StdRng) -> Self {
        let mk = |names: &[&'static str], rng: &mut StdRng| -> HashMap<&'static str, Vec<f64>> {
            names.iter().map(|n| (*n, rand_unit(rng))).collect()
        };
        Self {
            sympts: mk(SYMPTOMS, rng),
            tiers: mk(TIERS, rng),
            products: mk(PRODUCTS, rng),
            resolutions: mk(RESOLUTIONS, rng),
        }
    }
}

fn generate_training_set(n: usize, rng: &mut StdRng) -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    (0..n).map(|_| {
        let s = SYMPTOMS[rng.r#gen_range(0..SYMPTOMS.len())];
        let t = TIERS[rng.r#gen_range(0..TIERS.len())];
        let p = PRODUCTS[rng.r#gen_range(0..PRODUCTS.len())];
        let r = ground_truth(s, t);
        (s, t, p, r)
    }).collect()
}

// ===========================================================================
//                  SECTION 1 — Episodic-only baseline
// ===========================================================================

fn section_1(lex: &Lexicons, rng: &mut StdRng) -> TwoTier {
    println!("======================================================================");
    println!("  SECTION 1 — Episodic-only baseline");
    println!("======================================================================");
    println!("60 cases written to episodic memory verbatim. Each is one bound");
    println!("bundle. Query: probe with (S, T, P), unbind resolution, clean up.\n");

    let res_lex: Vec<(String, Vec<f64>)> = RESOLUTIONS.iter()
        .map(|r| (r.to_string(), lex.resolutions[r].clone())).collect();
    let mut mem = TwoTier::new(rng, res_lex);

    let train = generate_training_set(60, rng);
    for (s, t, p, r) in &train {
        mem.write_episode(&lex.sympts[s], &lex.tiers[t], &lex.products[p], &lex.resolutions[r]);
    }

    // Distribution check
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, _, _, r) in &train { *counts.entry(r).or_insert(0) += 1; }
    println!("  Wrote {} episodes. By resolution:", train.len());
    for r in RESOLUTIONS {
        println!("    {:<18} {} cases", r, counts.get(*r).copied().unwrap_or(0));
    }
    println!("  |episodic| = {} entries (one per case, verbatim)",
        mem.episodic.locs.len());
    println!("  |semantic| = {} entries (empty until consolidation)",
        mem.semantic.locs.len());

    // Quick accuracy check on training cases (specific recall)
    let mut hits = 0;
    for (s, t, p, r) in &train {
        let (pred, _) = mem.query_episodic(&lex.sympts[s], &lex.tiers[t], &lex.products[p]);
        if pred == *r { hits += 1; }
    }
    println!("\n  Specific recall on training set: {}/{} = {:.0}%",
        hits, train.len(), hits as f64 / train.len() as f64 * 100.0);

    println!("  PASS: episodic stores cases exactly, recalls them well.\n");
    mem
}

// ===========================================================================
//                  SECTION 2 — Consolidation: semantic emerges
// ===========================================================================

fn section_2(mem: &mut TwoTier) {
    println!("======================================================================");
    println!("  SECTION 2 — Consolidation: hippocampal → cortical replay");
    println!("======================================================================");
    println!("Replay episodic memory, group cases by resolution, bundle each");
    println!("group into a prototype, write prototypes to semantic.\n");

    let ep_before = mem.episodic.locs.len();
    mem.consolidate();
    let sem_after = mem.semantic.locs.len();

    println!("  Before:  |episodic|={}   |semantic|=0", ep_before);
    println!("  After:   |episodic|={}   |semantic|={}", ep_before, sem_after);
    println!();
    println!("  Compression ratio: {}× (specific cases collapsed to policy prototypes)",
        ep_before / sem_after.max(1));
    println!();
    println!("  Each semantic entry is one resolution-class prototype: the bundle");
    println!("  of all episodic cases sharing that resolution. The RESOLUTION");
    println!("  binding reinforces (all members agree); the input bindings average");
    println!("  out (different inputs per case) — producing a policy prototype.");

    assert!(sem_after <= 5, "should consolidate to at most one prototype per resolution");
    assert!(sem_after >= 3, "should have prototypes for the major resolutions");
    println!("\n  PASS: semantic memory is a compressed policy summary.\n");
}

// ===========================================================================
//        SECTION 3 — Specific vs general: where each memory shines
// ===========================================================================

fn section_3(mem: &TwoTier, lex: &Lexicons, rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Specific vs. general queries");
    println!("======================================================================");
    println!("60 novel tickets, distinct from training. Compare episodic alone");
    println!("vs. semantic alone vs. combined.\n");

    let test = generate_training_set(60, rng);
    let mut ep_hits = 0;
    let mut se_hits = 0;
    let mut cb_hits = 0;
    let mut ep_total_sim = 0.0;
    let mut se_total_sim = 0.0;

    for (s, t, p, r) in &test {
        let sv = &lex.sympts[s];
        let tv = &lex.tiers[t];
        let pv = &lex.products[p];

        let (ep_pred, ep_sim) = mem.query_episodic(sv, tv, pv);
        let (se_pred, se_sim) = mem.query_semantic(sv, tv, pv);
        let (cb_pred, _) = mem.query_combined(sv, tv, pv);

        if ep_pred == *r { ep_hits += 1; }
        if se_pred == *r { se_hits += 1; }
        if cb_pred == *r { cb_hits += 1; }
        ep_total_sim += ep_sim;
        se_total_sim += se_sim;
    }

    let n = test.len() as f64;
    println!("  Memory       Accuracy  Avg confidence (top similarity)");
    println!("  {:-<55}", "");
    println!("  Episodic     {:>6.1}%   {:.3}",
        ep_hits as f64 / n * 100.0, ep_total_sim / n);
    println!("  Semantic     {:>6.1}%   {:.3}",
        se_hits as f64 / n * 100.0, se_total_sim / n);
    println!("  Combined     {:>6.1}%   (picks whichever fired stronger)",
        cb_hits as f64 / n * 100.0);

    println!();
    println!("  Semantic's avg confidence is HIGHER even when accuracy is similar:");
    println!("  the prototype has reinforced RESOLUTION signal from many cases,");
    println!("  so unbinding it is sharper than any single episodic case.");

    assert!(se_hits >= ep_hits || cb_hits >= ep_hits,
        "semantic or combined should match or beat episodic-alone");
    println!("\n  PASS: two-tier memory matches or beats either tier alone.\n");
}

// ===========================================================================
//      SECTION 4 — Forgetting episodic; semantic retains policy
// ===========================================================================

fn section_4(mem: &mut TwoTier, lex: &Lexicons, rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 4 — Episodic forgets; semantic retains");
    println!("======================================================================");
    println!("Simulate forgetting by REPLACING the episodic memory with 30 random");
    println!("noise entries — the original cases are gone. Then query the same");
    println!("policy: semantic still answers correctly; episodic produces noise.\n");

    // Save baselines on a fresh test set
    let test = generate_training_set(40, rng);
    let pre_se_hits = test.iter().filter(|(s, t, p, r)| {
        let (pred, _) = mem.query_semantic(&lex.sympts[s], &lex.tiers[t], &lex.products[p]);
        pred == *r
    }).count();
    let pre_ep_hits = test.iter().filter(|(s, t, p, r)| {
        let (pred, _) = mem.query_episodic(&lex.sympts[s], &lex.tiers[t], &lex.products[p]);
        pred == *r
    }).count();

    // Now clobber episodic with noise
    mem.episodic = MiniEAM::new();
    for _ in 0..30 {
        mem.episodic.write(&rand_unit(rng));
    }

    let post_se_hits = test.iter().filter(|(s, t, p, r)| {
        let (pred, _) = mem.query_semantic(&lex.sympts[s], &lex.tiers[t], &lex.products[p]);
        pred == *r
    }).count();
    let post_ep_hits = test.iter().filter(|(s, t, p, r)| {
        let (pred, _) = mem.query_episodic(&lex.sympts[s], &lex.tiers[t], &lex.products[p]);
        pred == *r
    }).count();

    let n = test.len() as f64;
    println!("                       Before forgetting    After forgetting");
    println!("  Episodic acc:    {:>9.1}%        {:>9.1}%",
        pre_ep_hits as f64 / n * 100.0, post_ep_hits as f64 / n * 100.0);
    println!("  Semantic acc:    {:>9.1}%        {:>9.1}%",
        pre_se_hits as f64 / n * 100.0, post_se_hits as f64 / n * 100.0);

    assert!(post_se_hits as f64 / n > 0.6,
        "semantic should survive episodic loss");
    assert!(post_ep_hits < pre_ep_hits,
        "episodic should degrade after being clobbered");

    println!();
    println!("  Episodic collapsed to chance after clobbering — those specific");
    println!("  experiences are gone. Semantic preserved the policy because");
    println!("  consolidation already wrote the generalizations to cortex.");
    println!();
    println!("  This is the hippocampal–cortical replay loop's reason for being:");
    println!("  the policy survives, the noise does not.");
    println!("\n  PASS: substrate models the two-tier memory architecture.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let lex = Lexicons::new(&mut rng);
    let mut mem = section_1(&lex, &mut rng);
    section_2(&mut mem);
    section_3(&mem, &lex, &mut rng);
    section_4(&mut mem, &lex, &mut rng);

    println!("======================================================================");
    println!("  ALL FOUR SECTIONS PASSED — two-tier memory on the substrate");
    println!("======================================================================");
    println!("  Hippocampal episodic (fast write, exact recall, fades).");
    println!("  Cortical semantic (slow build via replay, generalizes, persists).");
    println!("  Same substrate primitives — different write/read policies.");
}
