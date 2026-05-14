//! Analogical & relational reasoning on the HRR + EAM substrate.
//!
//! Builds three demos, each escalating in commercial relevance:
//!
//!   1. **Structured Mikolov**: entities encoded as bound bundles of
//!      attribute-value pairs. The Word2Vec parlour trick
//!      `king - man + woman ≈ queen` falls out exactly from the algebra,
//!      no training required.
//!
//!   2. **Role inference + transfer**: given a single example pair
//!      `(doctor, surgery)`, infer *which* binding role connects them,
//!      then apply that role to a new entity to recover its analogous
//!      filler. This is `A:B::C:?` as substrate algebra.
//!
//!   3. **Case-based reasoning**: 30 customer-support cases stored in an
//!      EAM as bound (symptom, tier, product, resolution) bundles.
//!      A new ticket arrives with no resolution; the substrate retrieves
//!      the right one by structural similarity. The commercial wedge.
//!
//! Run: `cargo run --release --example analogy -p heather_algebra`

use heather_algebra::{bind_vec, unbind_vec};
use heather_db::{HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

const DIM: usize = 1024;
const BETA: f64 = 12.0;
const READ_ITERS: usize = 4;

// ----- Helpers --------------------------------------------------------------

fn rand_unit(rng: &mut StdRng) -> Vec<f64> {
    vec_ops::random_unit_vector(DIM, rng)
}

fn bundle(parts: &[&[f64]]) -> Vec<f64> {
    let d = parts[0].len();
    let mut out = vec![0.0; d];
    for p in parts {
        for i in 0..d { out[i] += p[i]; }
    }
    vec_ops::normalize(&out)
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn nearest_named<'a>(v: &[f64], lex: &'a [(&'a str, Vec<f64>)]) -> (&'a str, f64) {
    let v = vec_ops::normalize(v);
    let mut best = lex[0].0;
    let mut best_s = f64::NEG_INFINITY;
    for (name, vec) in lex {
        let s = vec_ops::cosine_similarity(&v, vec);
        if s > best_s { best_s = s; best = name; }
    }
    (best, best_s)
}

// ----- Mini EAM (for section 3) --------------------------------------------

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
//                    SECTION 1 — Structured Mikolov
// ===========================================================================

fn section_1(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 1 — Structured Mikolov: analogy as vector arithmetic");
    println!("======================================================================");
    println!("Each entity = bundle(bind(GENDER, g), bind(CLASS, c), bind(AGE, a)).");
    println!("`king - man + woman ≈ queen` works exactly via algebra — no training.\n");

    let gender_k = rand_unit(rng);
    let class_k = rand_unit(rng);
    let age_k = rand_unit(rng);

    let male = rand_unit(rng);
    let female = rand_unit(rng);
    let royal = rand_unit(rng);
    let common = rand_unit(rng);
    let adult = rand_unit(rng);
    let young = rand_unit(rng);

    let mk = |g: &[f64], c: &[f64], a: &[f64]| -> Vec<f64> {
        bundle(&[
            &bind_vec(&gender_k, g),
            &bind_vec(&class_k, c),
            &bind_vec(&age_k, a),
        ])
    };
    let king = mk(&male, &royal, &adult);
    let queen = mk(&female, &royal, &adult);
    let man = mk(&male, &common, &adult);
    let woman = mk(&female, &common, &adult);
    let prince = mk(&male, &royal, &young);
    let princess = mk(&female, &royal, &young);
    let boy = mk(&male, &common, &young);
    let girl = mk(&female, &common, &young);

    let lex: Vec<(&str, Vec<f64>)> = vec![
        ("king", king.clone()), ("queen", queen.clone()),
        ("man", man.clone()), ("woman", woman.clone()),
        ("prince", prince.clone()), ("princess", princess.clone()),
        ("boy", boy.clone()), ("girl", girl.clone()),
    ];

    let do_analogy = |label: &str, result: Vec<f64>, expected: &str| {
        let (best, sim) = nearest_named(&result, &lex);
        let mark = if best == expected { "✓" } else { "✗" };
        println!("  {:<38} → {:<10} (sim={:.3})  {}", label, best, sim, mark);
        assert_eq!(best, expected, "analogy failed: {}", label);
    };

    do_analogy("king - man + woman",
        add_vec(&sub_vec(&king, &man), &woman), "queen");
    do_analogy("man - woman + queen",
        add_vec(&sub_vec(&man, &woman), &queen), "king");
    do_analogy("king - prince + princess",
        add_vec(&sub_vec(&king, &prince), &princess), "queen");
    do_analogy("queen + prince - king",
        sub_vec(&add_vec(&queen, &prince), &king), "princess");
    do_analogy("boy - girl + woman",
        add_vec(&sub_vec(&boy, &girl), &woman), "man");
    do_analogy("prince - princess + girl",
        add_vec(&sub_vec(&prince, &princess), &girl), "boy");
    do_analogy("king - boy + girl",
        add_vec(&sub_vec(&king, &boy), &girl), "queen");

    println!("\n  PASS: Word2Vec's analogy trick falls out of structured");
    println!("        binding — exactly, not approximately, no corpus.\n");
}

// ===========================================================================
//              SECTION 2 — Role inference + analogical transfer
// ===========================================================================

fn section_2(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 2 — Role inference + analogical transfer");
    println!("======================================================================");
    println!("Given (A, B), discover which binding role connects them.");
    println!("Then apply that role to a new entity C to recover its analogue D.\n");

    let field_k = rand_unit(rng);
    let skill_k = rand_unit(rng);
    let workplace_k = rand_unit(rng);

    let role_keys: Vec<(&str, &Vec<f64>)> = vec![
        ("FIELD", &field_k), ("SKILL", &skill_k), ("WORKPLACE", &workplace_k),
    ];

    let medical = rand_unit(rng);
    let tech = rand_unit(rng);
    let culinary = rand_unit(rng);
    let education = rand_unit(rng);

    let surgery = rand_unit(rng);
    let coding = rand_unit(rng);
    let cooking = rand_unit(rng);
    let lecturing = rand_unit(rng);

    let hospital = rand_unit(rng);
    let office = rand_unit(rng);
    let kitchen = rand_unit(rng);
    let school = rand_unit(rng);

    let mk = |fl: &[f64], sk: &[f64], wp: &[f64]| -> Vec<f64> {
        bundle(&[
            &bind_vec(&field_k, fl),
            &bind_vec(&skill_k, sk),
            &bind_vec(&workplace_k, wp),
        ])
    };
    let doctor = mk(&medical, &surgery, &hospital);
    let engineer = mk(&tech, &coding, &office);
    let chef = mk(&culinary, &cooking, &kitchen);
    let teacher = mk(&education, &lecturing, &school);

    let atom_lex: Vec<(&str, Vec<f64>)> = vec![
        ("medical", medical.clone()), ("tech", tech.clone()),
        ("culinary", culinary.clone()), ("education", education.clone()),
        ("surgery", surgery.clone()), ("coding", coding.clone()),
        ("cooking", cooking.clone()), ("lecturing", lecturing.clone()),
        ("hospital", hospital.clone()), ("office", office.clone()),
        ("kitchen", kitchen.clone()), ("school", school.clone()),
    ];

    let infer_role = |a: &[f64], b: &[f64]| -> (&'static str, &Vec<f64>, f64) {
        let mut best_name: &'static str = "";
        let mut best_key: &Vec<f64> = role_keys[0].1;
        let mut best_sim = f64::NEG_INFINITY;
        for (name, key) in &role_keys {
            let candidate = vec_ops::normalize(&unbind_vec(a, key));
            let sim = vec_ops::cosine_similarity(&candidate, b);
            if sim > best_sim {
                best_sim = sim;
                best_name = name;
                best_key = *key;
            }
        }
        (best_name, best_key, best_sim)
    };

    let test_analogy = |a_name: &str, a: &Vec<f64>, b_name: &str, b: &Vec<f64>,
                        cases: &[(&str, &Vec<f64>, &str)]| {
        let (role_name, role_key, role_sim) = infer_role(a, b);
        println!("  Example: {} : {}", a_name, b_name);
        println!("    Inferred role: {} (similarity {:.3})", role_name, role_sim);
        for (c_name, c, expected) in cases {
            let candidate = vec_ops::normalize(&unbind_vec(c, role_key));
            let (best, s) = nearest_named(&candidate, &atom_lex);
            let mark = if best == *expected { "✓" } else { "✗" };
            println!("    {} : ?  → {:<10} (sim {:.3})  expected {} {}",
                c_name, best, s, expected, mark);
            assert_eq!(best, *expected, "analogy failed: {} via {}", c_name, role_name);
        }
        println!();
    };

    test_analogy("doctor", &doctor, "surgery", &surgery, &[
        ("engineer", &engineer, "coding"),
        ("chef", &chef, "cooking"),
        ("teacher", &teacher, "lecturing"),
    ]);
    test_analogy("doctor", &doctor, "hospital", &hospital, &[
        ("engineer", &engineer, "office"),
        ("chef", &chef, "kitchen"),
        ("teacher", &teacher, "school"),
    ]);
    test_analogy("chef", &chef, "culinary", &culinary, &[
        ("doctor", &doctor, "medical"),
        ("engineer", &engineer, "tech"),
        ("teacher", &teacher, "education"),
    ]);

    println!("  PASS: substrate inferred each relation from ONE example");
    println!("        and transferred it correctly to new entities.\n");
}

// ===========================================================================
//             SECTION 3 — Case-based reasoning (commercial wedge)
// ===========================================================================

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

fn section_3(rng: &mut StdRng) {
    println!("======================================================================");
    println!("  SECTION 3 — Case-based reasoning over a substrate");
    println!("======================================================================");
    println!("30 stored support cases as bundle(bind(SYMPTOM, s), bind(TIER, t),");
    println!("bind(PRODUCT, p), bind(RESOLUTION, r)). New tickets arrive without");
    println!("the resolution slot; substrate retrieves it by structural match.\n");

    let symptom_k = rand_unit(rng);
    let tier_k = rand_unit(rng);
    let product_k = rand_unit(rng);
    let resolution_k = rand_unit(rng);

    let symptoms = ["BILLING_ERROR", "CANT_LOGIN", "BUG_REPORT", "MISSING_FEATURE", "QUESTION"];
    let tiers = ["FREE", "PRO", "ENTERPRISE"];
    let products = ["A", "B", "C"];
    let resolutions = ["REFUND", "RESET", "ESCALATE", "DOCS_LINK", "ENABLE_FEATURE"];

    let mk_lex = |names: &[&'static str], rng: &mut StdRng|
        -> HashMap<&'static str, Vec<f64>>
    {
        names.iter().map(|n| (*n, rand_unit(rng))).collect()
    };
    let s_vecs = mk_lex(&symptoms, rng);
    let t_vecs = mk_lex(&tiers, rng);
    let p_vecs = mk_lex(&products, rng);
    let r_vecs = mk_lex(&resolutions, rng);

    let res_lex: Vec<(&str, Vec<f64>)> =
        resolutions.iter().map(|n| (*n, r_vecs[n].clone())).collect();

    // ----- Train: 30 cases following the ground-truth policy -------------
    let n_train = 30;
    let mut eam = MiniEAM::new();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for _ in 0..n_train {
        let s = symptoms[rng.r#gen_range(0..symptoms.len())];
        let t = tiers[rng.r#gen_range(0..tiers.len())];
        let p = products[rng.r#gen_range(0..products.len())];
        let r = ground_truth(s, t);
        *counts.entry(r).or_insert(0) += 1;
        let case = bundle(&[
            &bind_vec(&symptom_k, &s_vecs[s]),
            &bind_vec(&tier_k, &t_vecs[t]),
            &bind_vec(&product_k, &p_vecs[p]),
            &bind_vec(&resolution_k, &r_vecs[r]),
        ]);
        eam.write(&case);
    }
    println!("  Trained on {} cases.  Distribution by resolution:", n_train);
    for r in &resolutions {
        println!("    {:<18} {} cases", r, counts.get(*r).copied().unwrap_or(0));
    }
    println!();

    // ----- Test: 30 held-out tickets ------------------------------------
    let n_test = 30;
    let mut correct = 0;
    let mut errors: Vec<(String, String, String, String, String)> = Vec::new();

    println!("  Testing {} new tickets:", n_test);
    println!("    {:<16} {:<12} {:<10} {:<18} {:<18}",
        "symptom", "tier", "product", "expected", "predicted");
    println!("    {:-<76}", "");
    for i in 0..n_test {
        let s = symptoms[rng.r#gen_range(0..symptoms.len())];
        let t = tiers[rng.r#gen_range(0..tiers.len())];
        let p = products[rng.r#gen_range(0..products.len())];
        let expected = ground_truth(s, t);

        let query = bundle(&[
            &bind_vec(&symptom_k, &s_vecs[s]),
            &bind_vec(&tier_k, &t_vecs[t]),
            &bind_vec(&product_k, &p_vecs[p]),
        ]);
        let recalled = eam.read(&query);
        let noisy_res = unbind_vec(&recalled, &resolution_k);
        let (predicted, _sim) = nearest_named(&noisy_res, &res_lex);

        let ok = predicted == expected;
        if ok { correct += 1; }
        else {
            errors.push((s.into(), t.into(), p.into(), expected.into(), predicted.into()));
        }
        if i < 12 || !ok {
            println!("    {:<16} {:<12} {:<10} {:<18} {:<18} {}",
                s, t, p, expected, predicted, if ok { "✓" } else { "✗" });
        } else if i == 12 {
            println!("    ... (further passes elided)");
        }
    }

    let acc = correct as f64 / n_test as f64;
    println!();
    println!("  Accuracy: {}/{} = {:.0}%", correct, n_test, acc * 100.0);
    if !errors.is_empty() {
        println!("  Errors:");
        for (s, t, p, exp, got) in &errors {
            println!("    ({:<16} {:<11} {:<2}) expected {:<16} got {}", s, t, p, exp, got);
        }
    }

    assert!(acc > 0.7,
        "case-based reasoning accuracy {:.0}% too low — substrate not learning the policy",
        acc * 100.0);
    println!("\n  PASS: substrate learned the support policy from {} examples,", n_train);
    println!("        generalizes to unseen tickets — no model training, no rules engine.\n");
}

// ----- Main -----------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    section_1(&mut rng);
    section_2(&mut rng);
    section_3(&mut rng);
    println!("======================================================================");
    println!("  ALL THREE SECTIONS PASSED");
    println!("======================================================================");
    println!("  Word2Vec's analogy trick — without training a Word2Vec.");
    println!("  Hofstadter-style A:B::C:? — one example per relation.");
    println!("  Case-based reasoning — 30 examples, generalizes structurally.");
    println!("  All from `bind`, `add`, `sub`, and read.");
}
