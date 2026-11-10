//! Seeded before/after timing and output fingerprints for PR #95.
//! RAYON_NUM_THREADS=4 cargo run --release --locked -p heather_algebra --example release_merge_probe
use heather_algebra::{EAMSnapshot, ops};
use heather_db::{EAMConfig, HardLocation, LocationId, vec_ops};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn snapshot(n: usize, tight: bool, seed: u64) -> EAMSnapshot {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centre: Vec<f64> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let axes: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let locations = (0..n)
        .map(|i| {
            let mut v = if tight {
                centre.clone()
            } else {
                vec![0.0; 128]
            };
            if tight {
                for axis in &axes {
                    vec_ops::add_scaled(&mut v, axis, rng.gen_range(-0.2..0.2));
                }
            }
            for x in &mut v {
                *x += rng.gen_range(-1.0..1.0) * if tight { 0.01 } else { 1.0 };
            }
            let mut loc = HardLocation::new(LocationId(i as u64), vec_ops::normalize(&v));
            loc.counter = loc.address.clone();
            loc.write_count = 1.0;
            loc
        })
        .collect();
    EAMSnapshot::new(locations, EAMConfig::new(128).unwrap()).unwrap()
}
fn hash(snap: &EAMSnapshot) -> String {
    let mut digest = Sha256::new();
    for loc in &snap.locations {
        digest.update(loc.id.0.to_le_bytes());
        for value in loc
            .address
            .iter()
            .chain(&loc.counter)
            .chain(std::iter::once(&loc.write_count))
        {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}
fn main() {
    for tight in [true, false] {
        for n in [128, 512] {
            let a = snapshot(n, tight, 7);
            let b = snapshot(n, tight, 8);
            for operation in ["add", "sub"] {
                let run = || {
                    if operation == "add" {
                        ops::add_with_limit(&a, &b, 8).unwrap()
                    } else {
                        ops::sub_with_limit(&a, &b, 8).unwrap()
                    }
                };
                let warmup = run();
                let fingerprint = hash(&warmup);
                let mut ns = Vec::new();
                for _ in 0..7 {
                    let start = Instant::now();
                    let result = run();
                    ns.push(start.elapsed().as_nanos());
                    assert_eq!(
                        hash(&result),
                        fingerprint,
                        "output varied across repetitions"
                    );
                }
                println!(
                    "{{\"operation\":\"{}\",\"shape\":\"{}\",\"dimension\":128,\"locations_per_operand\":{},\"cross_k\":8,\"input_a_sha256\":\"{}\",\"input_b_sha256\":\"{}\",\"output_sha256\":\"{}\",\"output_locations\":{},\"nanoseconds\":{:?}}}",
                    operation,
                    if tight { "tight" } else { "uniform" },
                    n,
                    hash(&a),
                    hash(&b),
                    fingerprint,
                    warmup.locations.len(),
                    ns
                );
            }
        }
    }
}
